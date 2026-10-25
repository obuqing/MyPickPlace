"""Success-gated distance curriculum with separate unchanged full-task evaluation."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import envs
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from algorithms.carry_validation import CarryValidation
from train_carry import Progress

STAGES = [.05, .10, .15, None]


def promotion_streak(successes, attempted, streak):
    return streak + 1 if attempted > 0 and successes / attempted >= .8 else 0


def resume_cursor(summary, history):
    stage = int(summary['final_stage'])
    last = history[-1]
    if not 0 <= stage < len(STAGES) or last['step'] != summary['steps']:
        raise ValueError('Resume metadata does not match final checkpoint')
    streak = int(last['consecutive_passes']) if last['stage'] == stage else 0
    return int(summary['steps']), stage, streak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('runs/carry_curriculum_20000'))
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--resume', type=Path, help='Completed curriculum run; steps is additional budget')
    parser.add_argument('--seed', type=int, default=0, help='New rollout RNG seed; use a new seed when resuming')
    args = parser.parse_args()
    if args.steps < 2000 or args.steps % 2000:
        parser.error('Steps must be a positive multiple of 2000')
    start_step, stage, streak = 0, 0, 0
    source = args.resume.resolve() if args.resume else None
    if source:
        previous = json.loads((source/'summary.json').read_text())
        prior_history = json.loads((source/'curriculum_history.json').read_text())
        start_step, stage, streak = resume_cursor(previous, prior_history)
        for name in ('model.zip', 'replay_buffer.pkl'):
            if not (source/'training'/name).is_file():
                parser.error(f'Missing resume file: {name}')
    end_step = start_step + args.steps
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    training = root / 'training'
    training.mkdir()
    project = Path(__file__).resolve().parents[1]
    config = dict(stages=STAGES, promotion_rate=.8, consecutive_passes=2,
                  interval=2000, steps=end_step, additional_steps=args.steps, start_step=start_step,
                  fresh_model=source is None, resume_source=str(source) if source else None, training_seed=args.seed,
                  curriculum_validation_seeds=list(range(4000, 4005)),
                  full_validation_seeds=list(range(2000, 2005)), test_seeds=list(range(3000, 3010)),
                  height=.18, tolerance=.02, reward_changed=False,
                  error_convention='target_minus_object; negative Z means object too high')
    (root/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    hashes = {}
    for name in ('envs/carry_skill.py', 'envs/pick_place.py', 'algorithms/scripted_pick_place.py',
                 'algorithms/carry_evaluation.py', 'algorithms/carry_validation.py',
                 'scripts/train_carry_curriculum.py', 'scripts/train_carry.py', 'scripts/evaluate_carry.py'):
        data = (project/name).read_bytes()
        dest = root/'frozen'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    (root/'manifest.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
    state = dict(status='running', phase='initializing', steps=start_step, target_steps=end_step,
                 stage=stage, max_target_distance=STAGES[stage], started=datetime.now().isoformat())
    def save():
        (root/'status.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    def evaluate(checkpoint, destination, seed, episodes, radius=None):
        destination.mkdir(parents=True)
        command = [sys.executable, '-u', str(project/'scripts/evaluate_carry.py'), '--policy', 'sac',
                   '--model', str(checkpoint), '--output', str(destination), '--seed', str(seed), '--episodes', str(episodes)]
        if radius is not None:
            command += ['--max-target-distance', str(radius)]
        with (destination/'evaluation.log').open('w', encoding='utf-8') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        return json.loads((destination/'summary.json').read_text())['summary']['all']
    save()
    torch.set_num_threads(1)
    environment = Monitor(gym.make('MyCarrySkill-v1', max_target_distance=STAGES[stage],
                                  audit_path=training/'initializations.jsonl'), str(training/'monitor.csv'),
                          info_keywords=('move_success', 'reason', 'color'))
    callback = Progress(training)
    history = []
    begin = time.perf_counter()
    try:
        if source:
            model = SAC.load(source/'training/model.zip', env=environment, device='cpu', force_reset=True)
            model.load_replay_buffer(source/'training/replay_buffer.pkl')
            assert model.num_timesteps == start_step and model._n_updates == previous['updates']
            assert model.actor.optimizer.state and model.critic.optimizer.state
            assert model.replay_buffer.size() > 0
            model.set_random_seed(args.seed)
            resume_info = dict(source=str(source), steps=model.num_timesteps, updates=model._n_updates,
                               replay_size=model.replay_buffer.size(), stage=stage, consecutive_passes=streak,
                               optimizer_restored=True, new_rollout_seed=args.seed, exact_physics_continuation=False)
            for name in ('model.zip', 'replay_buffer.pkl'):
                resume_info[name+'_sha256'] = hashlib.sha256((source/'training'/name).read_bytes()).hexdigest()
            (root/'resume.json').write_text(json.dumps(resume_info, indent=2), encoding='utf-8')
            print(resume_info, flush=True)
        else:
            model = SAC('MultiInputPolicy', environment, seed=args.seed, device='cpu', policy_kwargs={'net_arch':[128,128]},
                        learning_starts=100, buffer_size=50000, batch_size=128, learning_rate=3e-4,
                        gamma=.98, train_freq=1, gradient_steps=1, verbose=0)
        validator = CarryValidation(training, episodes=5, seed=2000)
        state['phase'] = 'baseline_full_validation'
        save()
        validator.run(model)
        while model.num_timesteps < end_step:
            state.update(phase='training', stage=stage, max_target_distance=STAGES[stage], steps=model.num_timesteps)
            save()
            model.learn(total_timesteps=2000, callback=callback, reset_num_timesteps=False)
            step = model.num_timesteps
            checkpoint = training/f'curriculum_step_{step}.zip'
            model.save(checkpoint)
            state.update(phase='curriculum_validation', steps=step)
            save()
            metrics = evaluate(checkpoint, root/'curriculum_validation'/f'step_{step}', 4000, 5, STAGES[stage])
            streak = promotion_streak(metrics['move_successes'], metrics['attempted'], streak)
            promoted = streak >= 2 and stage < len(STAGES)-1
            history.append(dict(step=step, stage=stage, max_target_distance=STAGES[stage], metrics=metrics,
                                consecutive_passes=streak, promoted=promoted))
            (root/'curriculum_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
            if step % 4000 == 0 or step == end_step:
                state['phase'] = 'full_task_validation'
                save()
                validator.run(model)
            if promoted and step < end_step:
                stage += 1
                streak = 0
                # Keep replay/optimizer, reset rollout to avoid changing a goal halfway through an episode.
                environment.unwrapped.max_target_distance = STAGES[stage]
                model._last_obs = None
                print(dict(promoted_to_stage=stage, max_target_distance=STAGES[stage]), flush=True)
        model.save(training/'model.zip')
        model.save_replay_buffer(training/'replay_buffer.pkl')
        (training/'progress.json').write_text(json.dumps(dict(status='training_complete', steps=model.num_timesteps,
            updates=model._n_updates, completed_episodes=len(callback.episodes),
            training_carry_successes=sum(e['move_success'] for e in callback.episodes)), indent=2), encoding='utf-8')
        results = {}
        for label, name in [('best','best_model.zip'), ('last','model.zip')]:
            state['phase'] = 'test_'+label
            save()
            results[label] = evaluate(training/name, root/label, 3000, 10)
        train_seeds = {json.loads(line)['seed'] for line in (training/'initializations.jsonl').read_text().splitlines()}
        assert not train_seeds & set(config['curriculum_validation_seeds']+config['full_validation_seeds']+config['test_seeds'])
        restored = SAC.load(training/'model.zip', device='cpu')
        assert np.allclose(model.predict(model._last_obs, deterministic=True)[0],
                           restored.predict(model._last_obs, deterministic=True)[0], atol=1e-6, rtol=0)
        assert all(torch.isfinite(p).all() for p in model.policy.parameters())
        summary = dict(steps=model.num_timesteps, updates=model._n_updates, final_stage=stage,
                       start_step=start_step, additional_steps=args.steps,
                       best_step=validator.best_step, tests=results, elapsed_seconds=time.perf_counter()-begin)
        (root/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        diagnostics = []
        for item in history:
            rows = json.loads((root/'curriculum_validation'/f"step_{item['step']}"/'results.json').read_text())
            valid = [r for r in rows if r['initialization_success']]
            diagnostics.append(dict(step=item['step'], radius=item['max_target_distance'],
                mean_final_signed_z=item['metrics']['mean_final_error_xyz'][2],
                mean_final_absolute_z=item['metrics']['mean_final_abs_error_xyz'][2],
                mean_peak_height_overshoot=float(np.mean([max([0.]+[-t['error_xyz'][2] for t in r['trace']]) for r in valid])),
                mean_z_action=item['metrics']['mean_action_xyz'][2]))
        (root/'height_diagnostics.json').write_text(json.dumps(diagnostics, indent=2), encoding='utf-8')
        fig, axes = plt.subplots(1, 2, figsize=(12,4), constrained_layout=True)
        for key in ('mean_final_signed_z','mean_final_absolute_z','mean_peak_height_overshoot'):
            axes[0].plot([d['step'] for d in diagnostics], [d[key] for d in diagnostics], 'o-', label=key)
        axes[0].axhline(0, color='black', linewidth=.5)
        axes[0].set(ylabel='m', title='Curriculum height error (target - object)')
        axes[1].plot([d['step'] for d in diagnostics], [d['mean_z_action'] for d in diagnostics], 'o-')
        axes[1].set(ylabel='normalized action', title='Mean vertical action')
        for ax in axes:
            ax.set_xlabel('Training steps')
            ax.grid(alpha=.25)
        axes[0].legend(fontsize=7)
        fig.savefig(root/'height_diagnostics.png', dpi=160)
        plt.close(fig)
        subprocess.run([sys.executable, str(project/'scripts/plot_carry_training.py'), str(training)], check=True)
        lines = ['# Distance curriculum experiment', '',
                 ('Resumed SAC model, optimizer and replay buffer.' if source else 'Fresh SAC model;') +
                 ' Unchanged reward, height, tolerance and action scale. Distance stages: 5cm, 10cm, 15cm, full.',
                 'Promotion requires two consecutive deterministic validations >=80% (10 episodes each).',
                 'Short waypoint success is not full placement. Best checkpoint uses full-task validation only.',
                 'Signed Z = target height minus object height: negative means too high.', '',
                 '| Step | Radius (m) | Waypoint carry | Final distance (m) | Signed Z (m) | Absolute Z (m) | Promoted |',
                 '| --- | --- | --- | --- | --- | --- | --- |']
        for item in history:
            m = item['metrics']
            lines.append(f"| {item['step']} | {item['max_target_distance']} | {m['move_successes']}/{m['attempted']} | {m['mean_final_distance']} | {m['mean_final_error_xyz'][2]} | {m['mean_final_abs_error_xyz'][2]} | {item['promoted']} |")
        lines += ['', f"Best full-task checkpoint: {validator.best_step}; final stage: {stage}.", '',
                  '| Full task test | Carry | Placement | Distance (m) | Signed Z (m) |', '| --- | --- | --- | --- | --- |']
        for label,m in results.items():
            lines.append(f"| {label} | {m['move_successes']}/20 | {m['placement_successes']}/20 | {m['mean_final_distance']} | {m['mean_final_error_xyz'][2]} |")
        lines += ['', '## Next improvements', '']
        if stage == 0:
            lines.append('The 5cm stage did not pass twice; inspect height drift and action/reward balance before extending the distance.')
        elif results['best']['move_successes'] == 0:
            lines.append('Local skills have not transferred to full-distance success; inspect the first failed stage and full-task XYZ trajectories before increasing budget.')
        else:
            lines.append('Repeat with multiple training seeds and untouched test scenes, and inspect placement handoff failures.')
        (root/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        state.update(status='complete', phase='complete', final_stage=stage, best_step=validator.best_step,
                     finished=datetime.now().isoformat())
    except Exception as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}', finished=datetime.now().isoformat())
        raise
    finally:
        callback.stream.close()
        environment.close()
        save()


if __name__ == '__main__':
    main()
