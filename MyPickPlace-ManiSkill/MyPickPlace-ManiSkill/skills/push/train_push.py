"""Auditable SAC curriculum, validation-selected checkpoints and fixed test suite."""
import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from .push_skill import PushSkill, CONTRACT


def save(path, value):
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def evaluate(model, seeds, distances=(.05,.10,.15), video=None):
    # Validation must not reset the training RNG via ManiSkill reset(seed).
    states = (random.getstate(),np.random.get_state(),torch.get_rng_state())
    rows = []
    try:
        for distance in distances:
            env = PushSkill(distance_m=distance,render_mode='rgb_array' if video else None)
            try:
                for seed in seeds:
                    try:
                        obs, _ = env.reset(seed=seed)
                        while True:
                            action = env.scripted_action() if model is None else model.predict(obs,deterministic=True)[0]
                            obs, _, terminated, truncated, info = env.step(action)
                            if terminated or truncated:
                                break
                        rows.append(info)
                        if video:
                            imageio.mimsave(str(video/f'{distance*100:g}cm_seed{seed}.mp4'),env.frames,fps=20)
                    except RuntimeError as exc:
                        rows.append(dict(seed=seed,distance_m=distance,success=False,reason=str(exc)))
            finally:
                env.close()
    finally:
        random.setstate(states[0]); np.random.set_state(states[1]); torch.set_rng_state(states[2])
    return dict(successes=sum(r['success'] for r in rows),attempts=len(rows),rows=rows)


class Progress(BaseCallback):
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.file = (root/'training_steps.csv').open('w',newline='',encoding='utf-8')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['step','reward','x_error_m','lateral_error_m','action_x','action_y','done','success','reason'])
        self.episodes = []

    def _on_step(self):
        info = self.locals['infos'][0]
        self.writer.writerow([self.num_timesteps,float(self.locals['rewards'][0]),info['x_error_m'],
                              info['lateral_error_m'],*self.locals['actions'][0],
                              bool(self.locals['dones'][0]),info['success'],info['reason']])
        if self.locals['dones'][0]:
            self.episodes.append({k:v for k,v in info.items() if k not in ('terminal_observation','episode')})
        if self.num_timesteps % 500 == 0:
            self.file.flush()
            data = dict(steps=self.num_timesteps,updates=self.model._n_updates,
                        episodes=len(self.episodes),successes=sum(x['success'] for x in self.episodes))
            save(self.root/'progress.json',data)
            print(data,flush=True)
        return True


def parameters(module):
    return torch.cat([p.detach().cpu().flatten() for p in module.parameters()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=6000)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--test-episodes',type=int,default=5,help='Per distance; three distances')
    parser.add_argument('--resume',type=Path,help='Previous complete run; restores model and replay buffer')
    args = parser.parse_args()
    if args.steps < 2000 or args.steps % 2000 or not 0 <= args.seed < 100000 or args.test_episodes < 1:
        parser.error('steps must be a positive multiple of 2000; seed 0..99999; test episodes >=1')
    root = args.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    config = dict(contract=CONTRACT,additional_steps=args.steps,seed=args.seed,
                  stage_validation_seeds=[210000,210001],checkpoint_validation_seeds=[200000,200001],
                  test_seeds=list(range(300000,300000+args.test_episodes)),
                  caveat='After inspecting this test set it is a development regression, not a fresh holdout.',
                  algorithm='SAC',network=[128,128],device='cpu',learning_starts=200,ent_coef='auto_0.05',
                  promotion='Two consecutive stage validations with >=80% success',
                  llm_used=False,resume=str(args.resume) if args.resume else None)
    save(root/'config.json',config)
    frozen = root/'source'; frozen.mkdir()
    hashes = {}
    for name in ('push_distance.py','push_skill.py','train_push.py'):
        data = Path(__file__).with_name(name).read_bytes()
        (frozen/name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    save(root/'source_hashes.json',hashes)
    stage,streak = 0,0
    if args.resume:
        prior = json.loads((args.resume/'summary.json').read_text())
        stage,streak = prior['stage'],prior['promotion_streak']
        oldhashes = json.loads((args.resume/'source_hashes.json').read_text())
        if oldhashes != hashes:
            raise ValueError('Source changed: start a fresh run instead of resuming an incompatible buffer')
    env = Monitor(PushSkill(distance_m=CONTRACT['distances_m'][stage]),str(root/'monitor.csv'))
    callback = Progress(root)
    start_time = time.perf_counter()
    history = []
    report = dict(status='running')
    try:
        if args.resume:
            model = SAC.load(args.resume/'last_model.zip',env=env,device='cpu')
            model.load_replay_buffer(args.resume/'replay_buffer.pkl')
            env.reset(seed=args.seed)
        else:
            model = SAC('MlpPolicy',env,device='cpu',seed=args.seed,policy_kwargs={'net_arch':[128,128]},
                        learning_starts=200,buffer_size=50000,batch_size=128,gamma=.98,
                        learning_rate=3e-4,train_freq=1,gradient_steps=1,ent_coef='auto_0.05',verbose=0)
        initial_step = model.num_timesteps
        actor_before = parameters(model.actor).clone()
        critic_before = parameters(model.critic).clone()
        model.save(root/'initial_model.zip')
        initial = evaluate(model,config['checkpoint_validation_seeds'])
        save(root/'initial_validation.json',initial)
        def rank(result):
            return (result['successes']/result['attempts'],
                    -sum(r.get('x_error_m',1) + r.get('lateral_error_m',1) for r in result['rows'])/result['attempts'])
        best_rank = rank(initial)
        best_score = best_rank[0]
        best_step = initial_step
        model.save(root/'best_model.zip')
        if args.resume:
            prior_best = SAC.load(args.resume/'best_model.zip',device='cpu')
            prior_result = evaluate(prior_best,config['checkpoint_validation_seeds'])
            if rank(prior_result) > best_rank:
                best_rank = rank(prior_result); best_score = best_rank[0]
                best_step = prior['best_step']
                prior_best.save(root/'best_model.zip')
        for _ in range(args.steps//2000):
            model.learn(total_timesteps=2000,callback=callback,reset_num_timesteps=False)
            stage_result = evaluate(model,config['stage_validation_seeds'],[CONTRACT['distances_m'][stage]])
            validation = evaluate(model,config['checkpoint_validation_seeds'])
            score = validation['successes']/validation['attempts']
            if rank(validation) > best_rank:
                best_rank = rank(validation)
                best_score,best_step = score,model.num_timesteps
                model.save(root/'best_model.zip')
            streak = streak+1 if stage_result['successes']/stage_result['attempts'] >= .8 else 0
            row = dict(step=model.num_timesteps,stage=stage,streak=streak,
                       stage_validation=stage_result,full_validation=validation)
            history.append(row); save(root/'validation_history.json',history)
            print(dict(step=model.num_timesteps,stage=stage,stage_pass=stage_result['successes'],
                       validation_successes=validation['successes'],validation_attempts=validation['attempts']),flush=True)
            if streak >= 2 and stage < 2:
                stage += 1; streak = 0
                env.unwrapped.fixed_distance = CONTRACT['distances_m'][stage]
                model._last_obs = None  # next learn resets; goals never move mid-episode
            model.save(root/'last_model.zip')
            model.save_replay_buffer(root/'replay_buffer.pkl')
        restored = SAC.load(root/'last_model.zip',device='cpu')
        probe = np.zeros(26,dtype=np.float32)
        matched = bool(np.allclose(model.predict(probe,deterministic=True)[0],restored.predict(probe,deterministic=True)[0],atol=1e-6))
        actor_delta = float(torch.linalg.vector_norm(parameters(model.actor)-actor_before))
        critic_delta = float(torch.linalg.vector_norm(parameters(model.critic)-critic_before))
        finite = bool(torch.isfinite(parameters(model.actor)).all() and torch.isfinite(parameters(model.critic)).all())
        if not (matched and actor_delta > 0 and critic_delta > 0 and finite):
            raise RuntimeError('Parameter-update/checkpoint verification failed')
        baseline = evaluate(None,config['test_seeds'])
        best = SAC.load(root/'best_model.zip',device='cpu')
        test = evaluate(best,config['test_seeds'])
        save(root/'scripted_test.json',baseline); save(root/'sac_test.json',test)
        videos = root/'videos'; videos.mkdir()
        video_result = evaluate(best,[310000],video=videos)
        save(root/'video_results.json',video_result)
        save(root/'training_episodes.json',callback.episodes)
        save(root/'training_scene_seeds.json',env.unwrapped.scene_seeds)
        report.update(status='pipeline_verified',total_steps=model.num_timesteps,gradient_updates=model._n_updates,
                      actor_parameter_change_l2=actor_delta,critic_parameter_change_l2=critic_delta,
                      finite_parameters=finite,reloaded_actions_match=matched,best_step=best_step,
                      stage=stage,promotion_streak=streak,best_validation_rate=best_score,
                      scripted_test_successes=baseline['successes'],sac_test_successes=test['successes'],
                      test_attempts=test['attempts'],video_successes=video_result['successes'])
        text = (f'# Push SAC experiment\n\nStatus: {report["status"]}.\n\n'
                f'Trained {args.steps} additional steps; best checkpoint at {best_step}.\n'
                f'Scripted baseline: {baseline["successes"]}/{baseline["attempts"]}. '
                f'SAC: {test["successes"]}/{test["attempts"]}.\n\n'
                'These are fixed, same-distribution development tests, not proof of general robustness. '
                'Videos use a separate seed and are excluded from test counts. No LLM was used.\n'
                'Scripted approach and retreat surround learned XY pushing; Z and orientation remain scripted.\n')
        (root/'REPORT.md').write_text(text,encoding='utf-8')
        print(report,flush=True)
    except Exception as exc:
        report.update(status='failed',error=str(exc)); raise
    finally:
        report['elapsed_seconds'] = time.perf_counter()-start_time
        save(root/'summary.json',report)
        callback.file.close(); env.close()


if __name__ == '__main__':
    main()
