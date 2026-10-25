"""Frozen-policy evaluation on audited unseen seeds and two preselected full videos."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import envs
import imageio.v2 as imageio
import numpy as np
import torch
from stable_baselines3 import SAC
from algorithms.carry_evaluation import evaluate
from envs.carry_skill import CarrySkillEnv, InitializationError, CONTRACT


def used_scene_seeds(runs):
    """Read actual scene audits/results, not just declared training RNG seeds."""
    seeds, files = set(), []
    history = runs / 'scene_history.json'
    if history.exists():
        seeds.update(json.loads(history.read_text(encoding='utf-8'))['scene_seeds'])
    for pattern in ('*initializations*.jsonl', 'results.json'):
        for path in runs.rglob(pattern):
            text = path.read_text(encoding='utf-8-sig')
            data = [json.loads(line) for line in text.splitlines() if line.strip()] if path.suffix == '.jsonl' else json.loads(text)
            if not isinstance(data, list):
                continue
            count = 0
            for row in data:
                if isinstance(row, dict) and isinstance(row.get('seed'), int):
                    seeds.add(row['seed'])
                    count += 1
            if count:
                files.append(dict(path=str(path.resolve()), records=count,
                                  sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return seeds, files


class RecordedCarryEnv(CarrySkillEnv):
    """Observe every physical step, including scripted initialization and release."""
    def __init__(self, writer, **kwargs):
        self.writer = writer
        self.frame_count = 0
        super().__init__(render_mode='rgb_array', **kwargs)

    def _sim_step(self, action):
        result = super()._sim_step(action)
        self.writer.append_data(np.asarray(self.render(), dtype=np.uint8))
        self.frame_count += 1
        return result


def record(policy, color, seed, directory):
    path = directory/f'{color}.mp4'
    row = dict(color=color, seed=seed, video=str(path), initialization_success=False,
               move_success=False, placement_success=False, trace=[])
    with imageio.get_writer(str(path), fps=20) as writer:
        environment = RecordedCarryEnv(writer, target_color=color, audit_path=directory/f'{color}_initializations.jsonl')
        try:
            obs, info = environment.reset(seed=seed)
            row.update(initialization_success=True, initialization_steps=info['initialization_steps'])
            for _ in range(CONTRACT['max_carry_steps']):
                action = policy.predict(obs, deterministic=True)[0]
                obs, _, terminated, truncated, info = environment.step(action)
                row['trace'].append(dict(step=info['carry_steps'], action=action.tolist(),
                                         error_xyz=obs['delta'].tolist(), distance=info['distance']))
                if terminated or truncated:
                    break
            row.update(move_success=info['move_success'], reason=info['reason'], carry_steps=info['carry_steps'])
            if info['move_success']:
                row['placement'] = environment.finish_placement()
                row['placement_success'] = row['placement']['placement_success']
            row['frames'] = environment.frame_count
            expected = row['initialization_steps'] + row['carry_steps'] + row.get('placement', {}).get('placement_steps', 0)
            assert row['frames'] == expected, 'Video must include every physical control step'
        except InitializationError as exc:
            row.update(reason='initialization_failed', initialization=exc.record, frames=environment.frame_count)
        finally:
            environment.close()
    # Decode the actual MP4 to verify it is complete and playable.
    reader = imageio.get_reader(str(path))
    try:
        row['decoded_frames'] = reader.count_frames()
        row['resolution'] = list(reader.get_data(0).shape[:2])
        assert row['decoded_frames'] == row['frames'] and row['frames'] > 0
    finally:
        reader.close()
    row['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (directory/f'{color}.json').write_text(json.dumps(row, indent=2), encoding='utf-8')
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path('runs/carry_curriculum_40000/training/best_model.zip'))
    parser.add_argument('--output', type=Path, default=Path('runs/carry_generalization_200'))
    parser.add_argument('--episodes', type=int, default=100, help='Unseen scenes per color')
    parser.add_argument('--seed', type=int, default=50000)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('Episodes must be positive')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    state = dict(status='running', phase='seed_audit', started=datetime.now().isoformat())
    def save():
        (root/'status.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    save()
    begin = time.perf_counter()
    try:
        used, audits = used_scene_seeds(Path('runs'))
        selected = set(range(args.seed, args.seed + args.episodes))
        assert selected.isdisjoint(used), f'Seeds already used: {sorted(selected & used)}'
        model_path = args.model.resolve()
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        project = Path(__file__).resolve().parents[1]
        code = {}
        for name in ('scripts/validate_carry_generalization.py', 'algorithms/carry_evaluation.py', 'envs/carry_skill.py',
                     'envs/pick_place.py', 'algorithms/scripted_pick_place.py'):
            data = (project/name).read_bytes()
            destination = root/'frozen'/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            code[name] = hashlib.sha256(data).hexdigest()
        manifest = dict(model=str(model_path), model_sha256=digest, seed_start=args.seed,
                        scene_seeds=sorted(selected), episodes_per_color=args.episodes,
                        prior_unique_scene_seeds=len(used), audited_files=audits, seed_overlap=[],
                        deterministic=True, no_retries=True, training=False, llm_used=False,
                        video_selection='first preselected seed, both colors; no success-based replacement', code_sha256=code)
        (root/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        torch.set_num_threads(1)
        policy = SAC.load(model_path, device='cpu')
        state.update(phase='unseen_scene_evaluation', total_episodes=2*args.episodes)
        save()
        report = evaluate(policy, args.episodes, args.seed, root/'evaluation')
        rows = json.loads((root/'evaluation/results.json').read_text())
        assert len(rows) == 2*args.episodes
        state['phase'] = 'recording_videos'
        save()
        videos = root/'videos'
        videos.mkdir()
        recordings = [record(policy, color, args.seed, videos) for color in ('red','blue')]
        assert hashlib.sha256(model_path.read_bytes()).hexdigest() == digest
        failures = [dict(color=r['color'], seed=r['seed'],
                         reason=r.get('placement', {}).get('reason', r['reason']),
                         carry_success=r['move_success'], final_distance=r.get('final_distance'))
                    for r in rows if not r['placement_success']]
        counts = dict(Counter(r['reason'] for r in failures))
        paired = sum(all(r['placement_success'] for r in rows if r['seed'] == seed) for seed in selected)
        summary = dict(summary=report['summary'], paired_scene_successes=paired, scene_count=args.episodes,
                       failure_reasons=counts, failures=failures, videos=recordings,
                       elapsed_seconds=time.perf_counter()-begin, model_sha256=digest)
        (root/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        lines = ['# Unseen-scene carry evaluation', '',
                 f'Frozen step-40000 policy; {args.episodes} unseen scene seeds, each tested once per color.',
                 'Actual previous initialization and evaluation records were audited for seed overlap (none).',
                 'No training, retries, checkpoint selection or LLM calls. Scripted grasp/lift and release; SAC controls XYZ carry.',
                 'New seeds test the existing scene distribution; they do not test changed physics or a second training seed.', '',
                 '| Color | Initialization | Carry | Full placement | Mean arrival error (m) | Absolute height error (m) |',
                 '| --- | --- | --- | --- | --- | --- |']
        for color in ('red','blue','all'):
            s = report['summary'][color]
            n = s['attempted']
            lines.append(f"| {color} | {s['initialization_successes']}/{n} | {s['move_successes']}/{n} | {s['placement_successes']}/{n} | {s['mean_final_distance']} | {s['mean_final_abs_error_xyz'][2]} |")
        lines += ['', f'Both colors succeed in {paired}/{args.episodes} scenes.', f'Failure reasons: {counts}.', '',
                  '## Videos', '', 'Both videos use the first preselected scene, include physical grasp, SAC carry and release, and retain failures.', '']
        for row in recordings:
            lines.append(f"- [{row['color']}](videos/{row['color']}.mp4): seed {row['seed']}, placement={row['placement_success']}, {row['decoded_frames']} decoded frames.")
        lines += ['', '## Next improvements', '']
        if not failures and all(r['placement_success'] for r in recordings):
            lines.append('Proceed to a small LLM-to-structured-task-to-frozen-SAC integration evaluation with explicit rejection cases and end-to-end logs. Keep this model frozen; separately repeat training seeds and physics variation before broader robustness claims.')
        else:
            lines.append('Inspect the listed failed scenes and separate scripted initialization, learned carry and release failures. Add targeted development cases before connecting the LLM, then use a fresh untouched holdout for the next assessment.')
        (root/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        state.update(status='complete', phase='complete', finished=datetime.now().isoformat(),
                     placement_successes=report['summary']['all']['placement_successes'], attempted=len(rows))
    except Exception as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}', finished=datetime.now().isoformat())
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
