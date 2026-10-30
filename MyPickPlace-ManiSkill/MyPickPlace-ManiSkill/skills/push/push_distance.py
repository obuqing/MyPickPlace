"""Distance-controlled +X pushing starter, based on ManiSkill PushCubeEnv.

This is a scripted baseline, not a trained RL policy or an LLM integration.
"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
os.environ.setdefault('MS_ASSET_DIR', str(ROOT / 'work/maniskill-assets'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'work/matplotlib'))

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose


@register_env('MyPushDistance-v1', max_episode_steps=350)
class PushDistanceEnv(PushCubeEnv):
    goal_radius = 0.01

    def __init__(self, *args, distance_m=0.10, **kwargs):
        if not np.isfinite(distance_m) or not 0.03 <= distance_m <= 0.15:
            raise ValueError('distance_m must be finite and between 0.03 and 0.15 m')
        self.distance_m = float(distance_m)
        kwargs.setdefault('sim_backend', 'cpu')
        kwargs.setdefault('render_backend', 'cpu')
        super().__init__(*args, **kwargs)

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        self.start_position = self.obj.pose.p.clone()
        target = self.start_position.clone()
        target[:, 0] += self.distance_m
        target[:, 2] = 0.001
        self.goal_region.set_pose(Pose.create_from_pq(p=target, q=[0.70710678, 0, 0.70710678, 0]))

    def evaluate(self):
        delta = self.obj.pose.p - self.start_position
        x_error = (delta[:, 0] - self.distance_m).abs()
        lateral_error = delta[:, 1].abs()
        on_table = (self.obj.pose.p[:, 2] - self.cube_half_size).abs() < 0.005
        slow = torch.linalg.norm(self.obj.linear_velocity, dim=1) < 0.02
        success = (x_error <= 0.01) & (lateral_error <= 0.01) & on_table & slow
        return dict(success=success, x_error=x_error, lateral_error=lateral_error, on_table=on_table)


def vector(value):
    return value.detach().cpu().numpy()[0].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--distance-cm', type=float, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=ROOT / 'work/push_demo')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    env = gym.make('MyPushDistance-v1', distance_m=args.distance_cm/100,
                   obs_mode='state_dict', control_mode='pd_ee_delta_pose', render_mode='rgb_array')
    trace = []
    phase, count, stable = 'approach', 0, 0
    writer = None
    try:
        obs, _ = env.reset(seed=args.seed)
        start = vector(env.unwrapped.obj.pose.p)
        goal = start.copy()
        goal[0] += args.distance_cm/100
        writer = imageio.get_writer(str(args.output / 'push.mp4'), fps=20)
        for step in range(350):
            obj = vector(env.unwrapped.obj.pose.p)
            tcp_pose = vector(env.unwrapped.agent.tcp.pose.raw_pose)
            tcp = tcp_pose[:3]
            count += 1
            if phase == 'approach':
                target = start + [-0.06, 0, 0.14]
                if np.linalg.norm(target-tcp) < 0.008:
                    phase, count = 'descend', 0
            elif phase == 'descend':
                target = start + [-0.06, 0, 0.005]
                if np.linalg.norm(target-tcp) < 0.006:
                    phase, count = 'push', 0
            elif phase == 'push':
                target = np.array([min(tcp[0]+0.004, goal[0]-0.02), start[1], start[2]+0.005])
                if obj[0] >= goal[0]-0.004:
                    retreat = tcp + [-0.04, 0, 0.08]
                    phase, count = 'retreat', 0
            else:
                target = retreat
            action = np.zeros(7, dtype=np.float32)
            action[:3] = np.clip((target-tcp)*5, -0.15, 0.15)
            rotation = Rotation.from_quat(tcp_pose[[4,5,6,3]])
            error = (Rotation.from_quat([1,0,0,0])*rotation.inv()).as_rotvec()
            action[3:6] = np.clip(-error*5, -0.3, 0.3)
            action[6] = -1
            obs, reward, terminated, truncated, info = env.step(action)
            obj = vector(env.unwrapped.obj.pose.p)
            writer.append_data(env.render()[0].cpu().numpy().astype(np.uint8))
            trace.append({'step':step+1, 'phase':phase, 'position':obj.tolist(), 'action':action.tolist()})
            stable = stable+1 if phase == 'retreat' and bool(info['success'].item()) else 0
            if stable >= 10 or truncated.item() or (phase != 'push' and count > 120):
                break
        displacement = obj-start
        lifted = any(abs(row['position'][2]-float(start[2])) > 0.01 for row in trace)
        result = {'kind':'scripted_baseline_not_RL', 'seed':args.seed,
                  'requested_distance_m':args.distance_cm/100, 'displacement_m':displacement.tolist(),
                  'x_error_m':abs(float(displacement[0])-args.distance_cm/100),
                  'lateral_error_m':abs(float(displacement[1])), 'lifted':lifted,
                  'success':bool(stable >= 10 and not lifted), 'steps':len(trace), 'final_phase':phase,
                  'trace':trace}
        (args.output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in result.items() if k!='trace'},indent=2))
    finally:
        if writer is not None:
            writer.close()
        env.close()


if __name__ == '__main__':
    main()
