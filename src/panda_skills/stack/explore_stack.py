"""Step 1: Explore ManiSkill's built-in StackCube-v1 task.

Prints what the robot observes, what it controls, and how success is judged,
then runs random actions and saves a short video.

Run from the repo root:
    .\run-skills.cmd explore_stack
"""
import argparse
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import mani_skill.envs  # noqa: F401  (registers StackCube-v1 and other built-in tasks)
import numpy as np


def to_numpy(x):
    """ManiSkill returns batched torch tensors even for one environment."""
    return x.cpu().numpy()[0] if hasattr(x, "cpu") else np.asarray(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("runs/stack_explore"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        "StackCube-v1",
        obs_mode="state_dict",           # readable dictionary instead of one long vector
        control_mode="pd_ee_delta_pose",  # move the gripper in XYZ + rotation, like the team's skills
        reward_mode="dense",
        render_mode="rgb_array",
        # Render on the CPU side. The default GPU (CUDA) image path crashes with an
        # access violation on Windows.
        render_backend="cpu",
    )
    obs, info = env.reset(seed=args.seed)
    raw = env.unwrapped

    print("=== Observation (state_dict) ===")
    for group, values in obs.items():
        if isinstance(values, dict):
            for name, value in values.items():
                print(f"  {group}/{name}: shape {tuple(value.shape)}")
        else:
            print(f"  {group}: shape {tuple(values.shape)}")

    print("\n=== Action space ===")
    print(" ", env.action_space)
    print("  pd_ee_delta_pose = [dx, dy, dz, droll, dpitch, dyaw, gripper]; gripper +1 open, -1 close")

    print("\n=== Scene ===")
    print("  cube half size (m):", to_numpy(raw.cube_half_size))
    print("  cubeA (red, to pick) position:  ", np.round(to_numpy(raw.cubeA.pose.p), 3))
    print("  cubeB (green, base) position:   ", np.round(to_numpy(raw.cubeB.pose.p), 3))
    print("  gripper (TCP) position:         ", np.round(to_numpy(raw.agent.tcp.pose.p), 3))
    print("  max episode steps:", env.spec.max_episode_steps)

    print("\n=== Success check (env.evaluate) ===")
    for name, value in raw.evaluate().items():
        print(f"  {name}: {to_numpy(value)}")

    frames = [to_numpy(env.render()).astype(np.uint8)]
    total = 0.0
    for _ in range(args.steps):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total += float(to_numpy(reward))
        frames.append(to_numpy(env.render()).astype(np.uint8))
        if bool(to_numpy(terminated)) or bool(to_numpy(truncated)):
            break

    video = args.output / f"random_seed{args.seed}.mp4"
    imageio.mimsave(str(video), frames, fps=20)
    print(f"\nRandom actions: total dense reward {total:.2f}, success={bool(to_numpy(info['success']))}")
    print("Video saved to", video)
    env.close()


if __name__ == "__main__":
    main()
