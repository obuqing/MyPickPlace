"""Evaluate both cube colors and record exactly one episode per color."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import imageio.v2 as imageio
import mani_skill.envs
import numpy as np

import envs
from algorithms.scripted_pick_place import ScriptedPickPlacePolicy, array


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--color", choices=("red", "blue", "both"), default="both")
    parser.add_argument("--output", type=Path, default=Path("runs/two_color_scripted"))
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    colors = ("red", "blue") if args.color == "both" else (args.color,)
    results = []
    for color in colors:
        env = gym.make(
            "MyPickPlace-v1", target_color=color, obs_mode="state_dict",
            control_mode="pd_ee_delta_pose", render_mode="rgb_array",
            max_episode_steps=400,
        )
        try:
            for episode in range(args.episodes):
                seed = args.seed + episode
                obs, info = env.reset(seed=seed)
                policy = ScriptedPickPlacePolicy()
                policy.reset()
                other = "blue" if color == "red" else "red"
                other_start = array(obs["extra"][other + "_cube_pose"])[:3].copy()
                writer = imageio.get_writer(str(args.output / f"{color}.mp4"), fps=20) if episode == 0 else None
                trace = []
                stable = 0
                try:
                    if writer:
                        writer.append_data(env.render()[0].cpu().numpy().astype(np.uint8))
                    for step in range(400):
                        action = policy.act(obs)
                        if policy.failure:
                            break
                        obs, reward, terminated, truncated, info = env.step(action)
                        if info["is_wrong_grasped"].item():
                            policy.failure = "wrong_object_grasped"
                        stable = stable + 1 if info["success"].item() else 0
                        trace.append({
                            "step": step + 1, "phase": policy.phase,
                            "object_position": array(obs["extra"]["obj_pose"])[:3].tolist(),
                            "tcp_position": array(obs["extra"]["tcp_pose"])[:3].tolist(),
                            "grasped": bool(info["is_grasped"].item()),
                            "success": bool(info["success"].item()),
                        })
                        if writer:
                            writer.append_data(env.render()[0].cpu().numpy().astype(np.uint8))
                        # Require the full controller sequence and sustained final success.
                        if policy.failure or policy.done or truncated.item():
                            break
                    success = policy.done and not policy.failure and stable >= 5
                    row = {
                        "color": color, "seed": seed, "success": bool(success),
                        "reason": "completed" if success else policy.failure or "placement_not_stable",
                        "steps": len(trace),
                        "other_cube_displacement": float(np.linalg.norm(array(obs["extra"][other + "_cube_pose"])[:3] - other_start)),
                        "trace": trace,
                    }
                    results.append(row)
                    print(f"{color} seed={seed} success={success} steps={len(trace)} reason={row['reason']}", flush=True)
                    (args.output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
                finally:
                    if writer:
                        writer.close()
        finally:
            env.close()
    summary = {
        color: {"successes": sum(r["success"] for r in results if r["color"] == color), "episodes": args.episodes}
        for color in colors
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
