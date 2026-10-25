"""Seeded conditional carry and end-to-end metrics, with no language model."""

import json
from pathlib import Path
import time

import gymnasium as gym
import numpy as np

from envs.carry_skill import CONTRACT, InitializationError


def evaluate(policy, episodes=10, seed=1000, output=None, max_target_distance=None):
    output = Path(output) if output else None
    if output:
        output.mkdir(parents=True, exist_ok=True)
    env = gym.make("MyCarrySkill-v1", audit_path=output / "initializations.jsonl" if output else None,
                   max_target_distance=max_target_distance)
    rows = []
    try:
        for color in ("red", "blue"):
            for episode in range(episodes):
                episode_seed = seed + episode
                start = time.perf_counter()
                row = {"color": color, "seed": episode_seed, "initialization_success": False,
                       "move_success": False, "placement_success": False, "trace": []}
                try:
                    obs, initial = env.reset(seed=episode_seed, options={"target_color": color})
                    row.update(initialization_success=True, initialization_steps=initial["initialization_steps"],
                               initial_distance=float(np.linalg.norm(obs["delta"])), initial_error_xyz=obs["delta"].tolist(),
                               initial_object_position=obs["object_pose"][:3].tolist(), target_position=obs["target_pos"].tolist())
                    for step in range(CONTRACT["max_carry_steps"]):
                        action = env.action_space.sample() if policy is None else policy.predict(obs, deterministic=True)[0]
                        obs, reward, terminated, truncated, info = env.step(action)
                        row["trace"].append({"step": step+1, "distance": info["distance"],
                                             "is_grasped": info["is_grasped"], "reward": reward,
                                             "error_xyz": obs["delta"].tolist(),
                                             "object_position": obs["object_pose"][:3].tolist(),
                                             "action": np.asarray(action).tolist()})
                        if terminated or truncated:
                            break
                    row.update(move_success=info["move_success"], reason=info["reason"],
                               carry_steps=info["carry_steps"], final_distance=info["distance"])
                    if info["move_success"]:
                        placement = env.unwrapped.finish_placement()
                        row["placement"] = placement
                        row["placement_success"] = placement["placement_success"]
                except InitializationError as exc:
                    row.update(reason="initialization_failed", initialization=exc.record)
                row["seconds"] = time.perf_counter() - start
                rows.append(row)
                print(f"{color} seed={episode_seed}: init={row['initialization_success']} carry={row['move_success']} placed={row['placement_success']} reason={row['reason']}", flush=True)
                if output:
                    (output / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    finally:
        env.close()
    summary = {}
    for color in ("all", "red", "blue"):
        subset = rows if color == "all" else [r for r in rows if r["color"] == color]
        initialized = sum(r["initialization_success"] for r in subset)
        carried = sum(r["move_success"] for r in subset)
        placed = sum(r["placement_success"] for r in subset)
        valid = [r for r in subset if r["initialization_success"]]
        errors = [r["trace"][-1]["error_xyz"] for r in valid]
        actions = [t["action"] for r in valid for t in r["trace"]]
        summary[color] = {"attempted": len(subset), "initialization_successes": initialized,
                          "initialization_failures": len(subset)-initialized,
                          "move_successes": carried, "move_rate_given_initialization": carried/initialized if initialized else None,
                          "placement_successes": placed, "end_to_end_rate": placed/len(subset),
                          "drops": sum(r.get("reason") == "dropped" for r in subset),
                          "timeouts": sum(r.get("reason") == "timeout" for r in subset),
                          "mean_initial_distance": float(np.mean([r["initial_distance"] for r in valid])) if valid else None,
                          "mean_final_distance": float(np.mean([r["final_distance"] for r in valid])) if valid else None,
                          "mean_min_distance": float(np.mean([min([r["initial_distance"]]+[t["distance"] for t in r["trace"]]) for r in valid])) if valid else None,
                          "mean_return": float(np.mean([sum(t["reward"] for t in r["trace"]) for r in valid])) if valid else None,
                          "mean_final_error_xyz": np.mean(errors, axis=0).tolist() if errors else None,
                          "mean_final_abs_error_xyz": np.mean(np.abs(errors), axis=0).tolist() if errors else None,
                          "mean_action_xyz": np.mean(actions, axis=0).tolist() if actions else None}
    report = {"contract": CONTRACT, "episodes_per_color": episodes, "seed_start": seed,
              "max_target_distance": max_target_distance, "error_convention": "target_minus_object; negative Z means object too high",
              "language_model_used": False, "summary": summary}
    if output:
        (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
