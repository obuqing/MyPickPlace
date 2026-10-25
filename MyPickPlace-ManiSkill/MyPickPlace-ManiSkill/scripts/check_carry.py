"""Check the real Gym/SB3 interface and replay-buffer observation ownership."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_checker import check_env

import envs
from envs.carry_skill import CONTRACT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/carry_interface"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env = gym.make("MyCarrySkill-v1", audit_path=args.output / "initializations.jsonl")
    try:
        check_env(env.unwrapped, warn=True)
        obs, info = env.reset(seed=1000, options={"target_color": "blue"})
        snapshot = {key: value.copy() for key, value in obs.items()}
        env.step(np.zeros(3, dtype=np.float32))
        assert all(np.array_equal(obs[key], value) for key, value in snapshot.items()), "Observation aliased live physics state"
        before = env.unwrapped.steps
        for bad in (np.zeros(7), np.array([np.nan, 0, 0])):
            try:
                env.step(bad)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid action accepted")
        assert env.unwrapped.steps == before, "Invalid action advanced physics"
        result = {"sb3_check_env": True, "observations_are_copies": True,
                  "invalid_actions_rejected_before_simulation": True,
                  "observation_dimensions": sum(CONTRACT["observation_sizes"].values()),
                  "action_dimensions": 3, "contract": CONTRACT,
                  "initialization_attempts": len(env.unwrapped.initializations),
                  "initialization_failures": sum(not r["success"] for r in env.unwrapped.initializations)}
        (args.output / "checks.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(result, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
