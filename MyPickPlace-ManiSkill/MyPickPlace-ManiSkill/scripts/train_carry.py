"""Short SAC run with measured parameter updates and checkpoint round-trip checks."""

import argparse
import csv
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

import envs
from envs.carry_skill import CONTRACT
from algorithms.carry_validation import CarryValidation


def parameters(module):
    return torch.cat([p.detach().cpu().flatten() for p in module.parameters()]).clone()


class Progress(BaseCallback):
    def __init__(self, output):
        super().__init__()
        self.output = output
        self.episodes = []
        self.stream = (output / "training_steps.csv").open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.stream)
        self.writer.writerow(["step", "reward", "distance", "error_x", "error_y", "error_z", "action_x", "action_y", "action_z", "done", "reason"])

    def _on_step(self):
        info = self.locals["infos"][0]
        state = info.get("terminal_observation")
        error = state["delta"] if state is not None else self.locals["new_obs"]["delta"][0]
        action = self.locals["actions"][0]
        self.writer.writerow([self.num_timesteps, float(self.locals["rewards"][0]), float(np.linalg.norm(error)),
                              *np.asarray(error).tolist(), *np.asarray(action).tolist(), bool(self.locals["dones"][0]), info["reason"]])
        for info in self.locals["infos"]:
            if "episode" in info:
                self.episodes.append({"step": self.num_timesteps, "return": info["episode"]["r"],
                                      "length": info["episode"]["l"], "color": info["color"],
                                      "move_success": info["move_success"], "reason": info["reason"]})
        if self.num_timesteps % 128 == 0:
            self.stream.flush()
            progress = {"steps": self.num_timesteps, "updates": self.model._n_updates,
                        "completed_episodes": len(self.episodes),
                        "training_carry_successes": sum(e["move_success"] for e in self.episodes)}
            (self.output / "progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
            print(progress, flush=True)
        return True

    def _on_training_end(self):
        self.stream.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--eval-freq", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=5, help="Validation episodes per color")
    parser.add_argument("--eval-seed", type=int, default=2000)
    args = parser.parse_args()
    if args.steps < 200:
        parser.error("Use at least 200 steps so the run includes actual SAC updates")
    if args.eval_freq < 0 or args.eval_episodes < 1:
        parser.error("Invalid validation frequency or episode count")
    output = args.output or Path("runs") / ("carry_sac_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)
    config = {"contract": CONTRACT, "training_seed": args.seed, "steps": args.steps,
              "algorithm": "SAC", "policy": "MultiInputPolicy", "device": "cpu", "network": [128, 128],
              "learning_starts": 100, "buffer_size": 50000, "batch_size": 128,
              "learning_rate": 3e-4, "gamma": 0.98, "train_freq": 1, "gradient_steps": 1,
              "stable_baselines3": importlib.metadata.version("stable-baselines3"),
              "language_model_used": False, "purpose": "training_pipeline_smoke_test"}
    config.update(eval_freq=args.eval_freq, eval_episodes_per_color=args.eval_episodes, eval_seed_start=args.eval_seed,
                  purpose="periodically_validated_training" if args.eval_freq else config["purpose"])
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    torch.set_num_threads(1)
    env = Monitor(gym.make("MyCarrySkill-v1", target_color="random", audit_path=output / "initializations.jsonl"),
                  str(output / "monitor.csv"), info_keywords=("move_success", "reason", "color"))
    report = {"status": "training", "config": config}
    start = time.perf_counter()
    callback = None
    try:
        model = SAC("MultiInputPolicy", env, seed=args.seed, device="cpu", policy_kwargs={"net_arch": [128, 128]},
                    learning_starts=100, buffer_size=50000, batch_size=128, learning_rate=3e-4,
                    gamma=0.98, train_freq=1, gradient_steps=1, verbose=0)
        model.save(output / "untrained.zip")
        actor_before, critic_before = parameters(model.actor), parameters(model.critic)
        callback = Progress(output)
        validator = CarryValidation(output, args.eval_episodes, args.eval_seed) if args.eval_freq else None
        if validator:
            validator.run(model)
            while model.num_timesteps < args.steps:
                count = min(args.eval_freq, args.steps-model.num_timesteps)
                model.learn(total_timesteps=count, callback=callback, reset_num_timesteps=False)
                validator.run(model)
        else:
            model.learn(total_timesteps=args.steps, callback=callback)
        model.save(output / "model.zip")
        model.save_replay_buffer(output / "replay_buffer.pkl")
        restored = SAC.load(output / "model.zip", device="cpu")
        original_action = model.predict(model._last_obs, deterministic=True)[0]
        restored_action = restored.predict(model._last_obs, deterministic=True)[0]
        actor_delta = float(torch.linalg.vector_norm(parameters(model.actor) - actor_before))
        critic_delta = float(torch.linalg.vector_norm(parameters(model.critic) - critic_before))
        finite = bool(torch.isfinite(parameters(model.actor)).all() and torch.isfinite(parameters(model.critic)).all())
        matched = bool(np.allclose(original_action, restored_action, rtol=0, atol=1e-6))
        if not (model._n_updates > 0 and actor_delta > 0 and critic_delta > 0 and finite and matched):
            raise RuntimeError("SAC update/checkpoint verification failed")
        report.update(status="pipeline_verified", environment_steps=model.num_timesteps,
                      gradient_updates=model._n_updates, replay_buffer_size=model.replay_buffer.size(),
                      actor_parameter_change_l2=actor_delta, critic_parameter_change_l2=critic_delta,
                      finite_parameters=finite, reloaded_actions_match=matched,
                      completed_episodes=callback.episodes,
                      initialization_attempts=len(env.unwrapped.initializations),
                      initialization_failures=sum(not r["success"] for r in env.unwrapped.initializations))
        if validator:
            report.update(best_validation_step=validator.best_step, validation_steps=[h["step"] for h in validator.history],
                          validation_process_isolated=True)
        (output / "progress.json").write_text(json.dumps({
            "status": "pipeline_verified", "steps": model.num_timesteps, "updates": model._n_updates,
            "completed_episodes": len(callback.episodes),
            "training_carry_successes": sum(e["move_success"] for e in callback.episodes),
        }, indent=2), encoding="utf-8")
        print({key: value for key, value in report.items() if key not in ("config", "completed_episodes")}, flush=True)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - start
        (output / "training.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if callback:
            callback.stream.close()
        env.close()


if __name__ == "__main__":
    main()
