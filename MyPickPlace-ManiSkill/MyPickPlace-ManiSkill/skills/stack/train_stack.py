"""Step 5: Train SAC for the stacking skill, keep the best checkpoint, and test it.

Run from the repo root (new output folder each time):
    .\run-skills.cmd train_stack --steps 20000 --output runs/stack_sac_20000
"""
import argparse
import csv
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

from .stack_skill import CONTRACT, StackSkill

VALIDATION_SEEDS = list(range(20000, 20010))   # used to pick the best checkpoint
TEST_SEEDS = list(range(30000, 30020))         # touched only once, at the end
VIDEO_SEEDS = [40000, 40001]
DEMO_SEED_START = 10000                         # scenes used only for demonstrations


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def evaluate(model, seeds, video_dir=None):
    """Run full episodes. model=None runs the hand-written scripted baseline."""
    # Evaluation must not disturb the training random-number state.
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    rows = []
    env = StackSkill(render_mode="rgb_array" if video_dir else None)
    try:
        for seed in seeds:
            try:
                obs, _ = env.reset(seed=seed)
                while True:
                    action = env.scripted_action() if model is None else model.predict(obs, deterministic=True)[0]
                    obs, _, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
                rows.append(info)
                if video_dir:
                    imageio.mimsave(str(video_dir / f"seed{seed}_{info['reason']}.mp4"), env.frames, fps=30)
            except RuntimeError as exc:   # scripted grasp failed: counted as a failure, not retried
                rows.append(dict(seed=seed, success=False, reason=str(exc)))
    finally:
        env.close()
        random.setstate(states[0]); np.random.set_state(states[1]); torch.set_rng_state(states[2])
    return dict(successes=sum(r["success"] for r in rows), attempts=len(rows), rows=rows)


class Progress(BaseCallback):
    """Logs every training step to CSV and prints a summary every 1000 steps."""

    def __init__(self, root):
        super().__init__()
        self.file = (root / "training_steps.csv").open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["step", "reward", "xy_error_m", "z_error_m", "ax", "ay", "az", "done", "success", "reason"])
        self.episodes = []

    def _on_step(self):
        info = self.locals["infos"][0]
        self.writer.writerow([self.num_timesteps, float(self.locals["rewards"][0]), info["xy_error_m"],
                              info["z_error_m"], *self.locals["actions"][0], bool(self.locals["dones"][0]),
                              info["success"], info["reason"]])
        if self.locals["dones"][0]:
            self.episodes.append(dict(success=info["success"], reason=info["reason"], seed=info["seed"]))
        if self.num_timesteps % 1000 == 0:
            self.file.flush()
            recent = self.episodes[-20:]
            print(dict(step=self.num_timesteps, episodes=len(self.episodes),
                       recent_success=f"{sum(e['success'] for e in recent)}/{len(recent)}"), flush=True)
        return True


def add_demonstrations(model, n_episodes, noise=0.3, seed=DEMO_SEED_START):
    """Fill SAC's replay buffer with noisy scripted episodes before training.

    RL from scratch wastes thousands of steps waving the arm around. Showing it some
    (imperfect) examples first - "learning from demonstrations" - makes SAC find the
    goal much sooner. The noise gives it variety, so it also sees how to recover.
    """
    rng = np.random.default_rng(seed)
    env = StackSkill()
    added, stacked = 0, 0
    try:
        for k in range(n_episodes):
            try:
                obs, _ = env.reset(seed=seed + k)
            except RuntimeError:
                continue
            while True:
                action = np.clip(env.scripted_action() + rng.normal(0, noise, 3), -1, 1).astype(np.float32)
                next_obs, reward, terminated, truncated, info = env.step(action)
                model.replay_buffer.add(obs[None], next_obs[None], action[None], np.array([reward]),
                                        np.array([terminated or truncated]),
                                        [{"TimeLimit.truncated": truncated and not terminated}])
                added += 1
                obs = next_obs
                if terminated or truncated:
                    stacked += info["success"]
                    break
    finally:
        env.close()
    return dict(episodes=n_episodes, transitions=added, stacked=int(stacked))


class RobustStackSkill(StackSkill):
    """During training, a rare failed scripted grasp just moves on to a new scene."""

    def reset(self, *, seed=None, options=None):
        for _ in range(10):
            try:
                return super().reset(seed=seed, options=options)
            except RuntimeError:
                seed = None
        raise RuntimeError("Scripted grasp failed 10 times in a row")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30000, help="RL steps (one step = one ~1 cm move)")
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--demos", type=int, default=30, help="Scripted demonstration episodes (0 = none)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)   # never overwrite a previous run
    torch.set_num_threads(1)

    config = dict(contract=CONTRACT, steps=args.steps, seed=args.seed, algorithm="SAC",
                  network=[256, 256], learning_starts=500, ent_coef="auto_0.05", buffer_size=100000, batch_size=256,
                  gamma=0.98, learning_rate=3e-4, validation_seeds=VALIDATION_SEEDS,
                  test_seeds=TEST_SEEDS, demos=args.demos, device="cpu")
    save(root / "config.json", config)

    env = Monitor(RobustStackSkill(), str(root / "monitor.csv"))
    model = SAC("MlpPolicy", env, device="cpu", seed=args.seed,
                policy_kwargs={"net_arch": config["network"]},
                learning_starts=config["learning_starts"], buffer_size=config["buffer_size"],
                batch_size=config["batch_size"], gamma=config["gamma"],
                learning_rate=config["learning_rate"],
                # Start with little randomness (default 1.0 makes early actions almost pure noise).
                ent_coef=config["ent_coef"], verbose=0)
    callback = Progress(root)
    history, best = [], (-1.0, None)
    start = time.perf_counter()
    if args.demos:
        demo = add_demonstrations(model, args.demos)
        save(root / "demonstrations.json", demo)
        print(f"[demos] added {demo['transitions']} transitions from {demo['episodes']} scripted episodes "
              f"({demo['stacked']} stacked)", flush=True)
    try:
        while model.num_timesteps < args.steps:
            model.learn(total_timesteps=args.eval_every, callback=callback, reset_num_timesteps=False)
            val = evaluate(model, VALIDATION_SEEDS)
            rate = val["successes"] / val["attempts"]
            history.append(dict(step=model.num_timesteps, validation_success=rate,
                                minutes=round((time.perf_counter() - start) / 60, 1)))
            save(root / "validation_history.json", history)
            print(f"[validation] step {model.num_timesteps}: {val['successes']}/{val['attempts']} stacked", flush=True)
            if rate >= best[0]:   # ties go to the later (more trained) checkpoint
                best = (rate, model.num_timesteps)
                model.save(root / "best_model.zip")
            model.save(root / "last_model.zip")

        print("Final test: scripted baseline vs best SAC checkpoint ...", flush=True)
        scripted = evaluate(None, TEST_SEEDS)
        sac = evaluate(SAC.load(root / "best_model.zip", device="cpu"), TEST_SEEDS)
        save(root / "scripted_test.json", scripted)
        save(root / "sac_test.json", sac)
        videos = root / "videos"
        videos.mkdir()
        evaluate(SAC.load(root / "best_model.zip", device="cpu"), VIDEO_SEEDS, video_dir=videos)

        summary = dict(status="done", steps=model.num_timesteps, best_step=best[1],
                       best_validation_success=best[0],
                       scripted_test=f"{scripted['successes']}/{scripted['attempts']}",
                       sac_test=f"{sac['successes']}/{sac['attempts']}",
                       training_episodes=len(callback.episodes),
                       minutes=round((time.perf_counter() - start) / 60, 1))
        save(root / "summary.json", summary)
        (root / "REPORT.md").write_text(
            "# Stacking SAC experiment (StackCube-v1)\n\n"
            f"Trained {summary['steps']} RL steps in {summary['minutes']} min; best checkpoint at step {best[1]}.\n\n"
            f"| Policy | Test success (seeds {TEST_SEEDS[0]}-{TEST_SEEDS[-1]}) |\n|---|---|\n"
            f"| Scripted baseline | {summary['scripted_test']} |\n| SAC (best) | {summary['sac_test']} |\n\n"
            "Scripted grasp/lift and release surround the learned XYZ carry-and-align phase. "
            "Success is ManiSkill's own check: red cube on green cube, static, not grasped.\n",
            encoding="utf-8")
        print(summary, flush=True)
    finally:
        callback.file.close()
        env.close()


if __name__ == "__main__":
    main()
