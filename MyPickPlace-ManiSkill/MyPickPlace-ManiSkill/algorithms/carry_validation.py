"""Isolated validation and explicit best-checkpoint selection."""

import csv
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys


def selection_score(summary):
    s = summary["summary"]["all"]
    count = s["attempted"]
    if count <= 0:
        raise ValueError("Validation requires attempted episodes")
    distance = s.get("mean_final_distance")
    distance = float(distance) if distance is not None and math.isfinite(float(distance)) else math.inf
    return (s["move_successes"]/count, s["placement_successes"]/count,
            s["initialization_successes"]/count, -distance)


class CarryValidation:
    def __init__(self, output, episodes=5, seed=2000):
        self.output = Path(output)
        self.episodes = episodes
        self.seed = seed
        self.history = []
        self.best_score = None
        self.best_step = None

    def run(self, model):
        step = model.num_timesteps
        checkpoints = self.output / "checkpoints"
        checkpoints.mkdir(exist_ok=True)
        checkpoint = checkpoints / f"step_{step}.zip"
        model.save(checkpoint)
        directory = self.output / "validation" / f"step_{step}"
        directory.mkdir(parents=True)
        script = Path(__file__).resolve().parents[1] / "scripts/evaluate_carry.py"
        status = {"phase": "validation", "step": step, "episodes_per_color": self.episodes, "seed_start": self.seed}
        (self.output / "validation_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        # A child process protects training RNG, live physics and rollout state.
        with (directory / "evaluation.log").open("w", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-u", str(script), "--policy", "sac", "--model", str(checkpoint),
                            "--episodes", str(self.episodes), "--seed", str(self.seed), "--output", str(directory)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads((directory / "summary.json").read_text())
        score = selection_score(report)
        improved = self.best_score is None or score > self.best_score
        if improved:
            self.best_score, self.best_step = score, step
            shutil.copyfile(checkpoint, self.output / "best_model.zip")
            (self.output / "best_model.json").write_text(json.dumps({
                "step": step, "gradient_updates": model._n_updates, "checkpoint": str(checkpoint),
                "criteria": ["carry rate over all attempts", "full placement rate", "initialization rate", "lower final distance"],
                "validation": report,
            }, indent=2), encoding="utf-8")
        entry = {"step": step, "gradient_updates": model._n_updates, "is_new_best": improved,
                 "best_step": self.best_step, "summary": report["summary"], "checkpoint": str(checkpoint)}
        self.history.append(entry)
        (self.output / "validation_history.json").write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        with (self.output / "validation_history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["step", "carry_rate", "placement_rate", "initialization_failures", "mean_final_distance",
                             "mean_min_distance", "mean_return", "error_x", "error_y", "error_z", "action_x", "action_y", "action_z"])
            for item in self.history:
                s = item["summary"]["all"]
                writer.writerow([item["step"], s["move_successes"]/s["attempted"], s["end_to_end_rate"],
                                 s["initialization_failures"], s["mean_final_distance"], s["mean_min_distance"], s["mean_return"],
                                 *(s["mean_final_abs_error_xyz"] or [None]*3), *(s["mean_action_xyz"] or [None]*3)])
        status.update(phase="validation_complete", best_step=self.best_step)
        (self.output / "validation_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        print({"validation_step": step, "carry_successes": report["summary"]["all"]["move_successes"],
               "attempted": report["summary"]["all"]["attempted"], "best_step": self.best_step}, flush=True)
        return entry
