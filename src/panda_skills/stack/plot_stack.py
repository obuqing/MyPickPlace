"""Plot the learning curve of a stacking training run.

    .\run-skills.cmd plot_stack --run runs/stack_sac_20000

Saves learning_curve.png inside the run folder.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()

    # Training episodes: success of each episode while learning (with exploration noise).
    rows = list(csv.DictReader((args.run / "training_steps.csv").open(encoding="utf-8")))
    ends = [r for r in rows if r["done"] == "True"]
    steps = np.array([int(r["step"]) for r in ends])
    success = np.array([r["success"] == "True" for r in ends], dtype=float)
    window = 20
    rolling = np.convolve(success, np.ones(window) / window, mode="valid")

    # Validation: deterministic policy on 10 fixed scenes every 2000 steps.
    history = json.loads((args.run / "validation_history.json").read_text(encoding="utf-8"))
    summary = json.loads((args.run / "summary.json").read_text(encoding="utf-8"))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps[window - 1:], 100 * rolling, color="#9aa5b1", label=f"Training episodes ({window}-episode average)")
    ax.plot([h["step"] for h in history], [100 * h["validation_success"] for h in history],
            "o-", color="#2563eb", label="Validation (10 fixed scenes)")
    ax.axvline(summary["best_step"], color="#16a34a", linestyle="--",
               label=f"Best checkpoint (step {summary['best_step']})")
    scripted = summary["scripted_test"].split("/")
    ax.axhline(100 * int(scripted[0]) / int(scripted[1]), color="#dc2626", linestyle=":",
               label=f"Scripted baseline on test ({summary['scripted_test']})")
    ax.set_xlabel("SAC training steps")
    ax.set_ylabel("Stacking success (%)")
    ax.set_ylim(-5, 105)
    ax.set_title(f"StackCube-v1: SAC test {summary['sac_test']} vs scripted {summary['scripted_test']}")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = args.run / "learning_curve.png"
    fig.savefig(out, dpi=150)
    print("Saved", out)


if __name__ == "__main__":
    main()
