"""Standalone plots from recorded validation and state/action telemetry."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_plots(training):
    history = json.loads((training / "validation_history.json").read_text())
    steps = [h["step"] for h in history]
    values = [h["summary"]["all"] for h in history]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes[0, 0].plot(steps, [v["move_successes"]/v["attempted"] for v in values], "o-", label="Carry")
    axes[0, 0].plot(steps, [v["end_to_end_rate"] for v in values], "s-", label="Full placement")
    axes[0, 0].set(title="Independent validation success", ylim=(-.05, 1.05), ylabel="Rate")
    axes[0, 0].legend()
    for name, label in (("mean_initial_distance", "Initial"), ("mean_final_distance", "Final"), ("mean_min_distance", "Minimum")):
        axes[0, 1].plot(steps, [v[name] for v in values], "o-", label=label)
    axes[0, 1].set(title="Target distance", ylabel="m")
    axes[0, 1].legend()
    for i, axis in enumerate("XYZ"):
        axes[0, 2].plot(steps, [v["mean_final_abs_error_xyz"][i] if v["mean_final_abs_error_xyz"] else np.nan for v in values], "o-", label=axis)
        axes[1, 0].plot(steps, [v["mean_action_xyz"][i] if v["mean_action_xyz"] else np.nan for v in values], "o-", label=axis)
    axes[0, 2].set(title="Final absolute XYZ error", ylabel="m")
    axes[1, 0].set(title="Mean normalized XYZ action", ylabel="Action", ylim=(-1.05, 1.05))
    axes[0, 2].legend(); axes[1, 0].legend()
    axes[1, 1].plot(steps, [v["mean_return"] for v in values], "o-")
    axes[1, 1].set(title="Validation episode return", ylabel="Return")
    with (training / "training_steps.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if rows:
        for offset in range(0, len(rows), 128):
            block = rows[offset:offset+128]
            axes[1, 2].scatter(int(block[-1]["step"]), np.mean([float(r["distance"]) for r in block]), color="tab:blue", s=15)
    axes[1, 2].set(title="Training distance (128-step bins)", ylabel="m")
    for axis in axes.flat:
        axis.set_xlabel("Learned carry steps")
        axis.grid(alpha=.25)
    fig.savefig(training / "validation_curves.png", dpi=160)
    plt.close(fig)

    # One fixed red and blue validation trajectory per checkpoint, including t=0.
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for h in history:
        path = training / "validation" / f"step_{h['step']}" / "results.json"
        episodes = json.loads(path.read_text())
        for row_index, color in enumerate(("red", "blue")):
            row = next((r for r in episodes if r["color"] == color and r["initialization_success"]), None)
            if row:
                errors = [row["initial_error_xyz"]] + [t["error_xyz"] for t in row["trace"]]
                for axis_index in range(3):
                    axes[row_index, axis_index].plot(np.arange(len(errors)), np.array(errors)[:, axis_index], label=str(h["step"]))
    for i, color in enumerate(("red", "blue")):
        for j, axis in enumerate("XYZ"):
            axes[i, j].set(title=f"{color}: target - object ({axis})", xlabel="Carry control step", ylabel="m")
            axes[i, j].grid(alpha=.25)
    axes[0, 0].legend(title="Training steps")
    fig.savefig(training / "validation_xyz_trajectories.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("training", type=Path)
    make_plots(parser.parse_args().training)
