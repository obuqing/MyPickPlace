"""SB3 callbacks and helpers shared by the carry-skill SAC training scripts."""

import csv
import json

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


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
