"""Policies consume only the common numerical carry observation."""

import numpy as np


class ScriptedCarryPolicy:
    def predict(self, observation, deterministic=True):
        # Invert the adapter's 0.15 scale to reuse the original 0.5-gain servo.
        return np.clip(observation["delta"] * (5 / 0.15), -1, 1).astype(np.float32), None
