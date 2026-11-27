"""Shared episode evaluation for the CPU SAC skills."""

from contextlib import closing, contextmanager
import json
import random

import imageio.v2 as imageio
import numpy as np
import torch


@contextmanager
def preserve_random_state():
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def evaluate_episodes(env, model, seeds, *, video_path=None, fps=30, failure_fields=None):
    rows = []
    with closing(env):
        for seed in seeds:
            try:
                obs, _ = env.reset(seed=seed)
                while True:
                    action = env.scripted_action() if model is None else model.predict(obs, deterministic=True)[0]
                    obs, _, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
                rows.append(info)
                if video_path is not None:
                    imageio.mimsave(str(video_path(seed, info)), env.frames, fps=fps)
            except RuntimeError as exc:
                rows.append(dict(seed=seed, **(failure_fields or {}), success=False, reason=str(exc)))
    return rows


def summarize(rows):
    return dict(successes=sum(row["success"] for row in rows), attempts=len(rows), rows=rows)


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
