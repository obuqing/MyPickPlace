"""Carry env variant that streams every physical step to a video writer."""

import numpy as np

from panda_skills.envs.carry import CarrySkillEnv


class RecordedCarryEnv(CarrySkillEnv):
    """Observe every physical step, including scripted initialization and release."""
    def __init__(self, writer, **kwargs):
        self.writer = writer
        self.frame_count = 0
        super().__init__(render_mode='rgb_array', **kwargs)

    def _sim_step(self, action):
        result = super()._sim_step(action)
        self.writer.append_data(np.asarray(self.render(), dtype=np.uint8))
        self.frame_count += 1
        return result
