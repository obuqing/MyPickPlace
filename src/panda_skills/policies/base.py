from abc import ABC, abstractmethod
import numpy as np


class Policy(ABC):
    """
    Minimal algorithm interface.

    Any future RL / IL / planning / optimization method only needs to expose:
        reset()
        act(observation) -> action
    """

    def reset(self):
        pass

    @abstractmethod
    def act(self, observation):
        raise NotImplementedError


class RandomPolicy(Policy):
    """Smoke-test policy only. It is not expected to solve the task."""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        return self.action_space.sample()


class ZeroPolicy(Policy):
    """Useful for checking reset/render behavior without robot motion."""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        return np.zeros(self.action_space.shape, dtype=self.action_space.dtype)
