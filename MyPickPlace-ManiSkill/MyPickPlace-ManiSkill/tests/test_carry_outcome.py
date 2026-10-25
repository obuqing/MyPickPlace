"""Episode semantics: grasp, height and 3D distance cannot be bypassed."""

import unittest
from pathlib import Path
import sys
import numpy as np
from unittest.mock import Mock, patch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.carry_skill import CarrySkillEnv, InitializationError, carry_outcome


class CarryOutcomeTests(unittest.TestCase):
    def outcome(self, **kwargs):
        state = dict(distance=0.01, grasped=True, object_height=0.18,
                     contact_loss_steps=0, wrong_grasped=False, steps=1)
        return carry_outcome(**(state | kwargs))

    def test_arrival_is_immediate_without_a_hold_gate(self):
        self.assertEqual(self.outcome(), (True, True, False, "arrived"))

    def test_xy_alignment_on_table_is_not_carry_success(self):
        distance = float(np.linalg.norm([0, 0, 0.18 - 0.025]))
        success, terminated, _, reason = self.outcome(distance=distance, object_height=0.025)
        self.assertFalse(success)
        self.assertTrue(terminated)
        self.assertEqual(reason, "dropped")

    def test_contact_noise_is_not_success_and_persistent_loss_terminates(self):
        self.assertEqual(self.outcome(grasped=False, contact_loss_steps=2), (False, False, False, "running"))
        self.assertEqual(self.outcome(grasped=False, contact_loss_steps=3), (False, True, False, "dropped"))

    def test_wrong_object_cannot_complete_the_task(self):
        self.assertEqual(self.outcome(wrong_grasped=True), (False, True, False, "wrong_object_grasped"))

    def test_timeout_is_a_truncation_not_terminal_failure(self):
        self.assertEqual(self.outcome(distance=0.2, steps=120), (False, False, True, "timeout"))
        self.assertEqual(self.outcome(steps=120), (True, True, False, "arrived"))

    def test_failed_initializer_is_recorded_once_and_not_retried(self):
        env = CarrySkillEnv.__new__(CarrySkillEnv)
        env.target_color_setting = "red"
        env.audit_path = None
        env.initializations = []
        env.base = Mock()
        obs = {"extra": {"obj_pose": torch.tensor([[0., 0., .025, 1., 0., 0., 0.]])}}
        env.base.reset.return_value = (obs, {})
        env.base.unwrapped.select_target.return_value = obs
        policy = Mock()
        policy.failure = None

        def fail(_):
            policy.failure = "forced_grasp_failure"
            return np.zeros(7)

        policy.act.side_effect = fail
        with patch("envs.carry_skill.ScriptedPickPlacePolicy", return_value=policy):
            with self.assertRaises(InitializationError) as error:
                env.reset(seed=123)
        self.assertEqual(len(env.initializations), 1)
        self.assertFalse(env.initializations[0]["success"])
        self.assertEqual(error.exception.record["initialization_steps"], 0)
        self.assertEqual(env.base.reset.call_count, 1)
        env.base.step.assert_not_called()


if __name__ == "__main__":
    unittest.main()
