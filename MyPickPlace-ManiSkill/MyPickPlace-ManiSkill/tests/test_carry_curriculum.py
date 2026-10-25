import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from envs.carry_skill import curriculum_target
from train_carry_curriculum import promotion_streak, resume_cursor


class CurriculumTests(unittest.TestCase):
    def test_radius_and_height_preserved(self):
        start = np.array([0., 0., .175])
        goal = np.array([.2, .1, .18])
        target = curriculum_target(start, goal, .05)
        self.assertAlmostEqual(float(np.linalg.norm(target-start)), .05, places=6)
        self.assertAlmostEqual(float(target[2]), .18, places=6)

    def test_full_and_already_close_targets_unchanged(self):
        goal = np.array([.02, 0, .18])
        np.testing.assert_allclose(curriculum_target([0,0,.18], goal, .05), goal)
        np.testing.assert_allclose(curriculum_target([0,0,.18], goal, None), goal)

    def test_reject_impossible_height_and_trivial_radius(self):
        with self.assertRaises(ValueError):
            curriculum_target([0,0,.1], [0,0,.18], .05)
        with self.assertRaises(ValueError):
            curriculum_target([0,0,.18], [1,0,.18], .02)

    def test_promotion_needs_consecutive_passes(self):
        self.assertEqual(promotion_streak(8,10,0), 1)
        self.assertEqual(promotion_streak(9,10,1), 2)
        self.assertEqual(promotion_streak(7,10,1), 0)
        self.assertEqual(promotion_streak(0,0,1), 0)

    def test_resume_keeps_last_pass_without_counting_it_twice(self):
        summary = dict(steps=20000, final_stage=0)
        history = [dict(step=20000, stage=0, consecutive_passes=1)]
        self.assertEqual(resume_cursor(summary, history), (20000, 0, 1))
        self.assertEqual(promotion_streak(7,10,resume_cursor(summary,history)[2]), 0)

    def test_resume_rejects_mismatched_checkpoint_metadata(self):
        with self.assertRaises(ValueError):
            resume_cursor(dict(steps=20000,final_stage=0), [dict(step=18000,stage=0,consecutive_passes=1)])
