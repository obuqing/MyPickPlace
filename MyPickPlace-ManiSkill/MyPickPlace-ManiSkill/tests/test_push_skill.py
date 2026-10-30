import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
from skills.push.push_skill import outcome, PushSkill


class OutcomeTests(unittest.TestCase):
    def setUp(self):
        self.start = np.array([0.,0.,.02])
        self.goal = np.array([.1,0.,.02])

    def check(self, position, velocity=(0,0,0), tilt=0, invalid=False):
        return outcome(self.start,self.goal,np.array(position),np.array(velocity),tilt,invalid)['success']

    def test_correct_settled_push(self):
        self.assertTrue(self.check([.098,.002,.02]))

    def test_overshoot_rejected(self):
        self.assertFalse(self.check([.12,0,.02]))

    def test_sideways_motion_rejected(self):
        self.assertFalse(self.check([.1,.03,.02]))

    def test_lift_and_tip_rejected(self):
        self.assertFalse(self.check([.1,0,.04]))
        self.assertFalse(self.check([.1,0,.02],tilt=.5))

    def test_prior_lift_cannot_be_erased_by_landing(self):
        self.assertFalse(self.check([.1,0,.02],invalid=True))

    def test_moving_cube_not_success(self):
        self.assertFalse(self.check([.1,0,.02],velocity=(.1,0,0)))

    def test_no_motion_not_success(self):
        self.assertFalse(self.check([0,0,.02]))


class PhysicsInterfaceTests(unittest.TestCase):
    def test_reset_and_zero_action_do_not_execute_a_hidden_push(self):
        import torch
        torch.set_num_threads(1)
        env = PushSkill(distance_m=.05)
        try:
            obs, _ = env.reset(seed=220002)
            np.testing.assert_allclose(env.goal-env.start,[.05,0,0],atol=1e-6)
            self.assertEqual(obs.shape,(26,))
            self.assertTrue(np.isfinite(obs).all())
            start = env._position().copy()
            fixed_goal = env.goal.copy()
            for _ in range(10):
                env.step(np.zeros(2,dtype=np.float32))
            self.assertLess(np.linalg.norm(env._position()-start),.001)
            np.testing.assert_array_equal(env.goal,fixed_goal)
            with self.assertRaises(ValueError):
                env.step([float('nan'),0])
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
