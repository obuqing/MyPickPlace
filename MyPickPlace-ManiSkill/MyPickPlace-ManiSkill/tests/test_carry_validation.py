import csv
import io
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import numpy as np
from algorithms.carry_validation import selection_score
from train_carry import Progress


def summary(carry=0,placed=0,initialized=10,distance=.2):
    return {"summary":{"all":{"attempted":10,"move_successes":carry,"placement_successes":placed,
                               "initialization_successes":initialized,"mean_final_distance":distance}}}


class ValidationTests(unittest.TestCase):
    def test_success_overrides_closer_failure(self):
        self.assertGreater(selection_score(summary(carry=1,distance=.03)),selection_score(summary(distance=.021)))

    def test_all_zero_success_uses_distance_not_training_step(self):
        self.assertGreater(selection_score(summary(distance=.15)),selection_score(summary(distance=.24)))
        self.assertEqual(selection_score(summary()),selection_score(summary()))

    def test_failed_initializations_cannot_improve_score_by_omission(self):
        self.assertGreater(selection_score(summary(initialized=10,distance=.24)),selection_score(summary(initialized=1,distance=.01)))

    def test_terminal_trace_uses_terminal_state_not_autoreset_state(self):
        callback=Progress.__new__(Progress)
        callback.stream=io.StringIO(); callback.writer=csv.writer(callback.stream)
        callback.episodes=[]; callback.num_timesteps=1
        callback.locals={"infos":[{"terminal_observation":{"delta":np.array([.01,0,0])},"reason":"arrived"}],
                         "new_obs":{"delta":np.array([[.5,0,0]])},"actions":np.zeros((1,3)),
                         "rewards":np.array([10]),"dones":np.array([True])}
        callback._on_step()
        row=next(csv.reader(io.StringIO(callback.stream.getvalue())))
        self.assertAlmostEqual(float(row[2]),.01)


if __name__=="__main__":
    unittest.main()
