"""Current planner metrics must not count wrong/invalid plans as successes."""

import json
from pathlib import Path
import sys
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT))

from algorithms.llm_planner import LLMPlanner, PROMPTS
from algorithms.planning_metrics import summarize


class ComparisonTests(unittest.TestCase):
    def test_wrong_object_and_false_accept_are_not_successes(self):
        rows = [
            {"expected_object": "cube_red", "valid_plan": True, "semantic_correct": True,
             "plan": {"status": "execute"}, "execution": {"success": True}},
            {"expected_object": "cube_red", "valid_plan": True, "semantic_correct": False,
             "plan": {"status": "execute"}},
            {"expected_object": None, "valid_plan": True, "semantic_correct": False,
             "plan": {"status": "execute"}},
            {"expected_object": None, "valid_plan": False, "semantic_correct": False,
             "raw_plan": {"status": "reject"}},
        ]
        for row in rows:
            row["planning_seconds"] = 1.0
        score = summarize(rows, 4, False)
        self.assertEqual(score["correct_interpretations"], 1)
        self.assertEqual(score["end_to_end_successes"], 1)
        self.assertEqual(score["supported_cases"], 2)
        self.assertEqual(score["false_acceptances"], 1)
        self.assertEqual(score["wrong_object_selections"], 1)
        self.assertEqual(score["invalid_rejections"], 1)
        self.assertIsNone(summarize(rows, 4, True)["end_to_end_successes"])

    def test_prompt_versions_are_explicit_and_logged(self):
        for version in PROMPTS:
            planner = LLMPlanner(provider="ollama", prompt_version=version)
            self.assertEqual(planner.prompt_version, version)
            self.assertEqual(planner.system_prompt, PROMPTS[version])
            self.assertEqual(len(planner.prompt_sha256), 64)

    def test_new_cases_are_separate_and_absent_from_prompt(self):
        old = json.loads((PROJECT / "configs/english_instruction_cases.json").read_text())
        new = json.loads((PROJECT / "configs/english_instruction_holdout.json").read_text())
        self.assertEqual(len(new), 20)
        self.assertEqual(sum(case["expected_object"] is not None for case in new), 12)
        self.assertFalse({case["instruction"] for case in old} & {case["instruction"] for case in new})
        for case in old + new:
            self.assertNotIn(case["instruction"], PROMPTS["two_stage"])


if __name__ == "__main__":
    unittest.main()
