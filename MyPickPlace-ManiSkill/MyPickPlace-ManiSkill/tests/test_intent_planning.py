"""Grounding/compiler checks plus the real transport contract with fake inference."""

from copy import deepcopy
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.intent_planning import INTENT_SCHEMA, LEGACY_INTENT_SCHEMA, IntentFormatError, compile_intent
from algorithms.llm_planner import LLMPlanner, PlanningError
from span_fixtures import spanned, referenced
from algorithms.source_spans import schema_for_source


SCENE = {
    "objects": [{"id": "cube_red", "color": "red", "position": [0, -0.1, .025]},
                {"id": "cube_blue", "color": "blue", "position": [0, .1, .025]}],
    "targets": [{"id": "goal", "color": "green", "position": [.14, 0, .001]}],
    "robot": {"is_grasping": False},
}


def intent(color="red", destination="green goal"):
    return {"tasks": [{"action": "place", "action_evidence": "Move",
                       "object_quantity": "one", "object_color": color,
                       "object_evidence": color + " cube", "destination_type": "goal",
                       "destination_color": "green", "destination_evidence": destination}],
            "correction_evidence": "", "reason": "Literal test fixture"}


class IntentTests(unittest.TestCase):
    def compile(self, extracted, text):
        return compile_intent(extracted, text, SCENE)[0]

    def test_supported_object_generates_fixed_plan(self):
        for color in ("red", "blue"):
            plan = self.compile(intent(color), f"Move the {color} cube to the green goal.")
            self.assertEqual(plan["object_id"], "cube_" + color)
            self.assertEqual(plan["skills"], ["reach", "grasp", "move", "place"])

    def test_unknown_colors_are_not_replaced(self):
        for color in ("green", "yellow", "orange"):
            plan = self.compile(intent(color), f"Move the {color} cube to the green goal.")
            self.assertEqual(plan["status"], "reject")
            self.assertIsNone(plan["object_id"])
            self.assertIsNone(plan["target_id"])
            self.assertEqual(plan["skills"], [])

    def test_ambiguity_and_free_choice_rejected(self):
        extracted = intent()
        extracted["tasks"][0].update(object_quantity="unspecified", object_color=None, object_evidence="any cube")
        self.assertEqual(self.compile(extracted, "Move any cube to the green goal.")["status"], "reject")

    def test_fabricated_evidence_and_color_mismatch_rejected(self):
        for reference in ("red cube", "green cube"):
            extracted = intent()
            extracted["tasks"][0]["object_evidence"] = reference
            self.assertEqual(self.compile(extracted, "Move the green cube to the green goal.")["status"], "reject")

    def test_multiple_tasks_and_quantities_rejected(self):
        extracted = intent()
        extracted["tasks"].append(deepcopy(extracted["tasks"][0]))
        self.assertEqual(self.compile(extracted, "Move red cube and blue cube to green goal.")["status"], "reject")
        extracted = intent()
        extracted["tasks"][0]["object_quantity"] = "multiple"
        self.assertEqual(self.compile(extracted, "Move both cubes to green goal.")["status"], "reject")

    def test_unsupported_or_missing_destination(self):
        for destination_type in ("relative", "coordinates", "other", "unspecified"):
            extracted = intent()
            extracted["tasks"][0]["destination_type"] = destination_type
            self.assertEqual(self.compile(extracted, "Move red cube to green goal.")["status"], "reject")
        for color in (None, "blue"):
            extracted = intent(destination="blue target")
            extracted["tasks"][0]["destination_color"] = color
            self.assertEqual(self.compile(extracted, "Move red cube to blue target.")["status"], "reject")

    def test_missing_destination_color_uses_grounded_quote(self):
        extracted = intent()
        extracted["tasks"][0]["destination_color"] = None
        plan, checks = compile_intent(extracted, "Move red cube to green goal.", SCENE)
        self.assertEqual(plan["status"], "execute")
        self.assertIsNone(extracted["tasks"][0]["destination_color"])
        self.assertEqual(checks[0]["resolved_destination_color"], "green")
        extracted["tasks"][0].update(destination_evidence="green or blue target")
        self.assertEqual(self.compile(extracted, "Move red cube to green or blue target.")["status"], "reject")

    def test_correction_cannot_support_the_old_color(self):
        text = "Move red cube to green goal; actually use blue cube instead."
        extracted = intent()
        extracted["correction_evidence"] = "actually use blue cube instead"
        self.assertNotEqual(self.compile(extracted, text)["object_id"], "cube_red")
        extracted["tasks"][0].update(object_color="blue", object_evidence="blue cube")
        self.assertEqual(self.compile(extracted, text)["object_id"], "cube_blue")

    def test_destination_only_correction_preserves_object_but_checks_goal(self):
        extracted = intent()
        extracted["correction_evidence"] = "Actually, use the green goal"
        text = "Move red cube to blue target. Actually, use the green goal."
        self.assertEqual(self.compile(extracted, text)["object_id"], "cube_red")
        extracted["tasks"][0].update(destination_evidence="blue goal", destination_color="blue")
        extracted["correction_evidence"] = "Actually, use the blue goal"
        self.assertEqual(self.compile(extracted, "Move red cube to green target. Actually, use the blue goal.")["status"], "reject")

    def test_object_change_without_cube_noun_cannot_be_a_destination_correction(self):
        for color in ("blue", "pink", "turquoise"):
            extracted = intent()
            extracted["correction_evidence"] = f"Actually, put {color} in the green goal"
            text = f"Move red cube to green goal. Actually, put {color} in the green goal."
            plan = self.compile(extracted, text)
            self.assertEqual(plan["object_id"], "cube_blue" if color == "blue" else None)

    def test_no_positive_task_and_wrong_action(self):
        extracted = intent()
        extracted["tasks"] = []
        self.assertEqual(self.compile(extracted, "Don't move red.")["status"], "reject")
        extracted = intent()
        extracted["tasks"][0]["action"] = "pick_up"
        self.assertEqual(self.compile(extracted, "Move red cube to green goal.")["status"], "reject")

    def test_model_cannot_supply_a_skill_plan(self):
        extracted = intent()
        extracted["tasks"][0]["skills"] = ["move"]
        with self.assertRaises(IntentFormatError):
            self.compile(extracted, "Move red cube to green goal.")

    def test_default_extractor_does_not_receive_scene(self):
        text = "Move red cube to green goal."
        payload = intent()
        response = {"done": True, "done_reason": "stop", "message": {"content": json.dumps(payload)}}
        with patch.dict(os.environ, {}, clear=True):
            planner = LLMPlanner()
            with patch("algorithms.llm_planner.urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as post:
                plan = planner.plan("Move red cube to green goal.", SCENE)
        body = json.loads(post.call_args.args[0].data)
        self.assertEqual(body["format"], LEGACY_INTENT_SCHEMA)
        self.assertNotIn("scene", json.loads(body["messages"][1]["content"]))
        self.assertEqual(plan.object_id, "cube_red")
        self.assertEqual(planner.exchange["parsed_intent"], payload)
        self.assertEqual(planner.exchange["stages"], ["llm_intent_extraction", "deterministic_plan_compilation"])
        self.assertTrue(planner.exchange["compiler_checks"][0]["passed"])

    @staticmethod
    def response(extracted):
        value = {"done": True, "done_reason": "stop", "message": {"content": json.dumps(extracted)}}
        return io.BytesIO(json.dumps(value).encode())

    def test_one_reextraction_can_fix_a_truncated_reference(self):
        text = "The cube to move to the green goal is blue."
        first = intent("blue")
        first["tasks"][0].update(action_evidence="move", object_evidence="The cube")
        second = deepcopy(first)
        second["tasks"][0]["object_evidence"] = "The cube to move to the green goal is blue"
        second = referenced(second, text)
        planner = LLMPlanner(provider="ollama", prompt_version="two_stage")
        with patch("algorithms.llm_planner.urlopen", side_effect=[self.response(first), self.response(second)]) as post:
            plan = planner.plan(text, SCENE)
        self.assertEqual(plan.object_id, "cube_blue")
        self.assertEqual(post.call_count, 2)
        self.assertEqual(planner.exchange["repair_attempts"], 1)
        self.assertEqual(len(planner.exchange["attempts"]), 2)
        self.assertEqual(json.loads(post.call_args_list[0].args[0].data)["format"], LEGACY_INTENT_SCHEMA)
        self.assertEqual(json.loads(post.call_args_list[1].args[0].data)["format"], schema_for_source(text))
        self.assertEqual(planner.exchange["attempts"][0]["parsed_plan"]["status"], "reject")
        feedback = json.loads(post.call_args.args[0].data)["messages"][1]["content"]
        self.assertNotIn('"scene"', feedback)
        self.assertNotIn('"expected_object"', feedback)

    def test_failed_reextraction_stays_rejected_after_two_calls(self):
        extracted = intent()
        extracted["tasks"][0]["object_evidence"] = "the cube"
        text = "Don't choose red. Move the cube to green goal."
        repaired = referenced(extracted, text)
        planner = LLMPlanner(provider="ollama", prompt_version="two_stage")
        with patch("algorithms.llm_planner.urlopen", side_effect=[self.response(extracted), self.response(repaired)]) as post:
            plan = planner.plan(text, SCENE)
        self.assertEqual(plan.status, "reject")
        self.assertEqual(post.call_count, 2)

    def test_unsupported_object_is_not_retried_into_supported_color(self):
        planner = LLMPlanner(provider="ollama", prompt_version="two_stage")
        with patch("algorithms.llm_planner.urlopen", return_value=self.response(intent("yellow"))) as post:
            plan = planner.plan("Move yellow cube to green goal.", SCENE)
        self.assertEqual(plan.status, "reject")
        self.assertEqual(post.call_count, 1)
        self.assertEqual(planner.exchange["repair_attempts"], 0)

    def test_repair_service_failure_never_executes_the_original_intent(self):
        extracted = intent()
        extracted["tasks"][0]["object_evidence"] = "the cube"
        text = "Don't choose red. Move the cube to green goal."
        planner = LLMPlanner(provider="ollama", prompt_version="two_stage")
        with patch("algorithms.llm_planner.urlopen", side_effect=[self.response(extracted), TimeoutError()]):
            with self.assertRaises(PlanningError):
                planner.plan(text, SCENE)
        self.assertEqual(len(planner.exchange["attempts"]), 2)
        self.assertEqual(planner.exchange["attempts"][0]["parsed_plan"]["status"], "reject")


if __name__ == "__main__":
    unittest.main()
