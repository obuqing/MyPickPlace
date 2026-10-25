"""Protocol/validation tests use fake API responses, never real LLM inference."""

from copy import deepcopy
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.llm_planner import LLMPlanner, PLAN_SCHEMA, PlanningError, SKILLS, validate_plan
from algorithms.intent_planning import LEGACY_INTENT_SCHEMA


SCENE = {
    "objects": [{"id": "cube_red", "position": [0, -0.1, 0.025]},
                {"id": "cube_blue", "position": [0, 0.1, 0.025]}],
    "targets": [{"id": "goal", "position": [0.14, 0, 0.001]}],
    "robot": {"is_grasping": False},
}


def payload(color="blue"):
    return {"status": "execute", "object_id": "cube_" + color, "target_id": "goal",
            "skills": SKILLS.copy(), "reason": "Test fixture"}


class ValidationTests(unittest.TestCase):
    def test_retired_planners_cannot_be_selected(self):
        for version in ('baseline','v2'):
            with self.assertRaises(PlanningError):
                LLMPlanner(prompt_version=version)
    def test_each_existing_color_is_preserved(self):
        for color in ("red", "blue"):
            with self.subTest(color=color):
                self.assertEqual(validate_plan(payload(color), SCENE).object_id, "cube_" + color)

    def test_reject_is_non_executable(self):
        rejected = {"status": "reject", "object_id": None, "target_id": None,
                    "skills": [], "reason": "Ambiguous instruction"}
        self.assertEqual(validate_plan(rejected, SCENE).status, "reject")
        rejected["skills"] = ["grasp"]
        with self.assertRaises(PlanningError):
            validate_plan(rejected, SCENE)

    def test_invalid_actions_and_unknown_entities(self):
        invalid = [
            {"object_id": "cube_green"}, {"target_id": "somewhere"},
            {"skills": ["grasp", "reach", "move", "place"]},
            {"skills": ["reach", "grasp", "move"]}, {"skills": ["run_python"]},
            {"coordinates": [1, 2, 3]}, {"status": "anything"}, {"reason": ""},
        ]
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(PlanningError):
                validate_plan(payload() | change, SCENE)

    def test_missing_object_or_invalid_workspace(self):
        variants = []
        scene = deepcopy(SCENE)
        scene["objects"].pop()
        variants.append(scene)
        for position in ([0.9, 0, 0.001], [0.14, 0, float("nan")]):
            scene = deepcopy(SCENE)
            scene["targets"][0]["position"] = position
            variants.append(scene)
        scene = deepcopy(SCENE)
        scene["robot"]["is_grasping"] = True
        variants.append(scene)
        for scene in variants:
            with self.subTest(scene=scene), self.assertRaises(PlanningError):
                validate_plan(payload(), scene)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"LLM_PROMPT_VERSION": "two_stage"}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def response(self, provider="ollama", content=None, **extra):
        intent = {"tasks": [{"action": "place", "action_evidence": "Move", "object_quantity": "one",
                            "object_color": "blue", "object_evidence": "blue cube", "destination_type": "goal",
                            "destination_color": "green", "destination_evidence": "the green goal"}],
                  "correction_evidence": "", "reason": "Protocol fixture"}
        message = {"content": json.dumps(intent) if content is None else content}
        if provider == "ollama":
            response = {"done": True, "done_reason": "stop", "message": message} | extra
        else:
            response = {"choices": [{"finish_reason": "stop", "message": message} | extra]}
        return io.BytesIO(json.dumps(response).encode("utf-8"))

    def test_ollama_receives_instruction_and_extraction_schema_without_scene(self):
        with patch("algorithms.llm_planner.urlopen", return_value=self.response()) as post:
            plan = LLMPlanner().plan("Move the blue cube to the green goal.", SCENE)
        request = post.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")
        self.assertFalse(body["stream"])
        self.assertEqual(body["format"], LEGACY_INTENT_SCHEMA)
        self.assertNotIn("scene", json.loads(body["messages"][1]["content"]))
        self.assertIn("blue cube", body["messages"][1]["content"])
        self.assertEqual(plan.object_id, "cube_blue")

    def test_openai_schema_and_credentials_not_in_log(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "test-only-secret"}):
            planner = LLMPlanner("openai", "test-model")
            with patch("algorithms.llm_planner.urlopen", return_value=self.response("openai")) as post:
                planner.plan("Move the blue cube to the green goal.", SCENE)
        request = post.call_args.args[0]
        self.assertEqual(request.headers["Authorization"], "Bearer test-only-secret")
        self.assertTrue(json.loads(request.data)["response_format"]["json_schema"]["strict"])
        self.assertNotIn("test-only-secret", json.dumps(planner.exchange))

    def test_malformed_truncated_and_refused_response(self):
        responses = [self.response(content="not JSON"), self.response(done=False),
                     self.response(done_reason="length"),
                     self.response("openai", finish_reason="length"),
                     io.BytesIO(json.dumps({"choices": [{"finish_reason": "stop", "message": {"refusal": "No"}}]}).encode())]
        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}):
            for index, response in enumerate(responses):
                provider = "ollama" if index < 3 else "openai"
                with self.subTest(index=index), patch("algorithms.llm_planner.urlopen", return_value=response):
                    with self.assertRaises(PlanningError):
                        LLMPlanner(provider, "test-model").plan("blue", SCENE)

    def test_connection_error_does_not_fallback(self):
        for error in (URLError("offline"), TimeoutError(), HTTPError("test", 401, "Unauthorized", {}, None)):
            with self.subTest(error=error), patch("algorithms.llm_planner.urlopen", side_effect=error) as post:
                with self.assertRaises(PlanningError):
                    LLMPlanner().plan("blue", SCENE)
                self.assertEqual(post.call_count, 1)

    def test_remote_configuration_and_empty_instruction(self):
        with self.assertRaises(PlanningError):
            LLMPlanner("openai")
        with self.assertRaises(PlanningError):
            LLMPlanner().plan("  ", SCENE)


if __name__ == "__main__":
    unittest.main()
