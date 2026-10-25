"""LLM planning and deterministic validation; no simulator or SDK dependency."""

import json
import hashlib
import math
import os
from dataclasses import asdict, dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from algorithms.intent_planning import (
    INTENT_SCHEMA, INTENT_SYSTEM_PROMPT, LEGACY_INTENT_SCHEMA, LEGACY_INTENT_SYSTEM_PROMPT,
    IntentFormatError, compile_intent,
)
from algorithms.source_spans import source_tokens, correction_candidates, schema_for_source, source_references


SKILLS = ["reach", "grasp", "move", "place"]
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["execute", "reject"]},
        "object_id": {"type": ["string", "null"], "enum": ["cube_red", "cube_blue", None]},
        "target_id": {"type": ["string", "null"], "enum": ["goal", None]},
        "skills": {"type": "array", "items": {"type": "string", "enum": SKILLS}},
        "reason": {"type": "string"},
    },
    "required": ["status", "object_id", "target_id", "skills", "reason"],
    "additionalProperties": False,
}
PROMPTS = {"two_stage": LEGACY_INTENT_SYSTEM_PROMPT}
SYSTEM_PROMPT = LEGACY_INTENT_SYSTEM_PROMPT


class PlanningError(ValueError):
    pass


@dataclass(frozen=True)
class Plan:
    status: str
    object_id: str | None
    target_id: str | None
    skills: tuple[str, ...]
    reason: str

    def to_dict(self):
        value = asdict(self)
        value["skills"] = list(self.skills)
        return value


def _vector(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return [float(x) for x in value[0]]


def get_scene_state(observation):
    extra = observation["extra"]
    return {
        "objects": [
            {"id": f"cube_{color}", "color": color,
             "position": _vector(extra[f"{color}_cube_pose"])[:3]}
            for color in ("red", "blue")
        ],
        "targets": [{"id": "goal", "color": "green", "position": _vector(extra["goal_pos"])}],
        "robot": {"tcp_position": _vector(extra["tcp_pose"])[:3],
                  "is_grasping": bool(extra["is_grasped"].item())},
        "available_skills": SKILLS.copy(),
    }


def validate_plan(payload, scene):
    if not isinstance(payload, dict) or set(payload) != set(PLAN_SCHEMA["required"]):
        raise PlanningError("Plan must contain exactly status, object_id, target_id, skills, reason")
    if not isinstance(payload["reason"], str) or not payload["reason"].strip():
        raise PlanningError("Plan reason must be a nonempty string")
    if payload["status"] == "reject":
        if payload["object_id"] is not None or payload["target_id"] is not None or payload["skills"] != []:
            raise PlanningError("Rejected plans cannot contain executable actions")
    elif payload["status"] == "execute":
        if payload["object_id"] not in ("cube_red", "cube_blue") or payload["target_id"] != "goal":
            raise PlanningError("Unknown object or target")
        if payload["skills"] != SKILLS:
            raise PlanningError("Skills must be exactly reach -> grasp -> move -> place")
        for group, key in (("objects", "object_id"), ("targets", "target_id")):
            entity = next((item for item in scene[group] if item["id"] == payload[key]), None)
            if entity is None:
                raise PlanningError(f"Selected {key} is absent from the scene")
            position = entity["position"]
            if len(position) != 3 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in position):
                raise PlanningError("Scene positions must be finite 3D coordinates")
            x, y, z = position
            if group == "targets":
                bounds = (0.12 - 1e-5 <= x <= 0.16 + 1e-5
                          and abs(y) <= 0.15 + 1e-5 and abs(z - 0.001) < 0.005)
            else:
                bounds = -0.11 <= x <= 0.06 and abs(y) <= 0.16 and abs(z - 0.025) < 0.01
            if not bounds:
                raise PlanningError("Selected object or target is outside the supported initial workspace")
        if scene["robot"]["is_grasping"]:
            raise PlanningError("Planning requires an initially empty gripper")
    else:
        raise PlanningError("Unknown plan status")
    return Plan(payload["status"], payload["object_id"], payload["target_id"], tuple(payload["skills"]), payload["reason"])


class LLMPlanner:
    def __init__(self, provider=None, model=None, base_url=None, timeout=120, prompt_version=None):
        self.prompt_version = prompt_version or os.getenv("LLM_PROMPT_VERSION", "two_stage")
        if self.prompt_version not in PROMPTS:
            raise PlanningError("Only the two_stage planner is supported")
        self.system_prompt = PROMPTS[self.prompt_version]
        self.prompt_sha256 = hashlib.sha256(self.system_prompt.encode("utf-8")).hexdigest()
        self.provider = provider or os.getenv("LLM_PROVIDER", "ollama")
        if self.provider not in ("ollama", "openai"):
            raise PlanningError("LLM_PROVIDER must be ollama or openai")
        self.model = model or os.getenv("LLM_MODEL") or ("llama3:8b" if self.provider == "ollama" else None)
        if not self.model:
            raise PlanningError("Set LLM_MODEL to the model available from your API")
        self.base_url = (base_url or os.getenv("LLM_BASE_URL") or (
            "http://127.0.0.1:11434" if self.provider == "ollama" else "https://api.openai.com/v1"
        )).rstrip("/")
        self.api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if self.provider == "openai" and not self.api_key:
            raise PlanningError("Set LLM_API_KEY or OPENAI_API_KEY in your environment")
        self.timeout = timeout
        self.generation_options = {"temperature": 0, "num_predict": 512}
        self.repair_prompt_sha256 = hashlib.sha256(INTENT_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
        self.exchange = {}

    def plan(self, instruction, scene):
        if not isinstance(instruction, str) or not instruction.strip():
            raise PlanningError("Instruction cannot be empty")
        attempts = []
        feedback = None
        for attempt in range(2):
            try:
                payload = self._request_payload(instruction, scene, feedback)
            except PlanningError:
                attempts.append(dict(self.exchange))
                self.exchange.update(attempts=attempts, repair_attempts=attempt)
                raise
            self.exchange["parsed_intent"] = payload
            try:
                compiled, checks = compile_intent(payload, instruction, scene, require_spans=attempt > 0)
            except IntentFormatError as exc:
                attempts.append(dict(self.exchange))
                self.exchange.update(attempts=attempts, repair_attempts=attempt)
                raise PlanningError(f"Invalid extracted intent: {exc}") from None
            self.exchange["compiler_checks"] = checks
            self.exchange["stages"] = ["llm_intent_extraction", "deterministic_plan_compilation"]
            self.exchange["parsed_plan"] = compiled
            attempts.append(dict(self.exchange))
            repairable = any(check.get("repairable") for check in checks)
            if compiled["status"] == "reject" and repairable and attempt == 0:
                feedback = {"previous_intent": payload, "validation_feedback": compiled["reason"],
                            "repair_instruction": "Bind object and destination roles separately using indexed source evidence and the last applicable correction candidate. Object-only correction selects the replacement object and inherits the original destination. Destination-only correction preserves the latest pre-cue object color/reference and replaces only the destination reference. Both roles must use distinct source evidence; a goal color is not an object color. Preserve all tasks, quantities and unavailable colors/destinations. Never guess or substitute a supported task."}
                continue
            payload = compiled
            self.exchange.update(attempts=attempts, repair_attempts=attempt)
            return validate_plan(payload, scene)

    def _request_payload(self, instruction, scene, feedback=None):
        positional_repair = feedback is not None
        schema = schema_for_source(instruction) if positional_repair else LEGACY_INTENT_SCHEMA
        active_prompt = INTENT_SYSTEM_PROMPT if positional_repair else self.system_prompt
        context = {"instruction": instruction}
        if positional_repair:
            context["source_references"], context["correction_candidates"] = source_references(instruction)
            if feedback is not None and context["correction_candidates"]:
                active = context["correction_candidates"][-1]["replacement_span"]
                tokens = source_tokens(instruction)
                if active["start"] < active["end"]:
                    context["focused_replacement"] = instruction[tokens[active["start"]]["start_char"]:tokens[active["end"] - 1]["end_char"]]
        if feedback is not None:
            context.update(feedback)
        messages = [
            {"role": "system", "content": active_prompt + "\nJSON schema: " + json.dumps(schema)},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ]
        body = {"model": self.model, "messages": messages}
        if self.provider == "ollama":
            endpoint = self.base_url + "/api/chat"
            body.update(stream=False, format=schema, options={"temperature": 0, "num_predict": 1024} if positional_repair else self.generation_options)
        else:
            endpoint = self.base_url + "/chat/completions"
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "task_intent", "strict": True, "schema": schema,
            }}
        self.exchange = {"provider": self.provider, "model": self.model,
                         "prompt_version": self.prompt_version,
                         "prompt_sha256": hashlib.sha256(active_prompt.encode("utf-8")).hexdigest(),
                         "initial_prompt_sha256": self.prompt_sha256,
                         "repair_prompt_sha256": self.repair_prompt_sha256,
                         "extraction_mode": "source_references" if positional_repair else "literal_quotes",
                         "request": body}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        request = Request(endpoint, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            raise PlanningError(f"LLM HTTP {exc.code}; check model, credentials, endpoint and schema support") from None
        except (URLError, TimeoutError, OSError):
            raise PlanningError("Cannot reach LLM; check service, endpoint and timeout. No fallback was executed.") from None
        self.exchange["raw_response"] = raw
        try:
            response = json.loads(raw)
            if self.provider == "ollama":
                if not response.get("done") or response.get("done_reason") not in (None, "stop"):
                    raise PlanningError("LLM returned an incomplete response")
                message = response["message"]
            else:
                choice = response["choices"][0]
                if choice.get("finish_reason") != "stop":
                    raise PlanningError("LLM returned an incomplete response")
                message = choice["message"]
            if message.get("refusal"):
                raise PlanningError("LLM refused the planning request")
            payload = json.loads(message["content"])
        except (KeyError, IndexError, TypeError, AttributeError, json.JSONDecodeError):
            raise PlanningError("LLM response is not a valid JSON plan") from None
        return payload
