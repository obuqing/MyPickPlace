"""Three-color-only symbolic boundary; original two-color validation is unchanged."""
import math
from algorithms.llm_planner import LLMPlanner, PlanningError, Plan, PLAN_SCHEMA, SKILLS
from algorithms.intent_planning import compile_intent, IntentFormatError


def validate_three_plan(payload, scene):
    if not isinstance(payload, dict) or set(payload) != set(PLAN_SCHEMA["required"]):
        raise PlanningError("Plan must contain exactly status, object_id, target_id, skills, reason")
    if not isinstance(payload["reason"], str) or not payload["reason"].strip():
        raise PlanningError("Plan reason must be a nonempty string")
    if payload["status"] == "reject":
        if payload["object_id"] is not None or payload["target_id"] is not None or payload["skills"] != []:
            raise PlanningError("Rejected plans cannot contain executable actions")
    elif payload["status"] == "execute":
        if payload["object_id"] not in ("cube_red", "cube_blue", "cube_yellow") or payload["target_id"] != "goal":
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


class ThreeColorPlanner(LLMPlanner):
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
            return validate_three_plan(payload, scene)
