"""Literal intent extraction contract and deterministic skill-plan compiler."""

import re

from algorithms.source_spans import (
    SPAN_INTENT_SCHEMA, SPAN_SYSTEM_PROMPT, SpanEvidenceError, SpanFormatError,
    materialize_intent,
)
from algorithms.correction_checks import bind_role_correction, independent_task_evidence


def _object(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


LEGACY_INTENT_SCHEMA = _object({
    "tasks": {"type": "array", "items": _object({
        "action": {"type": "string", "enum": ["place", "pick_up", "stack", "other"]},
        "action_evidence": {"type": "string"},
        "object_quantity": {"type": "string", "enum": ["one", "multiple", "unspecified"]},
        "object_color": {"type": ["string", "null"]},
        "object_evidence": {"type": "string"},
        "destination_type": {"type": "string", "enum": ["goal", "relative", "coordinates", "other", "unspecified"]},
        "destination_color": {"type": ["string", "null"]},
        "destination_evidence": {"type": "string"},
    })},
    "correction_evidence": {"type": "string"},
    "reason": {"type": "string"},
})

LEGACY_INTENT_SYSTEM_PROMPT = """Extract the literal requested tasks from an English instruction.
Return only JSON matching the schema. You are a language extractor, NOT a planner:
do not choose available objects, decide what a robot can do, output skills, or
rewrite an impossible request into a possible one. No scene is supplied on purpose.

List all independent positive tasks in tasks. A prohibition or instruction to
leave something untouched is NOT a positive task. With no positive task, use [].
A correction cancels the earlier request: extract only the final corrected task,
and quote the corrective clause in correction_evidence. Otherwise use "".
Multiple independent moves are separate tasks. A request to move both/all cubes
can be one task with object_quantity=multiple, never silently a single object.

For every task:
- action is place for a request to transfer/move/put an object to a destination;
  pick_up for lifting only, stack for stacking, and other for other actions.
- action_evidence is an EXACT substring containing the requested action.
- object_quantity describes how many objects are requested to move, not how many
  objects are mentioned. Use one only for a uniquely specified single object;
  multiple for both/all/several; unspecified for an unresolved reference/free choice.
- object_color is the literal requested color in lowercase, or null if unspecified.
  Preserve ANY color, including colors that might not exist. Do not infer red or
  blue from "the cube", "that one", "any cube", or the destination's color.
- object_evidence quotes the EXACT noun phrase/color identifying the active object.
  Excluded/distractor objects are not the requested object. Keep quantity words.
  If a color is assigned later in a sentence, quote the complete binding clause,
  not a colorless phrase such as "the cube". Do not quote an excluded object.
- destination_type is goal for a goal/target/marked region, relative for next to,
  right/left of, on/under another object, coordinates for a numeric location,
  other for another explicit destination, unspecified if none is given.
- destination_color is the color EXPLICITLY stated for the destination, otherwise
  null. Preserve an unavailable color; never replace blue/purple with green.
- destination_evidence is an EXACT substring naming the requested destination.
  For a correction, a destination may be inherited from the earlier clause.

All evidence strings must be copied verbatim from the instruction, not paraphrased.
correction_evidence must include the COMPLETE replacement clause and its final
requested object, not just the correction cue or cancelled color. Exclusion or
contrast alone is not a correction. Before returning, ensure object_color and
object_evidence describe the final replacement, rather than only saying so in reason.
Unknown or ambiguous requests must still be described literally; the program will
decide whether they can execute. Do not emit a status, scene ID, or default object.

Examples:
Input: "Transfer the orange item into the violet marked area."
Output: {"tasks":[{"action":"place","action_evidence":"Transfer","object_quantity":"one","object_color":"orange","object_evidence":"the orange item","destination_type":"goal","destination_color":"violet","destination_evidence":"the violet marked area"}],"correction_evidence":"","reason":"One orange item is requested for the violet area."}
Input: "Take red to the marker; scratch that, take blue there."
Output: {"tasks":[{"action":"place","action_evidence":"take","object_quantity":"one","object_color":"blue","object_evidence":"blue","destination_type":"goal","destination_color":null,"destination_evidence":"the marker"}],"correction_evidence":"scratch that, take blue there","reason":"The correction replaces red with blue."}
Input: "Shift all blocks to the marker."
Output: {"tasks":[{"action":"place","action_evidence":"Shift","object_quantity":"multiple","object_color":null,"object_evidence":"all blocks","destination_type":"goal","destination_color":null,"destination_evidence":"the marker"}],"correction_evidence":"","reason":"All blocks are requested to move."}
Input: "Leave red untouched."
Output: {"tasks":[],"correction_evidence":"","reason":"There is no positive manipulation request."}
"""

INTENT_SCHEMA = SPAN_INTENT_SCHEMA
INTENT_SYSTEM_PROMPT = SPAN_SYSTEM_PROMPT


class IntentFormatError(ValueError):
    pass


def validate_intent_structure(intent):
    if not isinstance(intent, dict) or set(intent) != set(LEGACY_INTENT_SCHEMA["required"]):
        raise IntentFormatError("Intent must contain exactly tasks, correction_evidence and reason")
    if not isinstance(intent["tasks"], list) or not all(isinstance(intent[k], str) for k in ("correction_evidence", "reason")):
        raise IntentFormatError("Invalid intent field types")
    if not intent["reason"].strip():
        raise IntentFormatError("Intent reason cannot be empty")
    properties = LEGACY_INTENT_SCHEMA["properties"]["tasks"]["items"]["properties"]
    for task in intent["tasks"]:
        if not isinstance(task, dict) or set(task) != set(properties):
            raise IntentFormatError("Each intent task must contain exactly the declared fields")
        for key, schema in properties.items():
            value = task[key]
            nullable = isinstance(schema["type"], list)
            if not isinstance(value, str) and not (nullable and value is None):
                raise IntentFormatError(f"Invalid intent field type: {key}")
            if "enum" in schema and value not in schema["enum"]:
                raise IntentFormatError(f"Invalid intent field value: {key}")
    return intent


def _contains_word(text, word):
    return bool(re.search(r"\b" + re.escape(word) + r"\b", text, re.IGNORECASE))


def compile_intent(intent, instruction, scene, require_spans=False):
    """Ground extracted evidence, reject unsupported intent, emit canonical plan."""
    if isinstance(intent, dict) and ("correction" in intent or "correction_ref" in intent):
        try:
            grounded, positions = materialize_intent(intent, instruction)
        except SpanFormatError as exc:
            raise IntentFormatError(str(exc)) from None
        except SpanEvidenceError as exc:
            reason = str(exc)
            return {"status": "reject", "object_id": None, "target_id": None, "skills": [], "reason": reason}, [
                {"passed": False, "reason": reason, "repairable": True}]
        plan, checks = compile_intent(grounded, instruction, scene)
        return plan, [{"passed": True, "reason": "Source token spans resolved to exact character positions",
                       "correction_scope": next((p["scope"] for p in positions if p["field"] == "resolved_correction_scope"),
                                                 intent.get("correction", {}).get("scope", "none")),
                       "source_positions": positions, "grounded_intent": grounded}] + checks
    if require_spans:
        raise IntentFormatError("Model extraction must use token spans and explicit correction scope")
    validate_intent_structure(intent)
    intent, correction_binding = bind_role_correction(intent, instruction)
    checks = [correction_binding] if correction_binding else []

    def reject(reason, repairable=False):
        checks.append({"passed": False, "reason": reason, "repairable": repairable})
        return {"status": "reject", "object_id": None, "target_id": None,
                "skills": [], "reason": reason}, checks

    tasks = intent["tasks"]
    extra_moves = independent_task_evidence(instruction)
    if extra_moves:
        checks.append({"passed": False, "reason": "Source contains additional independent positive placement tasks",
                       "source_positions": extra_moves})
        return reject("Multiple independent placement tasks are explicit in the source instruction")
    if len(tasks) != 1:
        return reject("Exactly one positive manipulation task is required")
    task = tasks[0]
    if task["action"] != "place":
        return reject("Only tabletop pick-and-place is supported")
    if task["object_quantity"] != "one":
        return reject("The instruction must uniquely identify one object; multiple/free-choice objects are unsupported")
    text = instruction.casefold()
    for key in ("action_evidence", "object_evidence", "destination_evidence"):
        evidence = task[key]
        if not evidence.strip() or evidence.casefold() not in text:
            return reject(f"Extracted {key} is not grounded in the original instruction", repairable=True)
    correction = intent["correction_evidence"]
    if correction and correction.casefold() not in text:
        return reject("Correction evidence is not grounded in the instruction", repairable=True)
    color = task["object_color"]
    if not color or not _contains_word(task["object_evidence"], color):
        return reject("The selected object color is not explicitly supported by its quoted reference", repairable=True)
    if correction:
        # Do not silently choose the earlier object when a correction names a new one.
        if not _contains_word(correction, color):
            destination_only = (
                re.search(r"\b(goal|target|marker|region|area|disk|circle|ring)\b", correction, re.IGNORECASE)
                and not re.search(r"\b(cube|block|piece|item|object)\b", correction, re.IGNORECASE)
                and task["destination_evidence"].casefold() in correction.casefold()
                and re.fullmatch(
                    r"\s*(?:(?:actually|instead|rather|sorry|no|on second thought)[,:;\s-]*)?"
                    r"(?:please\s+)?(?:(?:use|choose|select|pick|make that)\s+)?(?:the\s+)?"
                    + re.escape(task["destination_evidence"])
                    + r"\s*(?:instead)?[.!]?\s*",
                    correction, re.IGNORECASE,
                )
            )
            if not destination_only:
                return reject("The corrected clause does not support the selected object color", repairable=True)
            checks.append({"passed": True, "reason": "Destination-only correction inherits the grounded object reference"})
    if re.search(r"\b(both|all|each|every|several)\b", task["object_evidence"], re.IGNORECASE):
        return reject("Object evidence requests multiple objects")
    objects = [obj for obj in scene["objects"] if obj.get("color", obj["id"].removeprefix("cube_")) == color.casefold()]
    if len(objects) != 1:
        return reject("The requested object color does not uniquely exist in the scene")
    if task["destination_type"] != "goal":
        return reject("Relative, coordinate-based and unspecified destinations are unsupported")
    targets = [target for target in scene["targets"] if target["id"] == "goal"]
    if len(targets) != 1:
        return reject("The goal is absent or ambiguous in the scene")
    target = targets[0]
    destination_color = task["destination_color"]
    quoted_colors = re.findall(
        r"\b(red|blue|green|yellow|orange|purple|pink|violet|cyan|magenta|black|white|brown|grey|gray)\b",
        task["destination_evidence"], re.IGNORECASE,
    )
    if not destination_color and quoted_colors:
        # Resolve an omitted slot only from an already grounded, unambiguous quote.
        unique_colors = {color.casefold() for color in quoted_colors}
        if len(unique_colors) != 1:
            return reject("The quoted destination contains conflicting colors")
        destination_color = unique_colors.pop()
        checks.append({"passed": True, "reason": "Resolved omitted destination color from its source quote",
                       "resolved_destination_color": destination_color,
                       "evidence": task["destination_evidence"]})
    if any(color.casefold() != (destination_color or "").casefold() for color in quoted_colors):
        return reject("Destination color slot disagrees with the quoted destination")
    if destination_color:
        if not _contains_word(task["destination_evidence"], destination_color):
            return reject("Destination color is not grounded in its quoted reference")
        if destination_color.casefold() != target.get("color", "green"):
            return reject("The requested destination color does not exist in the scene")
    else:
        unqualified_goal_words = {"the", "a", "an", "goal", "target", "marker", "marked", "region", "area", "disk", "circle", "ring", "destination", "zone", "tabletop", "circular"}
        if any(word.casefold() not in unqualified_goal_words for word in re.findall(r"\w+", task["destination_evidence"])):
            return reject("The destination has an unresolved explicit qualifier; it cannot default to the green goal")
    if not re.search(r"\b(goal|target|marker|marked|region|area|disk|circle|ring|destination|zone)\b", task["destination_evidence"], re.IGNORECASE):
        return reject("The quoted destination does not identify the supported goal")
    checks.append({"passed": True, "reason": "One grounded object and an existing goal; supported action and destination type"})
    return {"status": "execute", "object_id": objects[0]["id"], "target_id": "goal",
            "skills": ["reach", "grasp", "move", "place"],
            "reason": f"Place the explicitly requested {color} cube in the goal"}, checks
