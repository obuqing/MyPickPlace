"""Token-indexed intent evidence; character offsets are computed by the program."""

import re
from copy import deepcopy


def obj(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


SPAN = obj({"start": {"type": "integer"}, "end": {"type": "integer"}})
NUMERIC_INTENT_SCHEMA = obj({
    "tasks": {"type": "array", "items": obj({
        "action": {"type": "string", "enum": ["place", "pick_up", "stack", "other"]},
        "object_quantity": {"type": "string", "enum": ["one", "multiple", "unspecified"]},
        "object_color": {"type": ["string", "null"]},
        "destination_type": {"type": "string", "enum": ["goal", "relative", "coordinates", "other", "unspecified"]},
        "destination_color": {"type": ["string", "null"]},
        "action_span": SPAN, "object_span": SPAN, "color_span": SPAN, "destination_span": SPAN,
    })},
    "correction": obj({"scope": {"type": "string", "enum": ["none", "object", "destination", "both"]},
                       "cue_span": SPAN, "replacement_span": SPAN}),
    "reason": {"type": "string"},
})

SPAN_INTENT_SCHEMA = obj({
    "tasks": {"type": "array", "items": obj({
        "action": {"type": "string", "enum": ["place", "pick_up", "stack", "other"]},
        "object_quantity": {"type": "string", "enum": ["one", "multiple", "unspecified"]},
        "object_color": {"type": ["string", "null"]},
        "destination_type": {"type": "string", "enum": ["goal", "relative", "coordinates", "other", "unspecified"]},
        "destination_color": {"type": ["string", "null"]},
        "action_ref": {"type": "string"}, "object_ref": {"type": "string"},
        "color_ref": {"type": "string"}, "destination_ref": {"type": "string"},
    })},
    "correction_ref": {"type": "string"},
    "correction_scope": {"type": "string", "enum": ["none", "object", "destination", "both"]},
    "reason": {"type": "string"},
})

SPAN_SYSTEM_PROMPT = """Extract literal manipulation requests from English into JSON.
source_references lists exact source words/phrases with unique IDs and positions.
Select IDs from that list; do not calculate offsets or paraphrase quotations.
No scene is supplied. Preserve unavailable colors and destinations. Never guess an
object for an unspecified reference or change an impossible task into a possible one.

List every independent positive task. Do not list prohibitions or objects to leave
untouched. Use tasks=[] when there is no positive task. Moving both/all cubes has
object_quantity=multiple; free choice or unresolved pronouns has unspecified.
Use one only for a uniquely specified single requested object.

For each task:
- action: place for moving/putting to a destination, pick_up for lifting only,
  stack for stacking, other otherwise. action_ref selects the source action word.
  "Use [color] for [goal/target]" is also a placement request. A correction that
  only changes a color inherits the earlier action and destination; do not label
  the corrective words "I meant" as a new other action. Keep the earlier action_ref.
- object_color: final literal requested color or null. color_ref MUST select the
  exact word matching object_color, never an excluded/cancelled color.
- object_ref selects the active object phrase. It can simply equal color_ref,
  especially when a corrected request names only the color. For "the cube ... is
  orange, not violet", select the orange word as color_ref, not violet.
- destination_type: goal for goal/target/marked region, relative for a location
  relative to another object, coordinates for numeric positions, other for another
  explicit destination, unspecified if absent. destination_ref selects the actual
  destination phrase, preserving its stated color. destination_color is that
  explicitly stated color or null. Never convert a blue/purple target to green.
- Missing evidence uses ref="none". Do not invent an ID or silently choose a color.

correction_candidates supplies the final explicit correction with its source ID
and full replacement range. A correction replaces the earlier request, not an
extra independent move. For a correction copy that candidate ID as correction_ref.
Otherwise use "none". Exclusion or contrast alone is NOT a correction.
Set correction_scope explicitly: none, object, destination, or both.
Determine which ROLE the correction changes, not just which colors it mentions.
For a destination-only correction such as "Move orange to the purple target;
correction: use the green goal instead", scope=destination. Keep object_color=orange
and object_ref/color_ref pointing to the ORIGINAL orange token BEFORE the cue.
Keep the original placement action_ref. Only destination_ref/color are replaced.
The word green describes the GOAL, never the moved object. A destination-only
clause need not repeat the unchanged object; do not fill it with the goal's color.
For an object-only correction, inherit the original destination_ref and select the
replacement object's color_ref. For both, select both roles from the replacement.
Preserve inherited unsupported colors too: an orange object remains orange.
When correcting an object, color_ref MUST point to its FINAL replacement word.
A destination can be inherited from the earlier request; a destination-only
correction MUST inherit the object. The program resolves scope from positions;
an object's color_ref must not overlap its destination_ref.

Before returning, verify that the word shown for color_ref equals object_color.
For an object correction use the replacement color, not the first mentioned color.
Return only JSON. No skills, scene IDs, or executable status.
"""

CUE_PATTERN = re.compile(
    r"\b(?:on second thought|scratch that|change that to|cancel that(?: request)?|"
    r"but instead|i meant|i mean|correction|actually)\b|\bwait(?=\s*[,.:])", re.IGNORECASE,
)
COLOR_PATTERN = re.compile(r"\b(red|blue|green|yellow|orange|purple|pink|violet|cyan|magenta|black|white|brown|grey|gray)\b", re.IGNORECASE)


class SpanFormatError(ValueError):
    pass


class SpanEvidenceError(ValueError):
    pass


def source_tokens(text):
    return [{"index": i, "text": match.group(), "start_char": match.start(), "end_char": match.end()}
            for i, match in enumerate(re.finditer(r"\w+(?:['\u2019]\w+)*|[^\w\s]", text))]


def correction_candidates(text, tokens=None):
    tokens = source_tokens(text) if tokens is None else tokens
    candidates = []
    for match in CUE_PATTERN.finditer(text):
        indexes = [t["index"] for t in tokens if t["start_char"] < match.end() and t["end_char"] > match.start()]
        start, end = min(indexes), max(indexes) + 1
        replacement = end
        while replacement < len(tokens) and tokens[replacement]["text"] in (",", ":", ";", "-", "."):
            replacement += 1
        if replacement >= len(tokens) or not any(re.search(r"\w", t["text"]) for t in tokens[replacement:]):
            continue
        candidates.append({"cue_span": {"start": start, "end": end},
                           "replacement_span": {"start": replacement, "end": len(tokens)}})
    return candidates


def source_references(text):
    tokens = source_tokens(text)
    references = [{"id": f"t{t['index']}:{t['text']}", "text": t["text"],
                   "start": t["index"], "end": t["index"] + 1} for t in tokens]
    stops = {"to", "into", "onto", "on", "in", "for", "at", "from", "of", "and", "but", "use", "choose", "place", "put", "move"}
    for t in tokens:
        if t["text"].casefold() not in {"goal", "target", "marker", "region", "area", "disk", "circle", "ring", "destination", "zone"}:
            continue
        start = t["index"]
        while start > max(0, t["index"] - 3):
            previous = tokens[start - 1]["text"]
            if not re.fullmatch(r"\w+", previous) or previous.casefold() in stops:
                break
            start -= 1
            if previous.casefold() in {"the", "a", "an"}:
                break
        quote = text[tokens[start]["start_char"]:t["end_char"]]
        references.append({"id": f"p{start}_{t['index']+1}:{quote}", "text": quote,
                           "start": start, "end": t["index"] + 1})
    candidates = correction_candidates(text, tokens)
    latest = []
    if candidates:
        candidate = candidates[-1]
        span = candidate["cue_span"]
        quote = text[tokens[span["start"]]["start_char"]:tokens[span["end"]-1]["end_char"]]
        latest.append(dict(candidate, id=f"c{span['start']}_{span['end']}:{quote}", text=quote))
    return references, latest


def schema_for_source(text):
    schema = deepcopy(SPAN_INTENT_SCHEMA)
    references, candidates = source_references(text)
    choices = ["none"] + [reference["id"] for reference in references]
    properties = schema["properties"]["tasks"]["items"]["properties"]
    for name in ("action_ref", "object_ref", "color_ref", "destination_ref"):
        properties[name]["enum"] = choices
    literal_words = sorted({r["text"].casefold() for r in references if r["id"].startswith("t") and re.fullmatch(r"\w+", r["text"])})
    for name in ("object_color", "destination_color"):
        properties[name]["enum"] = [None] + literal_words
    schema["properties"]["correction_ref"]["enum"] = ["none"] + [c["id"] for c in candidates]
    return schema


def resolve_reference_intent(payload, instruction):
    validate_schema(payload, schema_for_source(instruction))
    references, candidates = source_references(instruction)
    lookup = {r["id"]: {"start": r["start"], "end": r["end"]} for r in references}
    lookup["none"] = {"start": 0, "end": 0}
    tokens = source_tokens(instruction)
    events = []

    def negated(index):
        return any(t["text"].casefold() in ("not", "don't", "never") for t in tokens[max(0, index-2):index])

    def negative_action(task):
        span = lookup[task["action_ref"]]
        if span["start"] == span["end"]:
            return False
        word = tokens[span["start"]]["text"].casefold()
        tail = []
        for t in tokens[span["end"]:]:
            if t["text"].casefold() in ("and", "but", "while", ".", ";", "!", "?"):
                break
            tail.append(t["text"].casefold())
        return negated(span["start"]) or (word == "leave" and any(w in tail for w in ("alone", "untouched"))) or (
            word in ("stay", "stays") and ("still" in tail or "put" in tail or "where" in tail))

    tasks = []
    for task in payload["tasks"]:
        if negative_action(task):
            events.append({"span": lookup[task["action_ref"]], "event": "excluded_negative_task",
                           "reason": "Source action explicitly prohibits movement or requests leaving the object untouched"})
            continue
        result = {k: task[k] for k in ("action", "object_quantity", "object_color", "destination_type", "destination_color")}
        for name in ("action", "object", "color", "destination"):
            result[name + "_span"] = dict(lookup[task[name + "_ref"]])
        if task["destination_ref"].startswith("t"):
            destination = result["destination_span"]
            phrases = [r for r in references if r["id"].startswith("p") and r["end"] >= destination["end"]
                       and r["start"] <= destination["start"]]
            if phrases:
                phrase = min(phrases, key=lambda r: (r["start"], -r["end"]))
                result["destination_span"] = {"start": phrase["start"], "end": phrase["end"]}
        destination = result["destination_span"]
        if result["destination_type"] == "relative" and destination["start"] < destination["end"]:
            destination_words = {t["text"].casefold() for t in tokens[destination["start"]:destination["end"]]}
            prefix = [t["text"].casefold() for t in tokens[max(0,destination["start"]-4):destination["start"]]]
            relative_words = {"next", "right", "left", "above", "below", "under", "behind", "beside", "between", "near"}
            if destination_words & {"goal", "target", "marker", "region", "area", "disk", "circle", "ring", "destination", "zone"} and not set(prefix) & relative_words and prefix and prefix[-1] in {"to", "into", "onto", "on", "in", "for", "at"}:
                result["destination_type"] = "goal"
                events.append({"span": destination, "event": "grounded_goal_role",
                               "reason": "A source locative preposition directly governs the named goal without a relative relation"})
        if result["action"] == "pick_up" and result["destination_type"] == "goal":
            start, end = result["action_span"]["start"], result["destination_span"]["end"]
            placement_words = {"move", "moved", "put", "place", "transfer", "carry", "transport", "bring", "take", "send", "relocate", "deliver", "set"}
            for t in tokens[start:end]:
                if t["text"].casefold() not in placement_words or negated(t["index"]):
                    continue
                between = tokens[t["index"]+1:result["destination_span"]["start"]]
                bound_object = any(word["text"].casefold() in {(result["object_color"] or "").casefold(), "it"} for word in between)
                if not bound_object:
                    continue
                conflicting = any(COLOR_PATTERN.fullmatch(word["text"]) and word["text"].casefold() != (result["object_color"] or "").casefold()
                                  and not negated(word["index"]) for word in between)
                if conflicting:
                    continue
                result["action"] = "place"
                result["action_span"] = {"start": t["index"], "end": t["index"]+1}
                events.append({"span": result["action_span"], "event": "placement_action_inherited",
                               "reason": "A source placement verb and a grounded destination complete the pickup-and-place request"})
                break
        tasks.append(result)
    correction = {"scope": "none", "cue_span": {"start": 0, "end": 0}, "replacement_span": {"start": 0, "end": 0}}
    if payload["correction_ref"] != "none":
        candidate = candidates[-1]
        correction.update(cue_span=candidate["cue_span"], replacement_span=candidate["replacement_span"])
        active = candidate["replacement_span"]
        inside = lambda span: active["start"] <= span["start"] < span["end"] <= active["end"]
        replacement_tasks = [t for t in tasks if inside(t["color_span"]) or inside(t["destination_span"])]
        if tasks and not replacement_tasks:
            raise SpanEvidenceError("No selected object/destination reference lies in the final replacement range")
        for task in tasks:
            if task not in replacement_tasks:
                events.append({"span": task["color_span"], "event": "cancelled_pre_correction_task",
                               "reason": "This task lies before the explicit final replacement; updated source tasks remain"})
        tasks = replacement_tasks
        if len(tasks) == 1:
            changed_object = inside(tasks[0]["color_span"])
            changed_destination = inside(tasks[0]["destination_span"])
            if not changed_object and not changed_destination:
                raise SpanEvidenceError("No selected object/destination reference lies in the final replacement range")
            derived_scope = "both" if changed_object and changed_destination else "object" if changed_object else "destination"
            correction["scope"] = derived_scope
            events.append({"span": candidate["cue_span"], "event": "resolved_correction_scope",
                           "reason": "Scope follows grounded object/destination positions; a model label cannot change role evidence",
                           "scope": derived_scope, "declared_scope": payload["correction_scope"]})
            reference_span = tasks[0]["object_span"]
            reference = instruction[tokens[reference_span["start"]]["start_char"]:tokens[reference_span["end"]-1]["end_char"]] if reference_span["start"] < reference_span["end"] else ""
            color = tasks[0]["object_color"]
            if changed_object and color and any(c.casefold() != color.casefold() for c in COLOR_PATTERN.findall(reference)):
                tasks[0]["object_span"] = dict(tasks[0]["color_span"])
    elif payload["correction_scope"] != "none":
        raise SpanEvidenceError("A nonempty correction scope requires an explicit correction reference")
    return {"tasks": tasks, "correction": correction, "reason": payload["reason"], "_position_events": events}


def validate_schema(value, schema, path="intent"):
    kind = schema["type"]
    kinds = kind if isinstance(kind, list) else [kind]
    valid = (("null" in kinds and value is None)
             or ("string" in kinds and isinstance(value, str))
             or ("integer" in kinds and type(value) is int)
             or ("object" in kinds and isinstance(value, dict))
             or ("array" in kinds and isinstance(value, list)))
    if not valid or ("enum" in schema and value not in schema["enum"]):
        raise SpanFormatError(f"Invalid field type/value: {path}")
    if kind == "object":
        if set(value) != set(schema["required"]):
            raise SpanFormatError(f"Invalid fields: {path}")
        for key, child in schema["properties"].items():
            validate_schema(value[key], child, path + "." + key)
    elif kind == "array":
        for index, child in enumerate(value):
            validate_schema(child, schema["items"], f"{path}[{index}]")


def materialize_intent(payload, instruction):
    """Resolve model-selected token positions without changing its selected color."""
    if isinstance(payload, dict) and "correction_ref" in payload:
        payload = resolve_reference_intent(payload, instruction)
    events = payload.pop("_position_events", [])
    validate_schema(payload, NUMERIC_INTENT_SCHEMA)
    tokens = source_tokens(instruction)
    positions = []

    def quote(span, label, allow_empty=False):
        start, end = span["start"], span["end"]
        if not (0 <= start <= end <= len(tokens)) or (start == end and not allow_empty):
            raise SpanEvidenceError(f"Invalid {label} token range [{start},{end}); source has {len(tokens)} tokens")
        if start == end:
            return ""
        first, last = tokens[start]["start_char"], tokens[end - 1]["end_char"]
        text = instruction[first:last]
        positions.append({"field": label, "token_start": start, "token_end": end,
                          "start_char": first, "end_char": last, "text": text})
        return text

    for event in events:
        if quote(event["span"], event["event"], allow_empty=True):
            positions[-1]["normalization_reason"] = event["reason"]
            if "scope" in event:
                positions[-1].update(scope=event["scope"], declared_scope=event["declared_scope"])

    result = {"tasks": [], "correction_evidence": "", "reason": payload["reason"] or "Intent extracted from indexed source evidence"}
    correction = payload["correction"]
    candidates = correction_candidates(instruction, tokens)
    # Unsupported counts/actions cannot become an executable plan and need no repair.
    potentially_executable = (len(payload["tasks"]) == 1 and payload["tasks"][0]["action"] == "place"
                              and payload["tasks"][0]["object_quantity"] == "one")
    if potentially_executable:
        if correction["scope"] == "none":
            if candidates:
                raise SpanEvidenceError("An explicit correction cue exists; bind its final replacement span instead of using scope none")
            if any(correction[key]["start"] != 0 or correction[key]["end"] != 0 for key in ("cue_span", "replacement_span")):
                raise SpanEvidenceError("Scope none requires empty correction spans")
        else:
            if not candidates or correction["cue_span"] != candidates[-1]["cue_span"] or correction["replacement_span"] != candidates[-1]["replacement_span"]:
                raise SpanEvidenceError("Correction cue/replacement must match the last supplied correction candidate exactly")
            quote(correction["cue_span"], "correction_cue")
            result["correction_evidence"] = quote(correction["replacement_span"], "correction_replacement")
    for index, task in enumerate(payload["tasks"]):
        legacy = {k: task[k] for k in ("action", "object_quantity", "object_color", "destination_type", "destination_color")}
        if not potentially_executable:
            legacy.update(action_evidence="", object_evidence="", destination_evidence="")
            result["tasks"].append(legacy)
            continue
        action = quote(task["action_span"], f"tasks[{index}].action")
        reference = quote(task["object_span"], f"tasks[{index}].object_reference")
        destination = quote(task["destination_span"], f"tasks[{index}].destination")
        color = task["object_color"]
        literal_color = quote(task["color_span"], f"tasks[{index}].object_color", allow_empty=color is None)
        if (color is not None and task["destination_span"]["start"] <= task["color_span"]["start"]
                < task["color_span"]["end"] <= task["destination_span"]["end"]):
            raise SpanEvidenceError("Object color evidence belongs to the destination phrase, not the moved object")
        if color is not None:
            if literal_color.casefold() != color.casefold():
                raise SpanEvidenceError("object_color disagrees with the exact color_span text; select the final positive color token")
            preceding = [t["text"].casefold() for t in tokens[max(0, task["color_span"]["start"] - 2):task["color_span"]["start"]]]
            if any(word in ("not", "except", "excluding", "don't") for word in preceding):
                raise SpanEvidenceError("The selected color_span is excluded or negated, not the active object")
            other_colors = [c.casefold() for c in COLOR_PATTERN.findall(reference) if c.casefold() != color.casefold()]
            if other_colors and task["color_span"]["start"] >= task["object_span"]["end"]:
                raise SpanEvidenceError("object_span quotes a different/cancelled colored object; point to the final object or its color token")
        active = correction["replacement_span"]
        if correction["scope"] == "destination":
            cue = correction["cue_span"]["start"]
            if not (0 <= task["object_span"]["start"] < task["object_span"]["end"] <= cue
                    and 0 <= task["color_span"]["start"] < task["color_span"]["end"] <= cue):
                raise SpanEvidenceError("Destination-only correction must preserve original pre-cue object evidence")
        if correction["scope"] in ("object", "both") and not (active["start"] <= task["color_span"]["start"] < task["color_span"]["end"] <= active["end"]):
            raise SpanEvidenceError("The selected object color position is outside the final object-correction scope")
        if correction["scope"] in ("destination", "both") and not (active["start"] <= task["destination_span"]["start"] < task["destination_span"]["end"] <= active["end"]):
            raise SpanEvidenceError("The destination position is outside the destination-correction scope")
        binding = {"start": min(task["object_span"]["start"], task["color_span"]["start"]),
                   "end": max(task["object_span"]["end"], task["color_span"]["end"])}
        legacy.update(action_evidence=action, object_evidence=quote(binding, f"tasks[{index}].object_binding"),
                      destination_evidence=destination)
        result["tasks"].append(legacy)
    return result, positions
