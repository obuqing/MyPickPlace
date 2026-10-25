"""Test-only builders; production positions always come from model extraction."""

from algorithms.source_spans import source_tokens, correction_candidates, source_references


def spanned(extracted, text, scope=None):
    tokens = source_tokens(text)

    def span(quote):
        start = text.casefold().rfind(quote.casefold()) if quote else -1
        if start < 0:
            return {"start": 0, "end": 0}
        end = start + len(quote)
        selected = [t["index"] for t in tokens if t["start_char"] < end and t["end_char"] > start]
        return {"start": min(selected), "end": max(selected) + 1}

    tasks = []
    for task in extracted["tasks"]:
        result = {k: task[k] for k in ("action", "object_quantity", "object_color", "destination_type", "destination_color")}
        result.update(action_span=span(task["action_evidence"]), object_span=span(task["object_evidence"]),
                      color_span=span(task["object_color"]), destination_span=span(task["destination_evidence"]))
        tasks.append(result)
    candidates = correction_candidates(text, tokens)
    scope = scope or ("object" if candidates else "none")
    correction = {"scope": scope, "cue_span": {"start": 0, "end": 0}, "replacement_span": {"start": 0, "end": 0}}
    if scope != "none" and candidates:
        correction.update(candidates[-1])
    return {"tasks": tasks, "correction": correction, "reason": extracted["reason"]}


def referenced(extracted, text):
    numeric = spanned(extracted, text)
    references, candidates = source_references(text)
    tasks = []
    for task in numeric["tasks"]:
        result = {k: task[k] for k in ("action", "object_quantity", "object_color", "destination_type", "destination_color")}
        for name in ("action", "object", "color", "destination"):
            span = task[name + "_span"]
            found = next((r for r in references if r["start"] == span["start"] and r["end"] == span["end"]), None)
            if found is None and name == "destination" and span["start"] != span["end"]:
                found = next((r for r in references if r["id"].startswith("p") and r["start"] <= span["start"] and r["end"] >= span["end"]), None)
            if found is None and span["start"] != span["end"]:
                found = next((r for r in references if r["start"] == span["start"] and r["end"] == span["start"]+1), None)
            result[name + "_ref"] = found["id"] if found else "none"
        tasks.append(result)
    scope = "none"
    if candidates and numeric["tasks"]:
        active = candidates[-1]["replacement_span"]
        inside = lambda span: active["start"] <= span["start"] < span["end"] <= active["end"]
        task = next((t for t in reversed(numeric["tasks"]) if inside(t["color_span"]) or inside(t["destination_span"])), numeric["tasks"][-1])
        changed_object, changed_destination = inside(task["color_span"]), inside(task["destination_span"])
        scope = "both" if changed_object and changed_destination else "object" if changed_object else "destination"
    return {"tasks": tasks, "correction_ref": candidates[-1]["id"] if candidates else "none",
            "correction_scope": scope, "reason": extracted["reason"]}
