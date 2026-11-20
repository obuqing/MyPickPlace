"""Planning evaluation metrics independent of historical experiment runners."""
import statistics


def summarize(rows, total, plan_only):
    positives = [r for r in rows if r["expected_object"] is not None]
    negatives = [r for r in rows if r["expected_object"] is None]
    times = [r["planning_seconds"] for r in rows]
    return {
        "source": "real_llm", "completed_cases": len(rows), "total_cases": total,
        "valid_plans": sum(r["valid_plan"] for r in rows),
        "correct_interpretations": sum(r["semantic_correct"] for r in rows),
        "correct_object_selections": sum(r["semantic_correct"] for r in positives),
        "supported_cases": len(positives),
        "correct_rejections": sum(r["semantic_correct"] for r in negatives),
        "unsupported_cases": len(negatives),
        "false_acceptances": sum(r["valid_plan"] and r.get("plan", {}).get("status") == "execute" for r in negatives),
        "wrong_object_selections": sum(r["valid_plan"] and r.get("plan", {}).get("status") == "execute" and not r["semantic_correct"] for r in positives),
        "invalid_rejections": sum(not r["valid_plan"] and r.get("raw_plan", {}).get("status") == "reject" for r in rows),
        "end_to_end_successes": None if plan_only else sum(bool(r.get("execution", {}).get("success")) for r in positives),
        "mean_planning_seconds": statistics.mean(times) if times else None,
        "first_planning_seconds": times[0] if times else None,
        "warm_mean_planning_seconds": statistics.mean(times[1:]) if len(times) > 1 else None,
        "model_calls": sum(len(r.get("llm_exchange", {}).get("attempts", [None])) for r in rows),
        "repair_attempts": sum(r.get("llm_exchange", {}).get("repair_attempts", 0) for r in rows),
    }
