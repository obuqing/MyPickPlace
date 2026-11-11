"""Score the Llama 3 planner on language_cases.json, and optionally execute the plans.

    .\\run-skills.cmd evaluate_stack_language --output runs/stack_language_eval
    .\\run-skills.cmd evaluate_stack_language --execute --output runs/stack_language_exec
"""
import argparse
import json
from pathlib import Path

from .stack_language import plan as llm_plan

CASES = Path(__file__).with_name("language_cases.json")
DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "stack" / "best_model.zip"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Also run accepted plans in the simulator")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=60000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = json.loads(CASES.read_text(encoding="utf-8"))

    env = model = None
    if args.execute:
        from stable_baselines3 import SAC
        from .three_cube_executor import ThreeCubeStacker
        model, env = SAC.load(args.model, device="cpu"), ThreeCubeStacker()

    rows = []
    try:
        for k, case in enumerate(cases):
            checked, audit = llm_plan(case["instruction"])
            got = "reject" if checked["status"] != "execute" else [[s["pick"], s["onto"]] for s in checked["steps"]]
            row = dict(instruction=case["instruction"], expected=case["expected"], got=got,
                       plan_correct=got == case["expected"], reason=checked["reason"],
                       llm_attempts=len(audit.get("attempts", [])), status=checked["status"])
            if env is not None and checked["status"] == "execute":
                env.reset_scene(args.seed + k)
                result = env.run_plan(checked["steps"], model)
                row.update(executed=True, stacked=result["success"],
                           step_reasons=[s["reason"] for s in result["steps"]])
            rows.append(row)
            print(("OK  " if row["plan_correct"] else "BAD ") + case["instruction"], "->", got,
                  ("| stacked" if row.get("stacked") else "| not stacked") if row.get("executed") else "", flush=True)
    finally:
        if env is not None:
            env.close()

    correct = sum(r["plan_correct"] for r in rows)
    executed = [r for r in rows if r.get("executed")]
    summary = dict(plans_correct=f"{correct}/{len(rows)}",
                   errors=sum(r["status"] == "error" for r in rows),
                   executed_stacked=f"{sum(r['stacked'] for r in executed)}/{len(executed)}" if args.execute else None)
    (args.output / "language_results.json").write_text(json.dumps(dict(summary=summary, rows=rows), indent=2,
                                                                  ensure_ascii=False), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
