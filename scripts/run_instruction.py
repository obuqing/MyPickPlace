"""Natural-language instruction -> LLM plan -> validation -> simulated execution."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

from panda_skills.execution.pick_instruction import execute_pick_instruction
from panda_skills.paths import CARRY_CHECKPOINT
from panda_skills.planning.llm_planner import PlanningError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--provider", choices=("ollama", "openai"))
    parser.add_argument("--model")
    parser.add_argument("--prompt-version", choices=("two_stage",))
    parser.add_argument("--base-url")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--executor", choices=("scripted", "rl"), default="scripted")
    parser.add_argument("--rl-model", type=Path, default=CARRY_CHECKPOINT)
    parser.add_argument("--plan-file", type=Path, help="Explicit offline replay; never reported as an LLM run")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path("runs/instructions") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=True)
    log = {
        "instruction": args.instruction, "seed": args.seed,
        "source": "offline_plan_replay" if args.plan_file else "llm",
        "status": "starting", "execution": None,
        "executor": args.executor,
    }
    try:
        if not args.instruction.strip():
            raise PlanningError("Instruction cannot be empty")
        exit_code = execute_pick_instruction(args, log)
    except (PlanningError, OSError, json.JSONDecodeError) as exc:
        log["status"] = "error"
        log["error"] = str(exc)
        print(f"Error: {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        path = output / "run.json"
        path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Log: {path}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
