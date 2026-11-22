"""Natural-language instruction -> LLM plan -> validation -> simulated execution."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checkpoint_paths import CARRY_CHECKPOINT, PUSH_CHECKPOINT

from panda_skills.execution.pick_instruction import execute_pick_instruction
from panda_skills.planning.llm_planner import PlanningError
from panda_skills.planning.task_routing import RoutingError, select_skill, complete_log


def run_push(args, output, log):
    if args.executor != 'rl':
        raise PlanningError('Push uses the frozen SAC executor; use --executor rl or omit --executor')
    from skills.push.push_instruction import execute
    result=execute(args.instruction,output/'push',args.push_model,args.seed,args.plan_only,
                   provider=args.provider,llm_model=args.model,base_url=args.base_url)
    log.update(plan=result['plan'],control_steps=result['control_steps'],push_result=result,
               rl_model=str(args.push_model.resolve()))
    log['llm_exchange']=json.loads((output/'push/llm_exchange.json').read_text(encoding='utf-8'))
    if result['planning_status']=='error' or result['execution_status']=='error':
        log.update(status='error',error=result.get('error') or result['plan'].get('reason'))
        return 1
    if result['planning_status']=='reject':
        log['status']='rejected'
        return 2
    if args.plan_only:
        log['status']='planned'
        return 0
    log['execution']=dict(success=result['physical_success'],reason=result['metrics']['reason'],
                          steps=result['control_steps'],controller='frozen_sac_xy',
                          metrics=result['metrics'],video=str(output/'push/execution.mp4'))
    log['rl_model_sha256']=result['checkpoint_sha256']
    log['status']='completed' if result['physical_success'] else 'execution_failed'
    return 0 if result['physical_success'] else 1


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", required=True)
    parser.add_argument('--skill',choices=('auto','pick_place','push'),default='auto')
    parser.add_argument("--provider", choices=("ollama", "openai"))
    parser.add_argument("--model")
    parser.add_argument("--prompt-version", choices=("two_stage",))
    parser.add_argument("--base-url")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--executor", choices=("scripted", "rl"), help='Default: scripted for pick_place, rl for push')
    parser.add_argument("--rl-model", type=Path, default=CARRY_CHECKPOINT)
    parser.add_argument('--push-model',type=Path,default=PUSH_CHECKPOINT)
    parser.add_argument("--plan-file", type=Path, help="Explicit offline replay; never reported as an LLM run")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or Path("runs/instructions") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    log = {
        "instruction": args.instruction, "seed": args.seed,
        "source": "offline_plan_replay" if args.plan_file else "llm",
        "status": "starting", "execution": None,
        "executor": args.executor,
        "skill": None, "environment_id": None, "control_steps": 0,
    }
    exit_code = 0
    try:
        if not args.instruction.strip():
            raise PlanningError("Instruction cannot be empty")
        skill=select_skill(args.instruction,args.skill,offline=bool(args.plan_file))
        args.executor=args.executor or ('rl' if skill=='push' else 'scripted')
        log.update(skill=skill,executor=args.executor,
                   environment_id='MyPushDistance-v1' if skill=='push' else ('MyCarrySkill-v1' if args.executor=='rl' else 'MyPickPlace-v1'))
        if skill=='push':
            return run_push(args,output,log)
        return execute_pick_instruction(args, log, record_control_steps=True)
    except RoutingError as exc:
        log.update(status='rejected',error=str(exc))
        exit_code=2
    except (PlanningError, OSError, ValueError, RuntimeError) as exc:
        log["status"] = "error"
        log["error"] = str(exc)
        print(f"Error: {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        complete_log(log,log['control_steps'])
        path = output / "run.json"
        path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Log: {path}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
