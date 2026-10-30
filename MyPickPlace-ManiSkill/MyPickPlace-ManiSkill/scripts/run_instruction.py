"""Natural-language instruction -> LLM plan -> validation -> simulated execution."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.llm_planner import LLMPlanner, PlanningError, get_scene_state, validate_plan


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
    parser.add_argument("--rl-model", type=Path, default=Path("runs/carry_curriculum_40000/training/best_model.zip"))
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
    env, planner = None, None
    exit_code = 0
    try:
        if not args.instruction.strip():
            raise PlanningError("Instruction cannot be empty")
        # Fail on missing model configuration before constructing the simulator.
        if not args.plan_file:
            planner = LLMPlanner(args.provider, args.model, args.base_url, prompt_version=args.prompt_version)
        import gymnasium as gym
        import mani_skill.envs
        import envs
        from algorithms.plan_executor import execute_plan

        if args.executor == "rl":
            env = gym.make("MyCarrySkill-v1")
            obs, info = env.reset(seed=args.seed, options={"defer_grasp": True})
            scene = env.unwrapped.current_scene()
        else:
            env = gym.make(
                "MyPickPlace-v1", obs_mode="state_dict", control_mode="pd_ee_delta_pose",
                render_mode="rgb_array", max_episode_steps=400,
            )
            obs, info = env.reset(seed=args.seed)
            scene = get_scene_state(obs)
        log["initial_scene"] = scene
        if args.plan_file:
            payload = json.loads(args.plan_file.read_text(encoding="utf-8-sig"))
            log["replayed_payload"] = payload
            plan = validate_plan(payload, scene)
        else:
            plan = planner.plan(args.instruction, scene)
        log["plan"] = plan.to_dict()
        if planner and "parsed_intent" in planner.exchange:
            log["intent"] = planner.exchange["parsed_intent"]
        print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2), flush=True)
        if plan.status == "reject":
            log["status"] = "rejected"
            exit_code = 2
        elif args.plan_only:
            log["status"] = "planned"
        else:
            if args.executor == "rl":
                import hashlib
                import torch
                from stable_baselines3 import SAC
                from algorithms.rl_plan_executor import execute_rl_plan
                torch.set_num_threads(1)
                log["rl_model"] = str(args.rl_model.resolve())
                log["rl_model_sha256"] = hashlib.sha256(args.rl_model.read_bytes()).hexdigest()
                policy = SAC.load(args.rl_model, device="cpu")
                log["execution"] = execute_rl_plan(env, plan, policy)
            else:
                log["execution"] = execute_plan(env, plan, obs)
            log["status"] = "completed" if log["execution"]["success"] else "execution_failed"
            exit_code = 0 if log["execution"]["success"] else 1
            print(f"{log['status']}: {log['execution']['reason']}", flush=True)
    except (PlanningError, OSError, json.JSONDecodeError) as exc:
        log["status"] = "error"
        log["error"] = str(exc)
        print(f"Error: {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        if planner is not None:
            log["llm_exchange"] = planner.exchange
        if env is not None:
            env.close()
        path = output / "run.json"
        path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Log: {path}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
