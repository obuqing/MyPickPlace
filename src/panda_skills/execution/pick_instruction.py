"""Plan and execute a single pick-and-place instruction."""

import hashlib
import json

from panda_skills.planning.llm_planner import LLMPlanner, get_scene_state, validate_plan


def execute_pick_instruction(args, log, *, record_control_steps=False):
    """Update the caller's log and return the command exit code."""
    env, planner = None, None
    try:
        # Fail on missing model configuration before constructing the simulator.
        if not args.plan_file:
            planner = LLMPlanner(args.provider, args.model, args.base_url, prompt_version=args.prompt_version)
        import gymnasium as gym
        import mani_skill.envs
        import panda_skills.envs
        from panda_skills.execution.plan_executor import execute_plan

        if args.executor == "rl":
            env = gym.make("MyCarrySkill-v1")
            obs, _ = env.reset(seed=args.seed, options={"defer_grasp": True})
            scene = env.unwrapped.current_scene()
        else:
            env = gym.make(
                "MyPickPlace-v1", obs_mode="state_dict", control_mode="pd_ee_delta_pose",
                render_mode="rgb_array", max_episode_steps=400,
            )
            obs, _ = env.reset(seed=args.seed)
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
            return 2
        if args.plan_only:
            log["status"] = "planned"
            return 0
        if args.executor == "rl":
            import torch
            from stable_baselines3 import SAC
            from panda_skills.execution.rl_plan_executor import execute_rl_plan
            torch.set_num_threads(1)
            log["rl_model"] = str(args.rl_model.resolve())
            log["rl_model_sha256"] = hashlib.sha256(args.rl_model.read_bytes()).hexdigest()
            policy = SAC.load(args.rl_model, device="cpu")
            execution = execute_rl_plan(env, plan, policy)
        else:
            execution = execute_plan(env, plan, obs)
        log["execution"] = execution
        log["status"] = "completed" if execution["success"] else "execution_failed"
        print(f"{log['status']}: {execution['reason']}", flush=True)
        return 0 if execution["success"] else 1
    finally:
        if planner is not None:
            log["llm_exchange"] = planner.exchange
        if env is not None:
            try:
                if record_control_steps:
                    base = getattr(env.unwrapped, "base", env)
                    count = getattr(base.unwrapped, "_elapsed_steps", None)
                    log["control_steps"] = int(count) if count is not None else None
            finally:
                env.close()
