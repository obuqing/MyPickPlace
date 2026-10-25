"""Execute a validated single-object plan using the existing scripted controller."""

from algorithms.llm_planner import PlanningError, get_scene_state, validate_plan
from algorithms.scripted_pick_place import ScriptedPickPlacePolicy, array


PHASE_SKILL = {
    "approach": "reach", "descend": "reach", "grasp": "grasp", "lift": "grasp",
    "move": "move", "lower": "place", "release": "place", "retreat": "place", "settle": "place",
}


def execute_plan(env, plan, observation, max_steps=400):
    # Revalidate against the live scene at the execution boundary.
    plan = validate_plan(plan.to_dict(), get_scene_state(observation))
    if plan.status != "execute":
        raise PlanningError("A rejected plan cannot be executed")
    obs = env.unwrapped.select_target(plan.object_id)
    policy = ScriptedPickPlacePolicy()
    policy.reset()
    trace, skill_results = [], []
    stable = 0
    skill_index = 0
    other_color = "blue" if plan.object_id == "cube_red" else "red"
    other_start = array(obs["extra"][other_color + "_cube_pose"])[:3].copy()
    for step in range(max_steps):
        previous_skill = PHASE_SKILL[policy.phase]
        if previous_skill != plan.skills[skill_index]:
            raise PlanningError("Controller skill differs from validated plan")
        action = policy.act(obs)
        if policy.failure:
            break
        obs, reward, terminated, truncated, info = env.step(action)
        if info["is_wrong_grasped"].item():
            policy.failure = "wrong_object_grasped"
        stable = stable + 1 if info["success"].item() else 0
        trace.append({
            "step": step + 1, "skill": previous_skill, "phase": policy.phase,
            "object_position": array(obs["extra"]["obj_pose"])[:3].tolist(),
            "grasped": bool(info["is_grasped"].item()),
            "success": bool(info["success"].item()),
        })
        if PHASE_SKILL[policy.phase] != previous_skill and not policy.failure:
            skill_results.append({"skill": previous_skill, "success": True, "step": step + 1})
            skill_index += 1
        # Instantaneous environment success can precede retreat/settling. Keep
        # stepping the same episode until the full sequence has completed.
        if policy.failure or policy.done or truncated.item():
            break
    success = policy.done and not policy.failure and stable >= 5
    reason = "completed" if success else policy.failure or (
        "placement_not_stable" if policy.done else "episode_timeout"
    )
    skill_results.append({"skill": plan.skills[skill_index], "success": bool(success), "reason": reason})
    import numpy as np
    return {
        "success": bool(success), "reason": reason,
        "object_id": plan.object_id, "target_id": plan.target_id,
        "steps": len(trace), "skill_results": skill_results, "trace": trace,
        "other_cube_displacement": float(np.linalg.norm(
            array(obs["extra"][other_color + "_cube_pose"])[:3] - other_start
        )),
        "final_scene": get_scene_state(obs),
    }
