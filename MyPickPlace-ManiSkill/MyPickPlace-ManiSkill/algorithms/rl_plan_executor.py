"""Validated symbolic task -> scripted grasp -> frozen SAC carry -> scripted release."""
import numpy as np
from algorithms.llm_planner import PlanningError, validate_plan
from envs.carry_skill import CONTRACT, InitializationError


def execute_rl_plan(environment, plan, policy):
    env = environment.unwrapped
    scene = env.current_scene()
    plan = validate_plan(plan.to_dict(), scene)
    if plan.status != 'execute':
        raise PlanningError('A rejected plan cannot be executed')
    if env.max_target_distance is not None:
        raise PlanningError('Language tasks require the full-distance environment')
    other_id = 'cube_blue' if plan.object_id == 'cube_red' else 'cube_red'
    other_start = np.array(next(o['position'] for o in scene['objects'] if o['id'] == other_id))
    result = dict(success=False, object_id=plan.object_id, target_id=plan.target_id,
                  controller='frozen_sac_xyz', initialization_success=False, move_success=False,
                  placement_success=False, trace=[], skill_results=[])
    try:
        obs, initial = env.initialize_grasp(plan.object_id)
    except InitializationError as exc:
        result.update(reason='initialization_failed', initialization=exc.record)
        return result
    result.update(initialization_success=True, initialization_steps=initial['initialization_steps'])
    result['skill_results'] += [dict(skill='reach', success=True, controller='scripted'),
                                dict(skill='grasp', success=True, controller='scripted')]
    for _ in range(CONTRACT['max_carry_steps']):
        action = policy.predict(obs, deterministic=True)[0]
        obs, reward, terminated, truncated, info = environment.step(action)
        result['trace'].append(dict(step=info['carry_steps'], action=np.asarray(action).tolist(),
                                    error_xyz=obs['delta'].tolist(), distance=info['distance'], reward=reward))
        if terminated or truncated:
            break
    result.update(move_success=info['move_success'], reason=info['reason'], carry_steps=info['carry_steps'],
                  final_distance=info['distance'])
    result['skill_results'].append(dict(skill='move', controller='frozen_sac_xyz', success=info['move_success']))
    if info['move_success']:
        placement = env.finish_placement()
        result.update(placement=placement, placement_success=placement['placement_success'],
                      success=placement['placement_success'], reason=placement['reason'])
        result['skill_results'].append(dict(skill='place', controller='scripted', success=placement['placement_success']))
    final_scene = env.current_scene()
    other_end = np.array(next(o['position'] for o in final_scene['objects'] if o['id'] == other_id))
    result.update(final_scene=final_scene, other_cube_displacement=float(np.linalg.norm(other_end-other_start)))
    return result
