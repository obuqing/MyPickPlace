"""Three-object execution with disturbance auditing for both non-target objects."""
import numpy as np
from algorithms.three_color_planner import validate_three_plan
from algorithms.llm_planner import PlanningError
from envs.carry_skill import CONTRACT, InitializationError


def execute_three_plan(env,plan,policy):
    scene=env.current_scene()
    plan=validate_three_plan(plan.to_dict(),scene)
    if plan.status!='execute':
        raise PlanningError('Rejected plan cannot execute')
    others={o['id']:np.array(o['position']) for o in scene['objects'] if o['id']!=plan.object_id}
    result=dict(success=False,initialization_success=False,move_success=False,
                placement_success=False,trace=[],object_id=plan.object_id)
    try:
        obs,initial=env.initialize_grasp(plan.object_id)
    except InitializationError as exc:
        result.update(reason='initialization_failed',initialization=exc.record)
        return result
    result.update(initialization_success=True,initialization_steps=initial['initialization_steps'])
    for _ in range(CONTRACT['max_carry_steps']):
        action=policy.predict(obs,deterministic=True)[0]
        obs,reward,terminated,truncated,info=env.step(action)
        result['trace'].append(dict(step=info['carry_steps'],distance=info['distance'],
            error_xyz=obs['delta'].tolist(),action=np.asarray(action).tolist(),
            nearest_other_pose=obs['other_pose'].tolist(),reward=reward))
        if terminated or truncated:
            break
    result.update(move_success=info['move_success'],reason=info['reason'],
                  carry_steps=info['carry_steps'],final_distance=info['distance'])
    if info['move_success']:
        placement=env.finish_placement()
        result.update(placement=placement,placement_success=placement['placement_success'],
                      success=placement['placement_success'],reason=placement['reason'])
    final=env.current_scene()
    displacements={o['id']:float(np.linalg.norm(np.array(o['position'])-others[o['id']]))
                   for o in final['objects'] if o['id'] in others}
    result.update(final_scene=final,other_displacements=displacements,
                  other_objects_undisturbed=all(v<=.01 for v in displacements.values()))
    result['strict_success']=result['success'] and result['other_objects_undisturbed']
    return result
