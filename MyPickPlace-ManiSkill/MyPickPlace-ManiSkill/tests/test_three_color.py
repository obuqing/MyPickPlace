from copy import deepcopy
import unittest
from algorithms.llm_planner import Plan, PlanningError, SKILLS, validate_plan
from algorithms.three_color_planner import validate_three_plan
from algorithms.intent_planning import compile_intent
from test_intent_planning import SCENE,intent


class ThreeColorTests(unittest.TestCase):
    def test_reset_refreshes_contacts_without_issuing_a_control_action(self):
        from envs.three_carry import ThreeCarryEnv
        class NoControl(ThreeCarryEnv):
            def _sim_step(self,action):
                raise AssertionError('Preparing the scene must not issue a control action')
        env=NoControl()
        try:
            obs,_=env.reset(seed=90014,options={'defer_grasp':True})
            self.assertEqual(sum(v.size for v in obs.values()),52)
            self.assertFalse(env.current_scene()['robot']['is_grasping'])
            self.assertFalse(any(env.base.unwrapped.last_reset_contact_refresh['after'].values()))
        finally:
            env.close()
    def test_yellow_is_allowed_only_in_explicit_three_object_scene(self):
        scene=deepcopy(SCENE)
        scene['objects'].append(dict(id='cube_yellow',color='yellow',position=[0,0,.025]))
        payload=dict(status='execute',object_id='cube_yellow',target_id='goal',skills=SKILLS,reason='Move yellow')
        self.assertEqual(validate_three_plan(payload,scene).object_id,'cube_yellow')
        with self.assertRaises(PlanningError):
            validate_plan(payload,scene)
        with self.assertRaises(PlanningError):
            validate_three_plan(payload,SCENE)

    def test_same_literal_yellow_intent_depends_on_real_scene_membership(self):
        scene=deepcopy(SCENE)
        scene['objects'].append(dict(id='cube_yellow',color='yellow',position=[0,0,.025]))
        extracted=intent('yellow','the green goal')
        self.assertEqual(compile_intent(extracted,'Move yellow cube to the green goal.',scene)[0]['object_id'],'cube_yellow')
        self.assertEqual(compile_intent(extracted,'Move yellow cube to the green goal.',SCENE)[0]['status'],'reject')
