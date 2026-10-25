from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from algorithms.llm_planner import Plan, PlanningError, SKILLS
from algorithms.rl_plan_executor import execute_rl_plan
from envs.carry_skill import CarrySkillEnv


SCENE = {'objects':[{'id':'cube_red','position':[0,-.1,.025]},
                    {'id':'cube_blue','position':[0,.1,.025]}],
         'targets':[{'id':'goal','position':[.14,0,.001]}], 'robot':{'is_grasping':False}}


class ExecutionBoundaryTests(unittest.TestCase):
    def environment(self):
        env = Mock()
        env.unwrapped = env
        env.current_scene.return_value = deepcopy(SCENE)
        env.max_target_distance = None
        return env

    def test_rejection_never_initializes_or_calls_policy(self):
        env, policy = self.environment(), Mock()
        plan = Plan('reject',None,None,(),'Ambiguous')
        with self.assertRaises(PlanningError):
            execute_rl_plan(env,plan,policy)
        env.initialize_grasp.assert_not_called()
        policy.predict.assert_not_called()

    def test_live_scene_is_revalidated_before_any_grasp(self):
        env, policy = self.environment(), Mock()
        env.current_scene.return_value['objects'] = []
        plan = Plan('execute','cube_red','goal',tuple(SKILLS),'Move red')
        with self.assertRaises(PlanningError):
            execute_rl_plan(env,plan,policy)
        env.initialize_grasp.assert_not_called()
        policy.predict.assert_not_called()

    def test_short_curriculum_cannot_execute_a_full_language_task(self):
        env, policy = self.environment(), Mock()
        env.max_target_distance = .05
        plan = Plan('execute','cube_blue','goal',tuple(SKILLS),'Move blue')
        with self.assertRaises(PlanningError):
            execute_rl_plan(env,plan,policy)
        env.initialize_grasp.assert_not_called()

    def test_grasp_cannot_be_silently_retried_in_same_prepared_scene(self):
        env = CarrySkillEnv.__new__(CarrySkillEnv)
        env._prepared = False
        with self.assertRaises(RuntimeError):
            env.initialize_grasp('cube_red')
