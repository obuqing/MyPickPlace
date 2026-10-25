"""Opt-in three-object adapter for the unchanged 52-value frozen SAC policy.

Copied initialization boundaries are isolated from the successful two-object env.
The inherited numeric controller, reward, carry termination and release are reused.
Only the nearest non-target pose fits the old policy's other_pose slot. All three
actors remain physical, and wrong grasp checks inspect both non-target actors.
"""
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np
from envs.three_pick_place import COLORS
from envs.carry_skill import CarrySkillEnv, CONTRACT, InitializationError, curriculum_target
from algorithms.scripted_pick_place import ScriptedPickPlacePolicy, array


class ThreeCarryEnv(CarrySkillEnv):
    def __init__(self, target_color="random", render_mode=None, audit_path=None, max_target_distance=None):
        if target_color not in ("red", "blue", "yellow", "random"):
            raise ValueError("target_color must be red, blue, yellow or random")
        self.target_color_setting = target_color
        self.max_target_distance = max_target_distance
        self.carry_target = None
        self.render_mode = render_mode
        self.audit_path = Path(audit_path) if audit_path else None
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            key: spaces.Box(-np.inf, np.inf, shape=(size,), dtype=np.float32)
            for key, size in CONTRACT["observation_sizes"].items()
        })
        # The outer skill owns the RL horizon; initialization/placement steps
        # never count as learned actions and cannot consume that horizon.
        self.base = gym.make("MyPickPlaceThree-v1", obs_mode="state_dict", control_mode="pd_ee_delta_pose",
                             render_mode=render_mode, max_episode_steps=1000)
        self.initializations = []
        self._obs = None
        self._finished = True
        self._prepared = False

    def reset(self, *, seed=None, options=None):
        # Initialize only Gym's RNG. Calling CarrySkillEnv.reset here would
        # physically grasp before the deferred-planning boundary.
        gym.Env.reset(self,seed=seed)
        options = options or {}
        self.episode_seed = int(seed) if seed is not None else int(self.np_random.integers(0, 2**31 - 1))
        self.color = options.get("target_color", self.target_color_setting)
        if self.color == "random":
            self.color = ("red", "blue", "yellow")[int(self.np_random.integers(3))]
        if self.color not in ("red", "blue", "yellow"):
            raise ValueError("Unknown target color")
        self._finished = True
        self.carry_target = None
        self._obs, _ = self.base.reset(seed=self.episode_seed)
        self._obs = self.base.unwrapped.select_target("cube_" + self.color)
        self._prepared = True
        if options.get("defer_grasp", False):
            return self._state(), {"scene_prepared": True, "seed": self.episode_seed, "color": self.color}
        return self.initialize_grasp()

    def initialize_grasp(self, object_id=None):
        """Initialize once in the prepared scene, without resetting or teleporting."""
        if not self._prepared:
            raise RuntimeError("Prepare a fresh scene before initializing a grasp")
        if object_id is not None:
            if object_id not in ("cube_red", "cube_blue", "cube_yellow"):
                raise ValueError("Unknown target object")
            self.color = object_id.removeprefix("cube_")
            self._obs = self.base.unwrapped.select_target(object_id)
        self._prepared = False
        policy = ScriptedPickPlacePolicy()
        policy.reset()
        reason = "initialization_timeout"
        init_steps = 0
        for _ in range(360):
            action = policy.act(self._obs)
            if policy.failure:
                reason = policy.failure
                break
            info, underlying_timeout = self._sim_step(action)
            init_steps += 1
            if info["is_wrong_grasped"].item():
                reason = "wrong_object_grasped"
                break
            if underlying_timeout:
                reason = "underlying_episode_limit"
                break
            if policy.phase == "move":
                state = self._state()
                if state["is_grasped"][0] and state["object_pose"][2] >= 0.17:
                    reason = "initialized"
                else:
                    reason = "invalid_lifted_grasp"
                break
        record = {"seed": self.episode_seed, "color": self.color, "success": reason == "initialized",
                  "reason": reason, "initialization_steps": init_steps,
                  "object_position": array(self._obs["extra"]["obj_pose"])[:3].tolist()}
        self._audit(record)
        if not record["success"]:
            raise InitializationError(record)
        self.init_steps = init_steps
        self.steps = 0
        self.contact_loss_steps = 0
        self.move_success = False
        goal = array(self._obs["extra"]["goal_pos"]).copy()
        goal[2] = CONTRACT["carry_height"]
        self.carry_target = curriculum_target(array(self._obs["extra"]["obj_pose"])[:3], goal, self.max_target_distance)
        self._finished = False
        state = self._state()
        self.previous_distance = float(np.linalg.norm(state["delta"]))
        return state, {"initialization_success": True, "initialization_steps": init_steps,
                       "seed": self.episode_seed, "color": self.color, "distance": self.previous_distance}

    def _state(self):
        state=super()._state()
        poses=[array(self._obs['extra'][color+'_cube_pose']) for color in COLORS if color!=self.color]
        selected=state['object_pose'][:3]
        nearest=min(poses,key=lambda p:float(np.linalg.norm(p[:3]-selected)))
        state['other_pose']=np.asarray(nearest,dtype=np.float32).copy()
        return state

    def current_scene(self):
        from algorithms.llm_planner import get_scene_state
        scene=get_scene_state(self._obs)
        scene['objects'].append(dict(id='cube_yellow',color='yellow',
            position=array(self._obs['extra']['yellow_cube_pose'])[:3].tolist()))
        return scene


register(id='MyCarryThree-v1',entry_point='envs.three_carry:ThreeCarryEnv')
