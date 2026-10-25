"""Single-env Gym interface: scripted physical grasp, learned XYZ transport."""

import json
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np
from scipy.spatial.transform import Rotation

from algorithms.scripted_pick_place import ScriptedPickPlacePolicy, array


CONTRACT = {
    "version": "carry_xyz_v1", "carry_height": 0.18, "position_tolerance": 0.02,
    "max_carry_steps": 120, "max_contact_loss_steps": 3,
    "translation_action_scale": 0.15, "gripper": "closed", "orientation": "downward",
    "observation_sizes": {"qpos": 9, "qvel": 9, "tcp_pose": 7, "object_pose": 7,
                          "object_velocity": 3, "object_angular_velocity": 3,
                          "target_pos": 3, "delta": 3, "is_grasped": 1, "other_pose": 7},
}


class InitializationError(RuntimeError):
    def __init__(self, record):
        self.record = record
        super().__init__(f"Grasp initialization failed: {record['reason']} (seed={record['seed']}, color={record['color']})")


def curriculum_target(start, goal, max_distance):
    """Cap displacement from the physical lifted pose; preserve the goal height."""
    start, goal = np.asarray(start, dtype=np.float32), np.asarray(goal, dtype=np.float32)
    target = goal.copy()
    if max_distance is not None:
        if not np.isfinite(max_distance) or max_distance <= CONTRACT["position_tolerance"]:
            raise ValueError("Curriculum distance must exceed the success tolerance")
        vertical = float(goal[2] - start[2])
        if abs(vertical) >= max_distance:
            raise ValueError("Lift height cannot satisfy curriculum radius")
        horizontal = goal[:2] - start[:2]
        length = float(np.linalg.norm(horizontal))
        radius = np.sqrt(max_distance**2 - vertical**2)
        if length > radius:
            target[:2] = start[:2] + horizontal * (radius / length)
    return target


def carry_outcome(distance, grasped, object_height, contact_loss_steps, wrong_grasped, steps, max_steps=120):
    """No stationary-hold gate; both grasp and 3D target position are required."""
    dropped = object_height < 0.08 or contact_loss_steps >= CONTRACT["max_contact_loss_steps"]
    success = bool(grasped and not dropped and not wrong_grasped and distance <= CONTRACT["position_tolerance"])
    terminated = bool(success or dropped or wrong_grasped)
    truncated = bool(steps >= max_steps and not terminated)
    reason = "arrived" if success else "wrong_object_grasped" if wrong_grasped else "dropped" if dropped else "timeout" if truncated else "running"
    return success, terminated, truncated, reason


class CarrySkillEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 20}

    def __init__(self, target_color="random", render_mode=None, audit_path=None, max_target_distance=None):
        if target_color not in ("red", "blue", "random"):
            raise ValueError("target_color must be red, blue or random")
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
        self.base = gym.make("MyPickPlace-v1", obs_mode="state_dict", control_mode="pd_ee_delta_pose",
                             render_mode=render_mode, max_episode_steps=1000)
        self.initializations = []
        self._obs = None
        self._finished = True
        self._prepared = False

    def _audit(self, record):
        self.initializations.append(record)
        if self.audit_path:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")

    def _state(self):
        extra = self._obs["extra"]
        selected = self.base.unwrapped.cube
        target = array(extra["goal_pos"]).copy()
        target[2] = CONTRACT["carry_height"]
        if self.carry_target is not None:
            target = self.carry_target.copy()
        other = "blue" if self.color == "red" else "red"
        state = {
            "qpos": array(self._obs["agent"]["qpos"]),
            "qvel": array(self._obs["agent"]["qvel"]),
            "tcp_pose": array(extra["tcp_pose"]), "object_pose": array(extra["obj_pose"]),
            "object_velocity": array(selected.linear_velocity),
            "object_angular_velocity": array(selected.angular_velocity),
            "target_pos": target, "delta": target - array(extra["obj_pose"])[:3],
            "is_grasped": np.array([extra["is_grasped"].item()]),
            "other_pose": array(extra[other + "_cube_pose"]),
        }
        # Copies prevent a later physics step from mutating replay-buffer inputs.
        return {key: np.asarray(value, dtype=np.float32).copy() for key, value in state.items()}

    def _sim_step(self, action):
        self._obs, _, _, truncated, info = self.base.step(np.asarray(action, dtype=np.float32))
        return info, bool(truncated.item())

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.episode_seed = int(seed) if seed is not None else int(self.np_random.integers(0, 2**31 - 1))
        self.color = options.get("target_color", self.target_color_setting)
        if self.color == "random":
            self.color = ("red", "blue")[int(self.np_random.integers(2))]
        if self.color not in ("red", "blue"):
            raise ValueError("Unknown target color")
        self._finished = True
        self.carry_target = None
        self._obs, _ = self.base.reset(seed=self.episode_seed)
        self._obs = self.base.unwrapped.select_target("cube_" + self.color)
        self._prepared = True
        if options.get("defer_grasp", False):
            return self._state(), {"scene_prepared": True, "seed": self.episode_seed, "color": self.color}
        return self.initialize_grasp()

    def current_scene(self):
        from algorithms.llm_planner import get_scene_state
        return get_scene_state(self._obs)

    def initialize_grasp(self, object_id=None):
        """Initialize once in the prepared scene, without resetting or teleporting."""
        if not self._prepared:
            raise RuntimeError("Prepare a fresh scene before initializing a grasp")
        if object_id is not None:
            if object_id not in ("cube_red", "cube_blue"):
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

    def _low_level_action(self, xyz):
        action = np.zeros(7, dtype=np.float32)
        action[:3] = np.clip(xyz, -1, 1) * CONTRACT["translation_action_scale"]
        pose = array(self._obs["extra"]["tcp_pose"])
        current = Rotation.from_quat(pose[[4, 5, 6, 3]])
        error = (Rotation.from_quat([1, 0, 0, 0]) * current.inv()).as_rotvec()
        action[3:6] = np.clip(-error * 5, -0.3, 0.3)
        action[6] = -1.0
        return action

    def step(self, action):
        if self._finished:
            raise RuntimeError("Call reset before stepping a finished carry episode")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (3,) or not np.isfinite(action).all():
            raise ValueError("Carry actions must be finite XYZ vectors with shape (3,)")
        action = np.clip(action, -1, 1)
        info, underlying_timeout = self._sim_step(self._low_level_action(action))
        self.steps += 1
        state = self._state()
        grasped = bool(state["is_grasped"][0])
        self.contact_loss_steps = 0 if grasped else self.contact_loss_steps + 1
        distance = float(np.linalg.norm(state["delta"]))
        success, terminated, truncated, reason = carry_outcome(
            distance, grasped, float(state["object_pose"][2]), self.contact_loss_steps,
            bool(info["is_wrong_grasped"].item()), self.steps, CONTRACT["max_carry_steps"],
        )
        if underlying_timeout and not terminated:
            truncated, reason = True, "underlying_episode_limit"
        progress = self.previous_distance - distance
        reward = 20 * progress - 0.5 * distance - 0.01 - 0.001 * float(np.square(action).sum())
        if success:
            reward += 10.0
        elif terminated:
            reward -= 10.0
        self.previous_distance = distance
        self.move_success = success
        self._finished = terminated or truncated
        return state, float(reward), terminated, truncated, {
            "success": success, "move_success": success, "reason": reason, "distance": distance,
            "is_grasped": grasped, "contact_loss_steps": self.contact_loss_steps,
            "initialization_success": True, "initialization_steps": self.init_steps,
            "carry_steps": self.steps, "seed": self.episode_seed, "color": self.color,
        }

    def finish_placement(self):
        """Rule lowering/release in the SAME scene after a successful carry."""
        if self.max_target_distance is not None:
            return {"placement_success": False, "reason": "curriculum_waypoint_only", "placement_steps": 0}
        if not self.move_success:
            return {"placement_success": False, "reason": "carry_not_successful", "placement_steps": 0}
        policy = ScriptedPickPlacePolicy()
        policy.reset()
        policy.phase = "lower"
        policy.grasp_offset = array(self._obs["extra"]["tcp_pose"])[:3] - array(self._obs["extra"]["obj_pose"])[:3]
        stable = 0
        reason = "placement_timeout"
        step = 0
        for _ in range(180):
            action = policy.act(self._obs)
            if policy.failure:
                reason = policy.failure
                break
            info, truncated = self._sim_step(action)
            step += 1
            stable = stable + 1 if info["success"].item() else 0
            if info["is_wrong_grasped"].item():
                reason = "wrong_object_grasped"
                break
            if policy.done:
                reason = "placed" if stable >= 5 else "placement_not_stable"
                break
            if truncated:
                reason = "underlying_episode_limit"
                break
        return {"placement_success": reason == "placed", "reason": reason, "placement_steps": step,
                "total_control_steps": self.init_steps + self.steps + step,
                "final_object_position": array(self._obs["extra"]["obj_pose"])[:3].tolist()}

    def render(self):
        frame = self.base.render()
        return frame[0].cpu().numpy() if hasattr(frame, "cpu") else frame

    def close(self):
        self.base.close()


register(id="MyCarrySkill-v1", entry_point="envs.carry_skill:CarrySkillEnv")
