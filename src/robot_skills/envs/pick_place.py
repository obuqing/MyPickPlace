from typing import Any
import sys

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("MyPickPlace-v1", max_episode_steps=200)
class MyPickPlaceEnv(BaseEnv):
    """
    Simple Panda pick-and-place task for algorithm development.

    Task:
      1. Pick up the selected red or blue cube (red by default).
      2. Move it to a green target region on the tabletop.
      3. Release it and become static.

    The task intentionally uses a compact state/action interface so that
    learning/planning algorithms can be developed independently of the scene.
    """

    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    cube_half_size = 0.025
    goal_radius = 0.04

    # Workspace used for episode randomization.
    object_x_range = (-0.10, 0.05)
    object_y_range = (-0.15, 0.15)
    # Keep tabletop goals reachable with the Panda's downward-facing gripper.
    goal_x_range = (0.12, 0.16)
    goal_y_range = (-0.15, 0.15)

    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.02,
        target_color="red",
        **kwargs,
    ):
        if target_color not in ("red", "blue"):
            raise ValueError("target_color must be 'red' or 'blue'")
        self.target_color = target_color
        self.robot_init_qpos_noise = robot_init_qpos_noise
        # SAPIEN's CUDA image transfer is unavailable on Windows. Use the
        # standard image readback path for both RGB rendering and the viewer.
        if sys.platform == "win32":
            kwargs.setdefault("sim_backend", "cpu")
            kwargs.setdefault("render_backend", "cpu")
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=[0.35, 0.35, 0.55],
            target=[0.05, 0.0, 0.05],
        )
        return [
            CameraConfig(
                "base_camera",
                pose,
                128,
                128,
                np.pi / 2,
                0.01,
                100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=[0.55, 0.55, 0.55],
            target=[0.05, 0.0, 0.05],
        )
        return CameraConfig(
            "render_camera",
            pose,
            512,
            512,
            1.0,
            0.01,
            100,
        )

    def _load_agent(self, options: dict):
        # Standard tabletop placement used by ManiSkill Panda tasks.
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self,
            robot_init_qpos_noise=self.robot_init_qpos_noise,
        )
        self.table_scene.build()

        self.red_cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[0.9, 0.1, 0.1, 1.0],
            name="cube_red",
            initial_pose=sapien.Pose(
                p=[0.0, 0.0, self.cube_half_size]
            ),
        )
        self.blue_cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[0.1, 0.3, 0.9, 1.0],
            name="cube_blue",
            initial_pose=sapien.Pose(p=[0.0, 0.10, self.cube_half_size]),
        )
        self.cubes = {"red": self.red_cube, "blue": self.blue_cube}
        # Existing target-relative observations and rewards follow this actor.
        self.cube = self.cubes[self.target_color]

        # Thin kinematic green disk used only as a visual target.
        # No collision, so the cube can be placed directly on top of it.
        self.goal_site = actors.build_cylinder(
            self.scene,
            radius=self.goal_radius,
            half_length=0.001,
            color=[0.1, 0.9, 0.1, 0.55],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(
                p=[0.14, 0.0, 0.001], q=[0.70710678, 0, 0.70710678, 0]
            ),
        )
        self._hidden_objects.append(self.goal_site)

    @staticmethod
    def _uniform_xy(
        b: int,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        device,
    ):
        xy = torch.rand((b, 2), device=device)
        xy[:, 0] = xy[:, 0] * (x_range[1] - x_range[0]) + x_range[0]
        xy[:, 1] = xy[:, 1] * (y_range[1] - y_range[0]) + y_range[0]
        return xy

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Separate Y bands leave at least 5 cm between cube surfaces.
            # Randomly swap bands so color does not imply a fixed side.
            swap = torch.rand(b, device=self.device) < 0.5
            for color, cube in self.cubes.items():
                cube_p = torch.zeros((b, 3), device=self.device)
                cube_p[:, :2] = self._uniform_xy(
                    b, self.object_x_range, (-0.15, -0.05), self.device
                )
                upper_band = swap if color == "red" else ~swap
                cube_p[:, 1] += upper_band.float() * 0.20
                cube_p[:, 2] = self.cube_half_size
                cube.set_pose(Pose.create_from_pq(p=cube_p, q=[1, 0, 0, 0]))

            # Random tabletop target position.
            goal_p = torch.zeros((b, 3), device=self.device)
            goal_p[:, :2] = self._uniform_xy(
                b,
                self.goal_x_range,
                self.goal_y_range,
                self.device,
            )
            goal_p[:, 2] = 0.001
            # SAPIEN cylinders run along local X; rotate the disk onto the table.
            self.goal_site.set_pose(Pose.create_from_pq(
                p=goal_p, q=[0.70710678, 0, 0.70710678, 0]
            ))

    def evaluate(self):
        obj_to_goal_xy = self.cube.pose.p[..., :2] - self.goal_site.pose.p[..., :2]
        is_obj_placed = (
            (torch.linalg.norm(obj_to_goal_xy, axis=1) <= self.goal_radius)
            & ((self.cube.pose.p[:, 2] - self.cube_half_size).abs() <= 0.008)
        )

        is_grasped = self.agent.is_grasping(self.cube)
        is_gripper_open = self.agent.robot.get_qpos()[:, -2:].mean(dim=1) > 0.035
        is_released = ~is_grasped & is_gripper_open
        is_robot_static = self.agent.is_static(0.2)
        is_obj_static = (
            (torch.linalg.norm(self.cube.linear_velocity, dim=1) <= 0.03)
            & (torch.linalg.norm(self.cube.angular_velocity, dim=1) <= 0.5)
        )
        other_cube = self.cubes["blue" if self.target_color == "red" else "red"]
        is_wrong_grasped = self.agent.is_grasping(other_cube)

        success = is_obj_placed & is_released & is_robot_static & is_obj_static & ~is_wrong_grasped

        return {
            "success": success,
            "is_obj_placed": is_obj_placed,
            "is_grasped": is_grasped,
            "is_released": is_released,
            "is_robot_static": is_robot_static,
            "is_obj_static": is_obj_static,
            "is_wrong_grasped": is_wrong_grasped,
        }

    def select_target(self, object_id: str):
        """Select an existing cube without resetting or moving the scene."""
        colors = {"cube_red": "red", "cube_blue": "blue"}
        if object_id not in colors:
            raise ValueError(f"Unknown object: {object_id}")
        self.target_color = colors[object_id]
        self.cube = self.cubes[self.target_color]
        return self.get_obs()

    def _get_obs_extra(self, info: dict):
        obs = {
            "tcp_pose": self.agent.tcp_pose.raw_pose,
            "goal_pos": self.goal_site.pose.p,
            "is_grasped": info["is_grasped"],
        }

        if "state" in self.obs_mode:
            obs.update(
                {
                    "obj_pose": self.cube.pose.raw_pose,
                    "red_cube_pose": self.red_cube.pose.raw_pose,
                    "blue_cube_pose": self.blue_cube.pose.raw_pose,
                    "target_color": torch.tensor(
                        [int(self.target_color == "red"), int(self.target_color == "blue")],
                        device=self.device, dtype=torch.float32,
                    ).expand(self.num_envs, -1),
                    "tcp_to_obj_pos": self.cube.pose.p - self.agent.tcp_pose.p,
                    "obj_to_goal_pos": self.goal_site.pose.p - self.cube.pose.p,
                }
            )
        return obs

    def compute_dense_reward(
        self,
        obs: Any,
        action: torch.Tensor,
        info: dict,
    ):
        # Stage 1: reach the object.
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp_pose.p,
            axis=1,
        )
        reaching_reward = 1.0 - torch.tanh(5.0 * tcp_to_obj_dist)

        # Stage 2: grasp.
        is_grasped = info["is_grasped"]
        grasp_reward = is_grasped.float()

        # Stage 3: transport toward target.
        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p[..., :2] - self.cube.pose.p[..., :2],
            axis=1,
        )
        transport_reward = (1.0 - torch.tanh(5.0 * obj_to_goal_dist)) * is_grasped

        # Stage 4: release after placement.
        release_reward = (
            info["is_obj_placed"] & info["is_released"]
        ).float()

        reward = (
            reaching_reward
            + grasp_reward
            + transport_reward
            + release_reward
        )

        reward[info["success"]] = 5.0
        return reward

    def compute_normalized_dense_reward(
        self,
        obs: Any,
        action: torch.Tensor,
        info: dict,
    ):
        return self.compute_dense_reward(obs, action, info) / 5.0
