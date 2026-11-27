"""StackThreeCube-v1: ManiSkill's StackCube scene with a third (blue) cube.

cubeA = red, cubeB = green, cubeC = blue. Which cube should go on which is not
fixed by the scene: the executor sets `goal_pair` before each stacking step, and
`evaluate()` reports success for that pair (same rule as StackCube-v1).
"""
import sapien
import torch

from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.envs.utils import randomization
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

COLORS = {"red": "cubeA", "green": "cubeB", "blue": "cubeC"}


@register_env("StackThreeCube-v1", max_episode_steps=100000)
class StackThreeCubeEnv(StackCubeEnv):
    MIN_CENTER_DISTANCE = 0.10   # leaves room for the hand around each cube

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.cubeC = actors.build_cube(self.scene, half_size=0.02, color=[0, 0, 1, 1], name="cubeC",
                                       initial_pose=sapien.Pose(p=[2, 0, 0.1]))
        self.goal_pair = ("cubeA", "cubeB")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            sampler = randomization.UniformPlacementSampler(
                bounds=[[-0.12, -0.18], [0.12, 0.18]], batch_size=b, device=self.device)
            radius = self.MIN_CENTER_DISTANCE / 2
            for cube in (self.cubeA, self.cubeB, self.cubeC):
                xyz = torch.zeros((b, 3))
                xyz[:, :2] = sampler.sample(radius, 100, verbose=False)
                xyz[:, 2] = 0.02
                q = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False)
                cube.set_pose(Pose.create_from_pq(p=xyz, q=q))

    def cube(self, color):
        return getattr(self, COLORS[color])

    def evaluate(self):
        top, bottom = getattr(self, self.goal_pair[0]), getattr(self, self.goal_pair[1])
        offset = top.pose.p - bottom.pose.p
        xy_flag = torch.linalg.norm(offset[..., :2], axis=1) <= torch.linalg.norm(self.cube_half_size[:2]) + 0.005
        z_flag = torch.abs(offset[..., 2] - self.cube_half_size[..., 2] * 2) <= 0.005
        on = torch.logical_and(xy_flag, z_flag)
        static = top.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        grasped = self.agent.is_grasping(top)
        return {"is_on": on, "is_static": static, "is_grasped": grasped,
                "success": (on * static * (~grasped)).bool()}

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(red_cube_pose=self.cubeA.pose.raw_pose, green_cube_pose=self.cubeB.pose.raw_pose,
                       blue_cube_pose=self.cubeC.pose.raw_pose)
        return obs

    def compute_dense_reward(self, obs, action, info):   # not used: the executor computes its own reward
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info)
