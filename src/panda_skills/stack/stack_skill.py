"""Stacking skill on ManiSkill's StackCube-v1: scripted grasp, learned XYZ, scripted release.

Same hybrid design as the team's carry and push skills:
  1. Scripted:  move above the red cube (cubeA), descend, close gripper, lift.
  2. Learned:   SAC outputs 3 numbers (dx, dy, dz) to carry the red cube until it
                sits just above the green cube (cubeB). Orientation stays fixed.
  3. Scripted:  open the gripper, retreat upward, let the cube settle.
Success is ManiSkill's own check: red cube on green cube, static, not grasped.
"""
import gymnasium as gym
import mani_skill.envs  # noqa: F401  (registers StackCube-v1)
import numpy as np
from scipy.spatial.transform import Rotation

CONTRACT = dict(
    version="stack_xyz_v3",
    base_env="StackCube-v1",
    action_dimension=3,
    action_scale_m=0.02,        # servo target offset per action (about 1 cm of motion)
    lift_height_m=0.10,         # cubeA bottom lifted to ~10 cm before RL starts
    max_rl_steps=100,
    release_xy_tolerance_m=0.01,
    release_z_tolerance_m=0.012,
    control="scripted grasp/lift and release/retreat; SAC XYZ; fixed downward orientation",
)


def vec(x):
    """First environment of a batched torch tensor -> numpy array."""
    return x.cpu().numpy()[0].astype(np.float64)


class StackSkill(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.action_space = gym.spaces.Box(-1, 1, (3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32)
        self.base = gym.make(
            CONTRACT["base_env"], obs_mode="state_dict", control_mode="pd_ee_delta_pose",
            reward_mode="dense", render_mode=render_mode, max_episode_steps=100000,
            render_backend="cpu",  # the default CUDA image path crashes on Windows
        )
        self.frames = []
        self.finished = True

    @property
    def raw(self):
        return self.base.unwrapped

    # ---------- helpers ----------
    def _tcp(self):
        return vec(self.raw.agent.tcp.pose.p)

    def _a(self):
        return vec(self.raw.cubeA.pose.p)

    def _b(self):
        return vec(self.raw.cubeB.pose.p)

    def _stack_target(self):
        """Where cubeA's center should be: directly on top of cubeB."""
        return self._b() + np.array([0, 0, 2 * self.half])

    def _grasped(self):
        return bool(vec(self.raw.agent.is_grasping(self.raw.cubeA)))

    def _servo(self, target, gripper, yaw=None):
        """One physics step moving the gripper toward `target` with fixed downward orientation."""
        pose = vec(self.raw.agent.tcp.pose.raw_pose)        # [x,y,z, qw,qx,qy,qz]
        action = np.zeros(7, dtype=np.float32)
        action[:3] = np.clip((target - pose[:3]) * 5, -0.15, 0.15)
        current = Rotation.from_quat(pose[[4, 5, 6, 3]])
        desired = Rotation.from_euler("xyz", [np.pi, 0, self.yaw if yaw is None else yaw])
        # Measured on this controller: a positive rotation action turns the gripper
        # the NEGATIVE way about the world axes, hence the minus sign (the push skill
        # uses the same convention). One action unit is about 0.1 rad.
        action[3:6] = np.clip(-(desired * current.inv()).as_rotvec() * 5, -1, 1)
        action[6] = gripper
        self.base.step(action)
        self.physical_steps += 1
        if self.render_mode == "rgb_array":
            self.frames.append(vec(self.base.render()).astype(np.uint8))

    def _go(self, target, gripper, steps=150, tol=0.005):
        for _ in range(steps):
            self._servo(target, gripper)
            if np.linalg.norm(self._tcp() - target) < tol:
                return True
        return False

    def _hold(self, gripper, steps):
        target = self._tcp()
        for _ in range(steps):
            self._servo(target, gripper)

    @staticmethod
    def _yaw_of(entity):
        q = vec(entity.pose.raw_pose)[3:]                      # wxyz
        return Rotation.from_quat(q[[1, 2, 3, 0]]).as_euler("xyz")[2]

    @staticmethod
    def _wrap90(angle):
        """Wrap an angle to [-45, 45) degrees (cube symmetry)."""
        return (angle + np.pi / 4) % (np.pi / 2) - np.pi / 4

    def _state(self):
        # Keep the observation small: only what matters for carrying and aligning.
        tcp, a, goal = self._tcp(), self._a(), self._stack_target()
        return np.concatenate([
            (goal - a) / 0.1,                       # where the red cube still has to go (3)
            (tcp - a) / 0.05,                       # cube's offset in the hand (3)
            [float(self._grasped())],               # still holding it? (1)
            [self.steps / CONTRACT["max_rl_steps"]],  # time used so far (1)
        ]).astype(np.float32)

    # ---------- gym API ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_seed = int(seed) if seed is not None else int(self.np_random.integers(0, 10**6))
        self.base.reset(seed=self.episode_seed)
        self.half = float(self.raw.cube_half_size[2])  # 1-D tensor, not batched
        self.physical_steps = self.steps = 0
        self.frames = []
        self.finished = False

        # Align the fingers with the red cube's faces (cubes are randomly rotated).
        # A cube looks the same every 90 degrees, so pick the equivalent yaw closest
        # to the gripper's current yaw to avoid large wrist rotations.
        cube_yaw = self._yaw_of(self.raw.cubeA)
        tcp_yaw = self._yaw_of(self.raw.agent.tcp)
        self.yaw = tcp_yaw + self._wrap90(cube_yaw - tcp_yaw)

        a = self._a()
        # 1) above cubeA, 2) down to grasp height, 3) close, 4) lift
        self._go(a + [0, 0, 0.08], gripper=1)
        self._go(a + [0, 0, 0.0], gripper=1, tol=0.003)
        self._hold(gripper=-1, steps=20)
        self._go(a + [0, 0, CONTRACT["lift_height_m"]], gripper=-1)
        if not self._grasped() or self._a()[2] < a[2] + 0.5 * CONTRACT["lift_height_m"]:
            raise RuntimeError(f"Scripted grasp failed at seed {self.episode_seed}; no retry")

        self.start_a = self._a()
        self.b_start = self._b()
        self.prev_dist = self._shaped_distance(self._a(), self._stack_target())
        return self._state(), {"seed": self.episode_seed}

    @staticmethod
    def _shaped_distance(a, goal):
        """Distance used for the reward: horizontal error + vertical error to a SAFE height.

        The safe height is up to 3 cm above the stacking height while the red cube is
        not yet over the green cube, so the policy is not rewarded for descending early
        (which leads to bumping the green cube from the side).
        """
        xy = float(np.linalg.norm((goal - a)[:2]))
        safe_z = goal[2] + min(0.03, 2.0 * xy)
        return xy + abs(float(a[2] - safe_z))

    def step(self, action):
        if self.finished:
            raise RuntimeError("Reset required after episode completion")
        action = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
        if action.shape != (3,):
            raise ValueError("Expected three XYZ actions")

        target = self._tcp() + CONTRACT["action_scale_m"] * action
        self._servo(target, gripper=-1)
        self.steps += 1

        a, goal = self._a(), self._stack_target()
        error = goal - a
        xy_err = float(np.linalg.norm(error[:2]))
        z_err = float(abs(error[2]))

        # Dense shaping: reward progress (over first, then down), small time/effort penalty.
        shaped = self._shaped_distance(a, goal)
        reward = 50 * (self.prev_dist - shaped) - 0.05 - 0.01 * float(action @ action)
        self.prev_dist = shaped

        dropped = not self._grasped()
        # Hitting cubeB from the side or pushing it away counts as failure.
        b_moved = float(np.linalg.norm(self._b()[:2] - self.b_start[:2])) > 0.01
        too_low = a[2] < goal[2] - 0.01
        arrived = xy_err < CONTRACT["release_xy_tolerance_m"] and z_err < CONTRACT["release_z_tolerance_m"]
        failed = dropped or b_moved or too_low
        terminated = bool(arrived or failed)
        truncated = bool(self.steps >= CONTRACT["max_rl_steps"] and not terminated)

        success = False
        reason = "dropped" if dropped else "hit_base" if b_moved else "too_low" if too_low else \
                 "arrived" if arrived else "timeout" if truncated else "running"
        if terminated or truncated:
            # Same scripted release for every outcome (cannot correct a bad placement).
            self._hold(gripper=1, steps=15)
            self._go(self._tcp() + [0, 0, 0.06], gripper=1, steps=40)
            self._hold(gripper=1, steps=20)
            ev = self.raw.evaluate()
            success = bool(vec(ev["success"]))
            reward += 20 if success else -5
            reason = "stacked" if success else reason
            self.finished = True

        info = dict(success=success, reason=reason, seed=self.episode_seed, rl_steps=self.steps,
                    xy_error_m=xy_err, z_error_m=z_err, physical_steps=self.physical_steps)
        return self._state(), float(reward), terminated, truncated, info

    def scripted_action(self):
        """Hand-written baseline policy: move over cubeB first, then descend."""
        a, goal = self._a(), self._stack_target()
        error = goal - a
        if np.linalg.norm(error[:2]) > 0.006:
            error[2] = 0.0  # stay at carry height until directly above the green cube
        return np.clip(error / CONTRACT["action_scale_m"], -1, 1).astype(np.float32)

    def close(self):
        self.base.close()
