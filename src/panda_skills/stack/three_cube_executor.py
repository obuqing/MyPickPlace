"""Execute a stacking plan (list of {"pick": color, "onto": color}) on StackThreeCube-v1.

Each step reuses the stacking skill exactly: scripted grasp and lift, the trained SAC
policy for carry-and-place, scripted release. The SAC policy only observes positions
RELATIVE to the target, so the same policy works for any colour pair and also for the
second level of a tower, provided the cube is lifted to the same height above its
target as in training (6 cm above the stacking height).
"""
import gymnasium as gym
import numpy as np

from . import three_cubes_env  # noqa: F401  (registers StackThreeCube-v1)
from .stack_skill import CONTRACT, StackSkill, vec
from .three_cubes_env import COLORS

TRAINED_START_ABOVE_TARGET_M = 0.06   # in training the red cube starts 6 cm above its stacking height


class ThreeCubeStacker(StackSkill):
    def __init__(self, render_mode=None):
        gym.Env.__init__(self)
        self.render_mode = render_mode
        self.action_space = gym.spaces.Box(-1, 1, (3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32)
        self.base = gym.make("StackThreeCube-v1", obs_mode="state_dict", control_mode="pd_ee_delta_pose",
                             reward_mode="none", render_mode=render_mode, render_backend="cpu")
        self.frames, self.finished = [], True

    # The stacking skill reads these, so redirecting them changes which cubes it works on.
    def _a(self):
        return vec(self.src.pose.p)

    def _b(self):
        return vec(self.dst.pose.p)

    def _grasped(self):
        return bool(vec(self.raw.agent.is_grasping(self.src)))

    def positions(self):
        return {c: [round(v, 4) for v in vec(self.raw.cube(c).pose.p)] for c in COLORS}

    def reset_scene(self, seed):
        self.base.reset(seed=seed)
        self.episode_seed = seed
        self.half = float(self.raw.cube_half_size[2])
        self.physical_steps, self.frames = 0, []

    def begin(self, pick, onto):
        """Scripted grasp of `pick` and lift so that the SAC policy starts as in training."""
        self.src, self.dst = self.raw.cube(pick), self.raw.cube(onto)
        self.raw.goal_pair = (COLORS[pick], COLORS[onto])
        self.steps, self.finished = 0, False
        self.yaw = self._grasp_yaw(pick)

        a = self._a()
        lift = max(CONTRACT["lift_height_m"], self._stack_target()[2] + TRAINED_START_ABOVE_TARGET_M - a[2])
        self._go(a + [0, 0, 0.08], gripper=1)
        self._go(a + [0, 0, 0.0], gripper=1, tol=0.003)
        self._hold(gripper=-1, steps=20)
        self._go(a + [0, 0, lift], gripper=-1)
        if not self._grasped() or self._a()[2] < a[2] + 0.5 * lift:
            return None
        self.start_a, self.b_start = self._a(), self._b()
        self.prev_dist = self._shaped_distance(self._a(), self._stack_target())
        return self._state()

    def _grasp_yaw(self, pick):
        """Choose how to turn the hand before grasping `pick`.

        The Panda hand is about 20 cm wide along the direction the fingers close. If that
        direction points at a nearby cube or tower, the hand body knocks it over while
        descending (seen in testing). A cube can be grasped every 90 degrees, so pick the
        option whose finger direction is most perpendicular to the nearest other cube.
        """
        cube_yaw, tcp_yaw = self._yaw_of(self.src), self._yaw_of(self.raw.agent.tcp)
        base = tcp_yaw + self._wrap90(cube_yaw - tcp_yaw)
        a = self._a()
        others = [vec(self.raw.cube(c).pose.p) for c in COLORS if c != pick]
        nearest = min(others, key=lambda p: np.linalg.norm((p - a)[:2]))
        direction = (nearest - a)[:2] / (np.linalg.norm((nearest - a)[:2]) + 1e-9)

        def exposure(yaw):   # |cos| between finger direction and the nearest cube
            fingers = np.array([np.sin(yaw), -np.cos(yaw)])
            return abs(float(fingers @ direction))
        return min([base, base + np.pi / 2, base - np.pi / 2], key=lambda y: (round(exposure(y), 2), abs(y - base)))

    def is_on(self, top, bottom):
        offset = vec(self.raw.cube(top).pose.p) - vec(self.raw.cube(bottom).pose.p)
        return bool(np.linalg.norm(offset[:2]) <= np.sqrt(2) * self.half + 0.005
                    and abs(offset[2] - 2 * self.half) <= 0.005)

    def run_plan(self, steps, model=None):
        """model=None uses the scripted carry; otherwise the SAC policy. Stops at the first failure."""
        results = []
        for k, step in enumerate(steps):
            obs = self.begin(step["pick"], step["onto"])
            if obs is None:
                results.append(dict(step=k, **step, success=False, reason="grasp_failed"))
                break
            while True:
                action = self.scripted_action() if model is None else model.predict(obs, deterministic=True)[0]
                obs, _, terminated, truncated, info = self.step(action)
                if terminated or truncated:
                    break
            if not info["success"] and self.is_on(step["pick"], step["onto"]) and not self._grasped():
                # ManiSkill's "static" test uses velocity. With a cube resting on a cube on a
                # cube, CPU physics reports a jittering ~2 cm/s velocity even though the cube
                # does not move (measured: position constant within 0.1 mm). So for this case
                # we judge "static" by position: it must move < 2 mm over 0.5 s.
                start = self._a()
                drift = 0.0
                for _ in range(10):
                    self._hold(gripper=1, steps=1)
                    drift = max(drift, float(np.linalg.norm(self._a() - start)))
                if drift < 0.002 and self.is_on(step["pick"], step["onto"]):
                    info = dict(info, success=True, reason="stacked_position_static")
            results.append(dict(step=k, **step, success=info["success"], reason=info["reason"],
                                rl_steps=info["rl_steps"]))
            if not info["success"]:
                break
        # Final check: every relation built by the plan must still hold (nothing knocked over).
        done = len(results) == len(steps) and all(r["success"] for r in results)
        final = {f"{s['pick']}_on_{s['onto']}": self.is_on(s["pick"], s["onto"]) for s in steps}
        return dict(success=bool(done and all(final.values())), steps=results, final_relations=final,
                    positions=self.positions(), physical_steps=self.physical_steps)
