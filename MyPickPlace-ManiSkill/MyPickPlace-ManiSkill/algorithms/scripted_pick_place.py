"""Observation-only, single-environment pick-and-place state machine."""

import numpy as np

from algorithms.policy import Policy


def array(tensor):
    return tensor.detach().cpu().numpy()[0]


class ScriptedPickPlacePolicy(Policy):
    phases = ("approach", "descend", "grasp", "lift", "move", "lower", "release", "retreat", "settle")

    def reset(self):
        self.phase = "approach"
        self.phase_steps = 0
        self.done = False
        self.failure = None
        self.grasp_offset = None
        self.drop_steps = 0

    def advance(self):
        self.phase = self.phases[self.phases.index(self.phase) + 1]
        self.phase_steps = 0

    def act(self, observation):
        from scipy.spatial.transform import Rotation

        obs = observation["extra"]
        tcp_pose = array(obs["tcp_pose"])
        tcp, obj, goal = tcp_pose[:3], array(obs["obj_pose"])[:3], array(obs["goal_pos"])
        grasped = bool(obs["is_grasped"].item())
        self.phase_steps += 1
        if self.phase_steps > 90:
            self.failure = "timeout_" + self.phase
        if self.phase in ("lift", "move", "lower"):
            self.drop_steps = 0 if grasped else self.drop_steps + 1
            if self.drop_steps >= 3:
                self.failure = "dropped_" + self.phase

        target = tcp.copy()
        gripper = 1.0
        if self.phase == "approach":
            target = obj + [0, 0, 0.16]
            if np.linalg.norm(target - tcp) < 0.008:
                self.advance()
        elif self.phase == "descend":
            target = obj + [0, 0, 0.002]
            if np.linalg.norm(target - tcp) < 0.005:
                self.advance()
        elif self.phase == "grasp":
            target = obj + [0, 0, 0.002]
            gripper = -1.0
            if self.phase_steps >= 12 and grasped:
                self.grasp_offset = tcp - obj
                self.advance()
            elif self.phase_steps >= 25:
                self.failure = "grasp_failed"
        elif self.phase == "lift":
            gripper = -1.0
            target = np.array([obj[0], obj[1], 0.18]) + self.grasp_offset
            if obj[2] >= 0.17:
                self.advance()
        elif self.phase == "move":
            gripper = -1.0
            target = np.array([goal[0], goal[1], 0.18]) + self.grasp_offset
            if np.linalg.norm(obj[:2] - goal[:2]) < 0.008:
                self.advance()
        elif self.phase == "lower":
            gripper = -1.0
            target = np.array([goal[0], goal[1], 0.028]) + self.grasp_offset
            if np.linalg.norm(target - tcp) < 0.005:
                self.release_tcp = target.copy()
                self.advance()
        elif self.phase == "release":
            target = self.release_tcp
            if self.phase_steps >= 15:
                self.advance()
        elif self.phase == "retreat":
            target = self.release_tcp + [0, 0, 0.15]
            if np.linalg.norm(target - tcp) < 0.008:
                self.rest_tcp = target.copy()
                self.advance()
        elif self.phase == "settle":
            target = self.rest_tcp
            self.done = self.phase_steps >= 15

        # Panda uses +/- 0.1 m translation; its normalized rotation scale is -0.1.
        current_rotation = Rotation.from_quat(tcp_pose[[4, 5, 6, 3]])
        rotation_error = (Rotation.from_quat([1, 0, 0, 0]) * current_rotation.inv()).as_rotvec()
        action = np.zeros(7, dtype=np.float32)
        action[:3] = np.clip((target - tcp) * 5, -0.15, 0.15)
        action[3:6] = np.clip(-rotation_error * 5, -0.3, 0.3)
        action[6] = gripper
        return action
