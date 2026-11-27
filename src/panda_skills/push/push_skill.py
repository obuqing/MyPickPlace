"""Single-environment pushing skill: scripted approach, learned XY, scripted retreat.

The controller holds Z/orientation and closes the fingers. It never grasps/lifts
the cube intentionally. Physics is stepped throughout; no object teleportation
occurs after reset. RL actions are not mixed with a scripted forward policy.
"""
import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation
from .push_distance import vector  # also registers MyPushDistance-v1

CONTRACT = dict(version='push_xy_v2', direction='+X', distances_m=[.05,.10,.15],
                action_dimension=2, action_scale_m=.003,
                max_push_steps=300, tolerance_m=.01, lateral_tolerance_m=.01,
                control='scripted approach and retreat; SAC XY; fixed Z and orientation')


def outcome(start, goal, position, velocity, tilt, invalid_seen=False):
    delta = position-start
    error = goal-position
    valid = (not invalid_seen and abs(float(delta[2])) <= .01 and tilt < .35)
    success = (valid and abs(float(error[0])) <= .01 and abs(float(error[1])) <= .01
               and np.linalg.norm(velocity) < .02)
    return dict(success=bool(success), x_error_m=abs(float(error[0])),
                lateral_error_m=abs(float(error[1])), displacement_m=delta.tolist(),
                invalid_motion=not valid)


class PushSkill(gym.Env):
    metadata = {'render_modes':['rgb_array'], 'render_fps':20}

    def __init__(self, distance_m=None, render_mode=None, audit=None):
        super().__init__()
        if distance_m is not None and (not np.isfinite(distance_m) or not .03 <= distance_m <= .15):
            raise ValueError('Distance must be between .03 and .15 m')
        self.fixed_distance = distance_m
        self.render_mode = render_mode
        self.audit = audit
        self.action_space = gym.spaces.Box(-1,1,(2,),dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf,np.inf,(26,),dtype=np.float32)
        self.base = gym.make('MyPushDistance-v1', distance_m=distance_m or .05,
                             obs_mode='state_dict', control_mode='pd_ee_delta_pose',
                             render_mode=render_mode, max_episode_steps=1500)
        self.frames = []
        self.scene_seeds = []

    @property
    def raw(self):
        return self.base.unwrapped

    def _position(self):
        return vector(self.raw.obj.pose.p)

    def _tilt(self):
        quat = vector(self.raw.obj.pose.raw_pose)[3:]
        return float(np.arccos(np.clip(Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()[2,2],-1,1)))

    def _state(self):
        obj = self._position()
        return np.concatenate([vector(self.raw.agent.robot.get_qpos()),
                               vector(self.raw.agent.robot.get_qvel())*.1,
                               (vector(self.raw.agent.tcp.pose.p)-obj)/.1,
                               (self.goal[:2]-obj[:2])/.15,
                               vector(self.raw.obj.linear_velocity)]).astype(np.float32)

    def _step_target(self, target):
        tcp_pose = vector(self.raw.agent.tcp.pose.raw_pose)
        action = np.zeros(7,dtype=np.float32)
        action[:3] = np.clip((target-tcp_pose[:3])*5,-.15,.15)
        rotation = Rotation.from_quat(tcp_pose[[4,5,6,3]])
        action[3:6] = np.clip(-(Rotation.from_quat([1,0,0,0])*rotation.inv()).as_rotvec()*5,-.3,.3)
        action[6] = -1
        self.base.step(action)
        self.physical_steps += 1
        if self.render_mode == 'rgb_array':
            self.frames.append(self.base.render()[0].cpu().numpy().astype(np.uint8))
        obj = self._position()
        self.invalid_seen |= abs(float(obj[2]-self.start[2])) > .01 or self._tilt() > .35

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_seed = int(seed) if seed is not None else int(self.np_random.integers(0,100000))
        self.scene_seeds.append(self.episode_seed)
        self.distance = self.fixed_distance or float(self.np_random.choice(CONTRACT['distances_m']))
        self.raw.distance_m = self.distance
        self.base.reset(seed=self.episode_seed)
        self.start = self._position()
        self.goal = self.start + [self.distance,0,0]
        self.invalid_seen = False
        self.physical_steps = self.steps = 0
        self.frames = []
        self.finished = False
        for offset in ([-.06,0,.14],[-.06,0,.005],[-.035,0,.005]):
            target = self.start+offset
            for _ in range(120):
                self._step_target(target)
                if np.linalg.norm(vector(self.raw.agent.tcp.pose.p)-target) < .006:
                    break
            else:
                raise RuntimeError(f'Approach failed at seed {self.episode_seed}; no retry')
        if self.invalid_seen or np.linalg.norm(self._position()-self.start) > .005:
            raise RuntimeError(f'Approach disturbed cube at seed {self.episode_seed}; no retry')
        self.previous_error = float(np.linalg.norm((self.goal-self._position())[:2]))
        self.previous_reach = float(np.linalg.norm(vector(self.raw.agent.tcp.pose.p)-(self._position()+[-.025,0,.005])))
        return self._state(), {'seed':self.episode_seed,'distance_m':self.distance}

    def step(self, action):
        if self.finished:
            raise RuntimeError('Reset required after episode completion')
        action = np.asarray(action,dtype=np.float32)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError('Expected two finite XY actions')
        action = np.clip(action,-1,1)
        target = vector(self.raw.agent.tcp.pose.p)
        # servo gain is 0.5 in position units, so 6 mm target offset produces
        # approximately 3 mm commanded end-effector displacement per action.
        target[:2] += .006*action
        target[2] = self.start[2]+.005
        self._step_target(target)
        self.steps += 1
        obj = self._position()
        error = float(np.linalg.norm((self.goal-obj)[:2]))
        behind = obj + [-.025,0,.005]
        reach_error = float(np.linalg.norm(vector(self.raw.agent.tcp.pose.p)-behind))
        reward = 250*(self.previous_error-error) + 25*(self.previous_reach-reach_error) - .03 - .005*float(action@action)
        self.previous_error = error
        self.previous_reach = reach_error
        failed = self.invalid_seen or abs(float(obj[1]-self.start[1])) > .04 or obj[0] > self.goal[0]+.02
        arrived = abs(float(self.goal[0]-obj[0])) < .004 and abs(float(self.goal[1]-obj[1])) < .008
        terminated = bool(failed or arrived)
        truncated = bool(self.steps >= CONTRACT['max_push_steps'] and not terminated)
        reason = 'invalid_motion' if failed else 'arrived' if arrived else 'timeout' if truncated else 'running'
        success = False
        if terminated or truncated:
            # Same physical retreat for baseline and SAC, including timeouts.
            # It cannot push forward to correct a failed learned trajectory.
            target = vector(self.raw.agent.tcp.pose.p)+[-.04,0,.08]
            stable = 0
            for _ in range(45):
                self._step_target(target)
                metrics = self.metrics()
                stable = stable+1 if metrics['success'] else 0
            success = bool(stable >= 10 and not failed)
            reward += 10 if success else -5
            self.finished = True
        info = self.metrics()
        info.update(success=success, reason='placed' if success else reason,
                    seed=self.episode_seed, distance_m=self.distance, push_steps=self.steps,
                    physical_steps=self.physical_steps)
        return self._state(), float(reward), terminated, truncated, info

    def metrics(self):
        return outcome(self.start,self.goal,self._position(),vector(self.raw.obj.linear_velocity),
                       self._tilt(),self.invalid_seen)

    def scripted_action(self):
        tcp = vector(self.raw.agent.tcp.pose.p)
        return np.array([.67,np.clip((self.start[1]-tcp[1])/.006,-1,1)],dtype=np.float32)

    def close(self):
        self.base.close()
