"""Opt-in red/blue/yellow scene; MyPickPlace-v1 is not changed or re-registered."""
import torch
import sapien
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from envs.pick_place import MyPickPlaceEnv

COLORS = ('red','blue','yellow')


@register_env('MyPickPlaceThree-v1',max_episode_steps=200)
class ThreePickPlaceEnv(MyPickPlaceEnv):
    def __init__(self,*args,target_color='red',**kwargs):
        if target_color not in COLORS:
            raise ValueError('Unknown three-color target')
        super().__init__(*args,target_color='red',**kwargs)
        self.target_color=target_color
        self.cube=self.cubes[target_color]

    def reset(self,*,seed=None,options=None):
        _,info=super().reset(seed=seed,options=options)
        # CPU contact queries still describe the previous physical frame after
        # teleport-style scene initialization. Refresh contacts for EVERY reset,
        # not only failures. No robot action is issued and no grasp flag is masked.
        self.last_reset_contact_refresh={
            'before':{c:bool(self.agent.is_grasping(a).item()) for c,a in self.cubes.items()},
            'physics_steps':1,'control_actions':0}
        self.scene.step()
        info.update(self.get_info())
        self.last_reset_contact_refresh['after']={c:bool(self.agent.is_grasping(a).item()) for c,a in self.cubes.items()}
        observation=self.get_obs(info)
        self._last_obs=observation
        return observation,info

    def _load_scene(self,options):
        super()._load_scene(options)
        self.yellow_cube=actors.build_cube(self.scene,half_size=self.cube_half_size,
            color=[.95,.8,.08,1.],name='cube_yellow',
            initial_pose=sapien.Pose(p=[0,0,self.cube_half_size]))
        self.cubes['yellow']=self.yellow_cube

    def _initialize_episode(self,env_idx,options):
        super()._initialize_episode(env_idx,options)
        b=len(env_idx)
        with torch.device(self.device):
            # Three shuffled Y bands, all in the original initial workspace.
            # Minimum Y separation .08m exceeds the .05m cube width.
            order=torch.rand((b,3),device=self.device).argsort(dim=1)
            centers=torch.tensor([-.12,0,.12],device=self.device)
            for index,color in enumerate(COLORS):
                position=torch.zeros((b,3),device=self.device)
                position[:,0]=torch.rand(b,device=self.device)*.15-.10
                position[:,1]=centers[order[:,index]]+torch.rand(b,device=self.device)*.04-.02
                position[:,2]=self.cube_half_size
                self.cubes[color].set_pose(Pose.create_from_pq(p=position,q=[1,0,0,0]))

    def evaluate(self):
        result=super().evaluate()
        wrong=torch.zeros_like(result['is_grasped'],dtype=torch.bool)
        for color,cube in self.cubes.items():
            if color!=self.target_color:
                wrong |= self.agent.is_grasping(cube)
        result['is_wrong_grasped']=wrong
        result['success'] &= ~wrong
        return result

    def select_target(self,object_id):
        color=object_id.removeprefix('cube_')
        if object_id!='cube_'+color or color not in self.cubes:
            raise ValueError('Unknown three-color object')
        self.target_color=color
        self.cube=self.cubes[color]
        return self.get_obs()

    def _get_obs_extra(self,info):
        result=super()._get_obs_extra(info)
        if 'state' in self.obs_mode:
            result['yellow_cube_pose']=self.yellow_cube.pose.raw_pose
            result['target_color']=torch.tensor([int(self.target_color==c) for c in COLORS],
                device=self.device,dtype=torch.float32).expand(self.num_envs,-1)
        return result
