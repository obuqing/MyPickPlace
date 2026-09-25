import gymnasium as gym
import mani_skill.envs

import envs  # registers MyPickPlace-v1


env = gym.make(
    "MyPickPlace-v1",
    robot_uids="panda",
    num_envs=1,
    obs_mode="state_dict",
    control_mode="pd_ee_delta_pose",
    render_mode="human",
)

obs, info = env.reset(seed=0)

print("=== ACTION SPACE ===")
print(env.action_space)

print("\n=== OBSERVATION SPACE ===")
print(env.observation_space)

print("\n=== OBSERVATION KEYS ===")
print(obs.keys())

print("\n=== INFO ===")
print(info)

env.render()
input("\nPress Enter to close...")
env.close()
