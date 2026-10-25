import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import mani_skill.envs

import envs  # registers MyPickPlace-v1
from algorithms.policy import RandomPolicy


def main():
    env = gym.make(
        "MyPickPlace-v1",
        robot_uids="panda",
        num_envs=1,
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        reward_mode="dense",
        render_mode="human",
    )

    policy = RandomPolicy(env.action_space)
    obs, info = env.reset(seed=0)
    policy.reset()

    for _ in range(500):
        action = policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if bool((terminated | truncated).item()):
            print("success:", info.get("success"))
            obs, info = env.reset()
            policy.reset()

    env.close()


if __name__ == "__main__":
    main()
