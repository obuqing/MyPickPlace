import argparse

import gymnasium as gym
import mani_skill.envs  # registers built-in ManiSkill tasks

# Importing envs registers MyPickPlace-v1.
import envs  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument(
        "--render-mode",
        choices=["human", "rgb_array"],
        default="human",
    )
    args = parser.parse_args()

    env = gym.make(
        "MyPickPlace-v1",
        robot_uids="panda",
        num_envs=1,
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        reward_mode="dense",
        render_mode=args.render_mode,
    )

    print("Observation space:")
    print(env.observation_space)
    print("\nAction space:")
    print(env.action_space)

    obs, info = env.reset(seed=args.seed)

    for step in range(args.steps):
        # Smoke test only: random actions do not intentionally solve the task.
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if args.render_mode == "human":
            env.render()

        if step % 20 == 0:
            print(
                f"step={step:03d} reward={reward} "
                f"success={info.get('success')}"
            )

        # ManiSkill outputs batched tensors even for one environment.
        done = bool((terminated | truncated).item())
        if done:
            print("Episode ended. Resetting...")
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
