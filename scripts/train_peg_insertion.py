"""Train a Panda peg-insertion policy (ManiSkill PegInsertionSide-v1) with GPU-parallel PPO.

Checkpoints go to checkpoints/peg_insertion/<run_name>/ (latest.pt, best.pt, ckpt_<step>.pt).

    uv run python scripts/train_peg_insertion.py                         # full run
    uv run python scripts/train_peg_insertion.py --num-envs 256 --total-timesteps 2000000   # quick try
    uv run python scripts/train_peg_insertion.py --checkpoint checkpoints/peg_insertion/<run>/latest.pt   # resume
    uv run python scripts/train_peg_insertion.py --evaluate --checkpoint checkpoints/peg_insertion/<run>/best.pt --capture-video

Every hyper-parameter in robot_skills.training.ppo.PPOConfig is a flag (see --help).
Defaults follow ManiSkill's published PPO baseline settings for this task: peg insertion
is a long-horizon, contact-rich task and typically needs on the order of 10^8 env steps
from state observations before success rates climb.
"""

from robot_skills.training.ppo import PPOConfig, main

PEG_INSERTION_DEFAULTS = PPOConfig(
    skill="peg_insertion",
    num_envs=2048,
    num_steps=100,
    num_eval_steps=100,
    update_epochs=8,
    num_minibatches=32,
    total_timesteps=250_000_000,
    gamma=0.97,
    gae_lambda=0.95,
    eval_freq=20,
)

if __name__ == "__main__":
    main(defaults=PEG_INSERTION_DEFAULTS)
