"""Train a Panda peg-insertion policy (ManiSkill PegInsertionSide-v1) with GPU-parallel PPO.

Training writes to runs/peg_insertion/<run_name>/ (latest.pt, best.pt, ckpt_<step>.pt).

    uv run python scripts/train_peg_insertion.py                         # full run
    uv run python scripts/train_peg_insertion.py --num-envs 256 --total-timesteps 2000000   # quick try
    uv run python scripts/train_peg_insertion.py --checkpoint runs/peg_insertion/<run>/latest.pt   # resume
    uv run python scripts/train_peg_insertion.py --evaluate --checkpoint runs/peg_insertion/<run>/best.pt --capture-video

Every hyper-parameter in panda_skills.training.ppo.PPOConfig is a flag (see --help).
Defaults follow ManiSkill's published PPO baseline settings for this task: peg insertion
is a long-horizon, contact-rich task and typically needs on the order of 10^8 env steps
from state observations before success rates climb.
"""

from panda_skills.paths import PEG_INSERTION_CHECKPOINT, RUNS_DIR
from panda_skills.training.ppo import PPOConfig, build_parser, train

PEG_INSERTION_DEFAULTS = PPOConfig(
    skill="peg_insertion",
    output_dir=RUNS_DIR,
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

def main(argv=None):
    parser = build_parser(PEG_INSERTION_DEFAULTS)
    args = parser.parse_args(argv)
    if args.evaluate and args.checkpoint is None and args.skill == "peg_insertion":
        if not PEG_INSERTION_CHECKPOINT.is_file():
            parser.error(f"No published checkpoint at {PEG_INSERTION_CHECKPOINT}; pass --checkpoint")
        args.checkpoint = PEG_INSERTION_CHECKPOINT
    train(PPOConfig(**vars(args)))


if __name__ == "__main__":
    main()
