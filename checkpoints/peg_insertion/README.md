# Peg insertion checkpoint

`best.pt` is the published GPU-parallel PPO policy for ManiSkill
`PegInsertionSide-v1` (state observations, obs dim 43, action dim 8). It is the
default checkpoint for `scripts/train_peg_insertion.py --evaluate`.

- SHA256: `8bafb9ab3287b05824e864c88b8df9c11b3fe41d7b749249ef928817bc25656e`
- Source run: `ppo_2048env_seed1_20261120` (2048 envs, seed 1).
- Selected checkpoint: iteration 660, 135,168,000 environment steps.
- Evaluation: 90.6% deterministic success over 128 episodes. The stored
  `best_success` (1.0) is the 16-env in-training evaluation that selected it.
- Architecture: actor and critic MLPs with two hidden layers of 256 units.
- Includes optimizer state, so `--checkpoint checkpoints/peg_insertion/best.pt`
  can also resume training.

Training runs save their checkpoints and metrics under
`runs/peg_insertion/<run_name>/` by default.
