# Frozen distance-pushing SAC checkpoint

`best_model.zip` is the locally trained SAC XY policy, not an LLM and not a
pick-and-place/carry policy. Keep `config.json` adjacent to it.

- SHA256: `d7e4e15f3794d07c982a6f038783f922e9dbb732aff1411bf34a5d0de756f7c5`
- Training: 12,000 environment steps, 11,800 gradient updates, seed 1.
- Selected checkpoint: step 10,000, selected on validation seeds.
- Curriculum: 5, 10, 15 cm along world +X; fixed Z and gripper orientation.
- Scripted approach and retreat surround learned XY pushing.
- Architecture: two hidden layers of 128 units, Stable-Baselines3 SAC 2.9.0.

`summary.json` records the original development run (including its 13/15
development evaluation). `sac_evaluation.json` and `scripted_evaluation.json`
record the later fixed 60-episode comparison described in
`evaluation_protocol.json`: both succeed 58/60. The checkpoint path in that
protocol is the original training-run provenance, not an installation path.
Do not infer generalization to arbitrary objects, directions, or surfaces.

The source was moved into `skills/push` during integration; imports and artifact
paths changed. Training/control parameters and checkpoint weights did not.
Unified-entry verification is recorded in the repository-root
`integration_validation.json`. This checkpoint is small enough to track in Git;
it is already tracked, while the original repository ignore rules remain unchanged.
