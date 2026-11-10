# Stacking skill (ManiSkill `StackCube-v1`)

Goal: put the red cube (cubeA) on top of the green cube (cubeB).
Success is ManiSkill's own check: red on green, red not moving, gripper released.

Hybrid design (same idea as the carry and push skills):

| Phase | Who controls it |
|---|---|
| Move above red cube, descend, close gripper, lift 10 cm | Script |
| Carry the red cube over the green cube and lower it | **SAC (3 actions: dx, dy, dz)** |
| Open gripper, move up, let it settle | Script |

Observation (8 numbers): red cube to stack target (3), gripper offset from the cube (3),
"still grasping" flag (1), fraction of the time budget used (1).
Reward: progress toward "over the green cube, then down", small time/effort cost,
+20 if ManiSkill reports success at the end, -5 otherwise.
Training starts with 30 noisy scripted demonstrations in SAC's replay buffer.

## Commands (run from the repository root)

```powershell
.\run-skills.cmd explore_stack                                   # step 1: inspect the task
.\run-skills.cmd run_trained_stack --scripted --episodes 5 --output runs/stack_scripted   # step 3: baseline
.\run-skills.cmd train_stack --steps 20000 --output runs/stack_sac_20000                  # step 5: train
.\run-skills.cmd run_trained_stack --model runs/stack_sac_20000/best_model.zip --episodes 5 --output runs/stack_demo  # step 6
```

Every output folder must be new. Training writes `config.json`, `training_steps.csv`,
`validation_history.json`, `best_model.zip`, `last_model.zip`, `scripted_test.json`,
`sac_test.json`, `videos/` and `REPORT.md`.

Reference run (CPU physics, 20,000 steps, about 15 min): best checkpoint at step 12,000;
test seeds 30000-30019: scripted 19/20, SAC 18/20. Validation success drops late in
training, which is why the best checkpoint (chosen on separate validation seeds) is kept.
