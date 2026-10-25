# Isolated three-color transfer experiment

The working two-color environment, planner, model, replay buffer and videos remain
at their original paths. A source snapshot and copied checkpoint are also saved
under `runs/three_color_trial/baseline`. The experiment checks their hashes at the end.

Opt-in environment IDs are `MyPickPlaceThree-v1` and `MyCarryThree-v1`. Import
`envs.three_carry` to register them; the ordinary two-color imports/defaults do not
change. The new scene contains red, blue and yellow cubes with shuffled, separated
positions within the existing workspace. Wrong-grasp checks inspect both other cubes.

The original SAC weights are frozen. The nearest non-target actor is mapped into
the original seven-value `other_pose` slot, preserving the 52-value observation.
Both other actors physically collide and are audited, but the network does not
separately observe the second one. This is an adapter transfer test, not a policy
trained to perceive arbitrary numbers of obstacles.

The three-color planner reuses the existing Llama extraction/compilation and its
one-repair limit, with a separate three-object validation boundary. Yellow remains
unsupported in the preserved two-color scene because it is not present there.

```powershell
.\run.cmd run_three_instruction --instruction "Place the yellow cube on the green goal." --seed 95000 --output runs/my_yellow_instruction
```

Use `--plan-only` to inspect a plan without grasping. Scene preparation refreshes
one physical substep to update CPU contacts, without sending a control action.
Grasping starts only after the accepted plan is validated in the same scene.

`runs/three_color_trial/REPORT.md` separates 30 physical-only trials, 12 real-Llama
instruction/rejection checks, three captioned videos and six original two-color
controls. Initial setup diagnostics are retained separately and are excluded from
the final fixed-protocol evaluation. No unsuccessful case is silently retried.

For broader transfer, test many more unseen seeds and closer/occluded layouts.
An all-object observation encoder with mixed two/three-object training is the next
step if the nearest-object adapter is insufficient.
