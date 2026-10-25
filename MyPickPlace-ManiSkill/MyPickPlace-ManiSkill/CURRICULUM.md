# Carry distance curriculum

Run `run.cmd train_carry_curriculum` from the workspace root. Default output:
`runs/carry_curriculum_20000` (must not already exist).

Starts a fresh SAC model for a controlled comparison with the previous seed-0 run.
Reward, action scale, 18cm height and 2cm success tolerance stay unchanged.
Targets are fixed per episode after the physical scripted lift; no object teleporting.
5cm, 10cm and 15cm displacement caps preserve the target height. The last stage
uses the original goal. Curriculum targets are numerical waypoints, not moved physical bins.

Every 2000 steps, evaluate ten deterministic red/blue episodes on seeds 4000–4004.
Advance only after two consecutive >=80% passes. Otherwise stay at the same level;
20000 steps is a budget cap, not a guarantee of reaching the full-distance stage.
Replay and optimizer persist; promotion resets the rollout so goals never change mid-episode.
Evaluation runs in separate processes. Every 4000 steps, evaluate the original full
task on seeds 2000–2004 and choose the best checkpoint using these results only.
Finally test best and last models on seeds 3000–3009 (20 episodes each).
Short waypoint arrival cannot trigger scripted placement or count as full task success.

`status.json` and `training/progress.json` track progress. After completion,
`REPORT.md`, `curriculum_history.json`, `height_diagnostics.json/.png` and
`training/validation_xyz_trajectories.png` provide results and next steps.
Signed error is target minus object: negative Z indicates excessive height.
Height diagnostics include final signed/absolute error, peak upward overshoot,
and mean vertical action. Full XYZ/action traces are saved for each evaluation.

Continue a completed run with a new output directory:

```powershell
.\run.cmd train_carry_curriculum --resume runs/carry_curriculum_40000 --steps 20000 --seed 200000 --output runs/carry_curriculum_60000
```

`--steps` is the additional budget when resuming. Model/optimizer, replay buffer,
step/update counters and the last validation pass streak are restored. A new physical
episode starts with the specified rollout seed; simulation and RNG state are not
an exact continuation. Initial full-task validation does not count as another
curriculum pass. `resume.json` records checks and source artifact hashes.

Frozen-policy generalization check and exactly two full execution videos:

```powershell
.\run.cmd validate_carry_generalization --model runs/carry_curriculum_40000/training/best_model.zip --episodes 100 --seed 50000 --output runs/carry_generalization_200
```

Audits actual previous scene records and rejects overlapping seeds. Tests each
scene once for each color without retries. Saves failures, arrival XYZ traces,
per-color metrics and a report. `videos/red.mp4` and `videos/blue.mp4` record the
first preselected scene, including scripted initialization and release around
the learned carry; every recorded control frame is checked by decoding the MP4.
The two video executions are separate repetitions and are not added to the
200-episode metrics. Same-distribution scene diversity is distinct from new
training seeds or changed physical parameters.
