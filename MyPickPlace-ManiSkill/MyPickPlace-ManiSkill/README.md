# MyPickPlace: Llama 3 + frozen SAC

The Panda robot moves one red or blue cube onto a green tabletop goal. Local Llama 3
extracts an English task; source evidence and scene validation compile a structured
plan. Scripted control grasps/lifts, frozen SAC controls XYZ transport, and scripted
control lowers/releases. Rejected instructions make no control steps.

## Run

Run commands from the outer workspace root. The existing `.venv` and portable
Ollama installation are used by `run.cmd`.

```powershell
.\run.cmd start_llama
.\run.cmd run_instruction --executor rl --instruction "Place the blue cube on the green goal." --seed 81000 --output runs/my_rl_instruction
```

The RL checkpoint is `runs/carry_curriculum_40000/training/best_model.zip`.
`--plan-only` skips execution. Explicit `--plan-file` replay is labeled offline.
Only the validated two-stage planner is supported; it allows at most one evidence
repair and no execution retry. See [LLM_RL.md](LLM_RL.md) for role corrections.

## Evaluate and train

```powershell
.\run.cmd evaluate_llm_rl --seed 80000 --output runs/my_llm_rl_evaluation
.\run.cmd evaluate_carry --policy sac --model runs/carry_curriculum_40000/training/best_model.zip --episodes 10 --seed 82000 --output runs/my_carry_evaluation
.\run.cmd check_planner_regressions --output runs/my_planner_regressions
.\run.cmd evaluate_colors --output runs/my_scripted_evaluation
.\run.cmd check_carry --output runs/my_interface_check
```

Use new output directories. Source-only planning cases live in
`configs/planner_regression_cases.json`, so evaluation needs no deleted experiment.
Prior scene seeds and instruction hashes are retained in `runs/scene_history.json`
for novelty checks; rejected/unsupported instructions remain regression fixtures.

The working curriculum trainer and replay buffer are retained for further training:

```powershell
.\run.cmd train_carry_curriculum --resume runs/carry_curriculum_40000 --steps 20000 --seed 200000 --output runs/carry_curriculum_60000
```

`--steps` is the additional budget. See [CURRICULUM.md](CURRICULUM.md) for promotion
criteria and diagnostics. This command starts new training only when you run it.

## Saved results

- `runs/carry_curriculum_40000`: successful trained model, replay buffer and report.
- `runs/carry_generalization_200`: 100 unseen scene seeds per color, 200/200 placements.
- `runs/correction_roles_verified/REPORT.md`: 52 integration outcomes and final
  compiler replay of 100 real-model planning outputs. Raw unsuccessful intermediate
  runs and superseded source snapshots have been removed.
- `runs/llm_rl_generalization_videos/red.mp4` and `blue.mp4`: latest captioned real
  Llama-to-RL videos, with new instruction wordings and scene seeds.

Older successful checkpoints/videos are preserved. Failed checkpoints, experiment
outputs, retired direct-planning prompts and one-off comparison scripts were removed.
Cleanup inventory and preserved-file hashes: `runs/cleanup_failed_artifacts.json`.

## Checks and interface

```powershell
.\.venv\python.exe -m unittest discover -s MyPickPlace-ManiSkill/MyPickPlace-ManiSkill/tests -q
.\run.cmd test_env --steps 20
```

`MyPickPlace-v1` has two cubes with randomized poses and a green goal. `MyCarrySkill-v1`
exposes 52 numerical observation values and three normalized XYZ actions, a closed
gripper and downward orientation. Carry arrival requires grasp plus 3D error <=2cm
at an 18cm target height. Full placement also requires release and settling.
The numerical state, controller scale, reward and current trained policy are retained.
