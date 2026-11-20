# Robot Skills: Panda manipulation on ManiSkill

A collection of Panda robot skills built on [ManiSkill 3](https://github.com/haosulab/ManiSkill)
(installed as a dependency; not vendored). Skills are registered by name in
`robot_skills.skills`, so new ones can be added without restructuring the project.

| Skill | Env id | How it's solved |
|---|---|---|
| `pick_place` / `pick_place_three` | `MyPickPlace-v1` / `MyPickPlaceThree-v1` | Llama 3 plan + scripted grasp/release |
| `carry` / `carry_three` | `MyCarrySkill-v1` / `MyCarryThree-v1` | Frozen SAC policy for XYZ transport |
| `peg_insertion` | `PegInsertionSide-v1` (ManiSkill built-in) | GPU-parallel PPO |

## Setup

Requires [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU (CUDA 12.8 torch wheels).

```bash
uv sync              # creates .venv from pyproject.toml + uv.lock
uv run pytest        # unit tests
```

Run every command from the project root.

## Layout

```
pyproject.toml, uv.lock   dependencies (managed by uv)
src/robot_skills/         installable package
  skills.py               skill registry (name -> env id + default env settings)
  envs/                   custom Gym/ManiSkill envs (importing registers them)
  policies/               scripted and base policies
  planning/               LLM instruction -> structured plan
  execution/              run a validated plan with scripted/RL policies
  training/               trainers: carry SAC helpers, generic GPU PPO (ppo.py)
  paths.py                project directories (configs/, checkpoints/, runs/)
scripts/                  command-line entry points
configs/                  planner cases and env configs
checkpoints/              trained policy checkpoints (gitignored)
runs/                     experiment outputs and reports (gitignored)
docs/                     per-skill notes
tests/
```

To add a skill: add an env under `src/robot_skills/envs/` (or reuse a ManiSkill
task), call `register_skill(...)` in `skills.py`, and train it with an existing trainer,
for example `robot_skills.training.ppo` for any GPU-vectorized ManiSkill env.

## Peg insertion (PPO)

```bash
uv run python scripts/train_peg_insertion.py                    # full run (~250M env steps)
uv run python scripts/train_peg_insertion.py --num-envs 256 --total-timesteps 2000000   # quick try
uv run python scripts/train_peg_insertion.py --checkpoint checkpoints/peg_insertion/<run>/latest.pt   # resume
uv run python scripts/train_peg_insertion.py --evaluate --checkpoint checkpoints/peg_insertion/<run>/best.pt --capture-video
uv run python scripts/train_peg_insertion.py --help             # all hyper-parameters
```

Each run writes to `checkpoints/peg_insertion/<run_name>/`: `config.json`, `metrics.jsonl`,
`latest.pt`, `best.pt` (highest eval success rate) and `ckpt_<step>.pt`.

## Pick/place + carry: Llama 3 + frozen SAC

The Panda robot moves one red or blue cube onto a green tabletop goal. Local Llama 3
extracts an English task; source evidence and scene validation compile a structured
plan. Scripted control grasps/lifts, frozen SAC controls XYZ transport, and scripted
control lowers/releases. Rejected instructions make no control steps.

### Run

`start_llama` launches the portable Ollama in `.cache/ollama` (Windows build).

```bash
uv run python scripts/start_llama.py
uv run python scripts/run_instruction.py --executor rl --instruction "Place the blue cube on the green goal." --seed 81000 --output runs/my_rl_instruction
```

The RL checkpoint is `runs/carry_curriculum_40000/training/best_model.zip`.
`--plan-only` skips execution. Explicit `--plan-file` replay is labeled offline.
Only the validated two-stage planner is supported; it allows at most one evidence
repair and no execution retry. See [docs/LLM_RL.md](docs/LLM_RL.md) for role corrections.

### Evaluate and train

```bash
uv run python scripts/evaluate_llm_rl.py --seed 80000 --output runs/my_llm_rl_evaluation
uv run python scripts/evaluate_carry.py --policy sac --model runs/carry_curriculum_40000/training/best_model.zip --episodes 10 --seed 82000 --output runs/my_carry_evaluation
uv run python scripts/check_planner_regressions.py --output runs/my_planner_regressions
uv run python scripts/evaluate_colors.py --output runs/my_scripted_evaluation
uv run python scripts/check_carry.py --output runs/my_interface_check
```

Use new output directories. Source-only planning cases live in
`configs/planner_regression_cases.json`, so evaluation needs no deleted experiment.
Prior scene seeds and instruction hashes are retained in `runs/scene_history.json`
for novelty checks; rejected/unsupported instructions remain regression fixtures.

The working curriculum trainer and replay buffer are retained for further training:

```bash
uv run python scripts/train_carry_curriculum.py --resume runs/carry_curriculum_40000 --steps 20000 --seed 200000 --output runs/carry_curriculum_60000
```

`--steps` is the additional budget. See [docs/CURRICULUM.md](docs/CURRICULUM.md) for promotion
criteria and diagnostics. This command starts new training only when you run it.

### Saved results

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

### Checks and interface

```bash
uv run pytest
uv run python scripts/test_env.py --steps 20
```

`MyPickPlace-v1` has two cubes with randomized poses and a green goal. `MyCarrySkill-v1`
exposes 52 numerical observation values and three normalized XYZ actions, a closed
gripper and downward orientation. Carry arrival requires grasp plus 3D error <=2cm
at an 18cm target height. Full placement also requires release and settling.
The numerical state, controller scale, reward and current trained policy are retained.
