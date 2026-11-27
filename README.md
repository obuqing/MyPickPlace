# Panda Skills: Panda manipulation on ManiSkill

A collection of Panda robot skills built on [ManiSkill 3](https://github.com/haosulab/ManiSkill)
(installed as a dependency; not vendored). Skills are registered by name in
`panda_skills.skills`, so new ones can be added without restructuring the project.

| Skill | Env id | How it's solved |
|---|---|---|
| `pick_place` / `pick_place_three` | `MyPickPlace-v1` / `MyPickPlaceThree-v1` | Llama 3 plan + scripted grasp/release |
| `carry` / `carry_three` | `MyCarrySkill-v1` / `MyCarryThree-v1` | Frozen SAC policy for XYZ transport |
| `peg_insertion` | `PegInsertionSide-v1` (ManiSkill built-in) | GPU-parallel PPO |

The repository also includes distance pushing, two- and three-cube stacking,
and the original pick/place and carry entry points.

| Workflow | Entry point and documentation |
|---|---|
| Pick/place and carry | `run.cmd` · [Windows commands](#windows-commands) |
| Distance pushing | `run-skills.cmd run_skills` · [Setup and commands](PUSH_INTEGRATION.md) |
| Cube stacking | `run-skills.cmd run_stack_instruction` · [Training and execution](docs/STACK.md) |
| Peg insertion | `uv run python scripts/train_peg_insertion.py` · [Setup](#setup) |

See [demo videos](demos/README.md) for successful runs of the published policies.
Training logs, replay buffers, and intermediate checkpoints stay in the local
`runs/` directory.

## Training and checkpoints

Use `python scripts/train.py --task <task>` with the Python environment for that
task. The launcher passes the remaining flags to the existing trainer. On Windows,
`run-skills.cmd train` uses the configured ManiSkill environment for the SAC tasks:

```powershell
.\run-skills.cmd train --task pick_and_place --steps 20000 --output runs/pick_and_place/curriculum
.\run-skills.cmd train --task push --steps 6000 --output runs/push/baseline
.\run-skills.cmd train --task stack --steps 20000 --output runs/stack/baseline
uv run python scripts/train.py --task peg_insertion --help
```

Use `--help` after `--task` to see that trainer's options. The original
task-specific training commands still work.

The Windows launchers and the shared instruction command use the same
pick/place and carry implementation in `src/panda_skills/`. Run the Python
command directly from `scripts/` when using an installed environment.

| Task | Default published checkpoint |
|---|---|
| `pick_and_place` | `checkpoints/pick_and_place/best_model.zip` (carry SAC) |
| `push` | `checkpoints/push/best_model.zip` (with adjacent `config.json`) |
| `stack` | `checkpoints/stack/best_model.zip` (shared by two- and three-cube stacking) |
| `peg_insertion` | `checkpoints/peg_insertion/best.pt` (reserved until a trained model is available) |

Keep individual training runs under `runs/<task>/`. To use a model from a
specific run, pass the trainer or inference command's existing model argument.

## Setup

The root UV project requires [uv](https://docs.astral.sh/uv/) and uses CUDA 12.8
PyTorch wheels. Peg insertion training requires an NVIDIA GPU. Set
`UV_PROJECT_ENVIRONMENT` in the shell running the UV commands to keep its
dependencies separate from the environment used by the Windows launchers.

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-peg"
```

```bash
export UV_PROJECT_ENVIRONMENT=.venv-peg  # Bash
```

```bash
uv sync --locked     # installs dependencies into .venv-peg
uv run pytest        # unit tests
```

Run every command from the project root.

## Layout

```
pyproject.toml, uv.lock   dependencies (managed by uv)
src/panda_skills/         installable package
  skills.py               skill registry (name -> env id + default env settings)
  envs/                   custom Gym/ManiSkill envs (importing registers them)
  policies/               scripted and base policies
  planning/               LLM instruction -> structured plan
  execution/              run a validated plan with scripted/RL policies
  push/                   distance pushing
  stack/                  two- and three-cube stacking
  training/               trainers: carry SAC helpers, generic GPU PPO (ppo.py)
  paths.py                project directories (configs/, checkpoints/, runs/)
scripts/                  command-line entry points
configs/                  planner cases and env configs
checkpoints/              published policies by task
demos/                    selected successful videos by task
runs/                     experiment outputs and reports (gitignored)
docs/                     per-skill notes
tests/
```

To add a skill: add an env under `src/panda_skills/envs/` (or reuse a ManiSkill
task), call `register_skill(...)` in `skills.py`, and train it with an existing trainer,
for example `panda_skills.training.ppo` for any GPU-vectorized ManiSkill env.

## Peg insertion (PPO)

```bash
uv run python scripts/train_peg_insertion.py                    # full run (~250M env steps)
uv run python scripts/train_peg_insertion.py --num-envs 256 --total-timesteps 2000000   # quick try
uv run python scripts/train_peg_insertion.py --checkpoint runs/peg_insertion/<run>/latest.pt   # resume
uv run python scripts/train_peg_insertion.py --evaluate --checkpoint runs/peg_insertion/<run>/best.pt --capture-video
uv run python scripts/train_peg_insertion.py --help             # all hyper-parameters
```

Each new run writes to `runs/peg_insertion/<run_name>/`: `config.json`, `metrics.jsonl`,
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

The RL checkpoint is `checkpoints/pick_and_place/best_model.zip`.
`--plan-only` skips execution. Explicit `--plan-file` replay is labeled offline.
Only the validated two-stage planner is supported; it allows at most one evidence
repair and no execution retry. See [docs/LLM_RL.md](docs/LLM_RL.md) for role corrections.

### Windows commands

Run from the repository root. With the Conda-style environment at `.venv`,
you can replace `uv run python scripts/<name>.py` with `.\run.cmd <name>`
and pass the same arguments. This also applies to the evaluation and resume
commands below.

```powershell
.\run.cmd start_llama
.\run.cmd run_instruction --executor rl --instruction "Place the blue cube on the green goal." --seed 81000 --output runs/my_rl_instruction
.\run.cmd run_three_instruction --instruction "Place the yellow cube on the green goal." --seed 95000 --output runs/my_yellow_instruction
.\run.cmd test_env --steps 20
```

For the full test suite in that environment:

```powershell
$env:PATH = "$PWD\.venv;$PWD\.venv\Library\bin;$PWD\.venv\Scripts;$env:PATH"
$env:PYTHONPATH = "$PWD\src"
.\.venv\python.exe -m unittest discover -s tests -q
```

See [the carry curriculum](docs/CURRICULUM.md) for resume and generalization
commands and [the three-color experiment](docs/THREE_COLORS.md) for scene details.

### Evaluate and train

```bash
uv run python scripts/evaluate_llm_rl.py --seed 80000 --output runs/my_llm_rl_evaluation
uv run python scripts/evaluate_carry.py --policy sac --model checkpoints/pick_and_place/best_model.zip --episodes 10 --seed 82000 --output runs/my_carry_evaluation
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

### Local experiment results

- `runs/carry_curriculum_40000`: successful trained model, replay buffer and report.
- `runs/carry_generalization_200`: 100 unseen scene seeds per color, 200/200 placements.
- `runs/correction_roles_verified/REPORT.md`: 52 integration outcomes and final
  compiler replay of 100 real-model planning outputs. Raw unsuccessful intermediate
  runs and superseded source snapshots have been removed.
- `runs/llm_rl_generalization_videos/red.mp4` and `blue.mp4`: latest captioned real
  Llama-to-RL videos, with new instruction wordings and scene seeds.

Older successful checkpoints/videos are preserved. Failed checkpoints, experiment
outputs, retired direct-planning prompts and one-off comparison scripts were removed.

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
