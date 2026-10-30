# Distance-Pushing Skill Integration

## Contribution and integration

Buqing Ou's pick-and-place pipeline retains its existing planner, environment,
and executors. This contribution adds a distance-pushing environment, SAC policy,
instruction validation, and routing through the shared `run_instruction` entry.
The router identifies the task category; the selected LLM planner and deterministic
validator decide whether execution is allowed. Routing is not LLM inference, and
the LLM does not train SAC during instruction execution.

Automatic routing uses frozen SAC for pushing and preserves the scripted default
for pick-and-place. The existing carry SAC workflow still requires
`--executor rl --rl-model <carry checkpoint>`.

## Prerequisites

Run commands from the repository root using the group's configured ManiSkill
environment. Pushing also uses `imageio`, `imageio-ffmpeg`, `scipy`, and
`matplotlib`, listed in the project's requirements.txt.

The integration was validated with Python 3.12, ManiSkill 3.0.1, SAPIEN 3.0.3,
Stable-Baselines3 2.9.0, and PyTorch 2.14.1, using CPU physics on Windows.
Install robot assets when prompted by ManiSkill.

On Windows, `run.cmd` looks for `.venv/python.exe`, then
`.venv/Scripts/python.exe`. Alternatively, set the `MYPICKPLACE_PYTHON`
environment variable to the absolute path of your configured Python executable.
The launcher does not install dependencies and returns an exit code without
waiting for a keypress. On Linux/macOS, or in an activated environment, invoke
the script directly:

```bash
python MyPickPlace-ManiSkill/MyPickPlace-ManiSkill/scripts/run_instruction.py --help
```

Pushing currently supports Ollama only, with default endpoint
`http://127.0.0.1:11434` and model `llama3:8b`. After installing Ollama, run
`ollama pull llama3:8b` and ensure the service is running.

Both tasks use `--provider`, `--model`, `--base-url` and the corresponding
`LLM_PROVIDER`, `LLM_MODEL`, `LLM_BASE_URL` environment variables consistently.
`--model` always selects the language model. Use `--push-model` to select a push
policy; `--rl-model` continues to select the carry policy.

## Shared entry commands

Execute a 10 cm push using the included policy (approximately 0.95 MB), without
retraining:

```powershell
.\run.cmd run_instruction --instruction "Push the blue cube forward 10 cm." --seed 540001 --output runs/my_push
```

Check a push plan without executing robot actions:

```powershell
.\run.cmd run_instruction --instruction "Push the blue cube forward 5 cm." --plan-only --output runs/my_push_plan
```

Run the original default pick-and-place workflow:

```powershell
.\run.cmd run_instruction --instruction "Put the blue cube in the green goal region." --output runs/my_pick_place
```

Run the original carry SAC workflow, after obtaining its checkpoint from Buqing:

```powershell
.\run.cmd run_instruction --instruction "Put the blue cube in the green goal region." --executor rl --rl-model runs/carry_curriculum_40000/training/best_model.zip
```

Use a new output directory for each run, or omit `--output` to generate a
timestamped directory. `--skill auto|push|pick_place` selects routing explicitly;
conflicting or mixed actions are rejected. `--plan-file` remains available only
for legacy pick-and-place replay, which is labeled as an offline source.

## Output interface

Both tasks produce `run.json`, retaining the original `plan`, `execution`, and
`status` fields and adding:

| Field | Meaning |
|---|---|
| schema_version | Currently 1 |
| skill | push or pick_place |
| environment_id | Selected simulation environment |
| control_steps | Actual physical control steps |
| physical_success | Execution outcome; null when not executed |
| reason | Rejection, error, or execution outcome reason |

Each task retains its own internal plan schema; consumers should inspect `skill`
before interpreting `plan`. Push runs additionally save `push/llm_exchange.json`,
`plan.json`, `result.json`, `push_trace.json`, and `execution.mp4` when generated.
Errors and rejections are logged. There is no scripted pushing fallback or
implicit physical retry.

Exit codes: 0 = completed or successfully planned; 2 = rejected; 1 = error or
execution failure.

## Scope and evidence

The pushing scene contains one blue cube. Forward means world-coordinate +X.
Accepted distances are 3–15 cm; training and primary evaluation use 5, 10, and
15 cm. Other distances in the accepted interval have not received equivalent
validation. Approach and retreat are scripted; SAC controls XY during pushing,
with fixed Z, orientation, and gripper configuration.

Success requires final longitudinal and lateral errors within 1 cm, low speed,
stability, and no excessive lift or tilt. The skill does not provide arbitrary
object recognition, other directions, multiple objects, or multi-step sequences
such as pushing followed by grasping.

The historical frozen-policy evaluation succeeded in 58/60 episodes: 20/20 at
5 cm, 20/20 at 10 cm, and 18/20 at 15 cm. The scripted baseline also achieved
58/60, so these results do not establish SAC superiority. These evaluations
predate repository integration; summaries and per-episode evidence are included
under `models/push`.

Unified-entry checks and test results are recorded in
`integration_validation.json`. Actual scripted pick-and-place execution was
verified. The repository does not include Buqing's carry checkpoint, so this
integration does not claim an end-to-end rerun of that SAC checkpoint.

## Training and tests

```powershell
.\run.cmd train_push --steps 12000 --seed 1 --output runs/push_training
.\run.cmd evaluate_push_language --output runs/push_language_checks
```

Run the complete test suite in the configured environment:

```bash
python -m unittest discover -s MyPickPlace-ManiSkill/MyPickPlace-ManiSkill/tests
```

Keep `models/push/config.json` beside the checkpoint; execution checks its control
contract. Store generated training artifacts, videos, and runs under `runs/`.
Do not commit virtual environments, Ollama model files, or machine-specific paths.
