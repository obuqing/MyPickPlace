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

## Three cubes + Llama 3 (`StackThreeCube-v1`)

Red, green and blue cubes; Llama 3 turns a sentence into ordered steps, code checks them,
and the SAME trained stacking policy executes every step (no retraining).

| File | Role |
|---|---|
| `three_cubes_env.py` | `StackThreeCube-v1`: StackCube-v1 plus a blue cube; cubes at least 10 cm apart |
| `three_cube_executor.py` | runs a plan step by step: scripted grasp, SAC carry/place, scripted release |
| `stack_language.py` | Llama 3 (Ollama, temperature 0, JSON schema) + deterministic plan checks + one repair |
| `run_stack_instruction.py` | sentence -> plan -> robot -> video |
| `evaluate_stack_language.py` + `language_cases.json` | 14 English/Chinese test sentences (9 valid, 5 must be rejected) |

Why one policy is enough: it only observes positions relative to its target, so the
executor lifts each cube to the same height above its target as in training (6 cm above
the stacking height), including on top of a 2-cube tower.

Two physics details found in testing:
- The Panda hand is ~20 cm wide along the finger direction and knocked over a nearby tower
  while descending. The executor now turns the hand (cubes allow 90-degree choices) so the
  fingers point away from the nearest cube.
- For a cube on top of a 2-cube tower, CPU physics reports a jittering ~2 cm/s velocity
  although the cube does not move, so ManiSkill's velocity-based "static" test fails.
  For that case the executor checks position stability instead (< 2 mm over 0.5 s).

Planner version 2 (current). Version 1 asked Llama for ordered robot steps; with llama3:8b it
planned 11/14 test sentences correctly and failed all 3 "X at the bottom ... Z on top"
sentences (e.g. it built blue-green-red for "green at the bottom, then red, then blue on top"),
and the checks accepted those wrong but physically possible plans. Version 2 asks Llama only
for the tower from bottom to top; code builds the steps (always bottom-up) and checks the tower
against explicit words in the sentence ("X at the bottom / in the middle / on top", "X on Y",
"X在最下面/中间/最上面", "把X放到Y上"). A contradiction is sent back to Llama once with the reason.

Plan checks (code, not the LLM): only colours named in the sentence; every named colour
used; no cube on itself; a cube with something on it cannot be picked; a cube can hold only
one cube; 1-2 steps; reject other colours, pushing/removing, negations, conditions.
A wrong order (e.g. top cube first) is sent back to Llama once with the reason.

```powershell
.\run.cmd start_llama                       # Ollama + llama3:8b must be running
.\run-skills.cmd run_stack_instruction --instruction "Build a tower with green at the bottom, then red, then blue on top." --seed 1 --output runs/tower1
.\run-skills.cmd run_stack_instruction --plan red:green,blue:red --seed 1 --output runs/tower_manual   # robot only, no LLM ("red:green" = red ON green)
.\run-skills.cmd evaluate_stack_language --output runs/stack_language_eval            # planner accuracy
.\run-skills.cmd evaluate_stack_language --execute --output runs/stack_language_exec  # planner + robot
```

Reference (my CPU test, 30 scenes: 20 two-step towers + 10 single stacks, fixed plans):
SAC policy 27/30, scripted carry 29/30. Planner plumbing, checks and repair were tested
against a mock Llama server; real Llama 3 accuracy must be measured with Ollama running.
