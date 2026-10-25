# Local Llama to frozen RL execution

The existing two-stage planner extracts intent and source evidence, then compiles
and validates a single-object task. Its prompts and one-repair limit are unchanged.
Only a validated `execute` plan can enter the RL executor. The executor revalidates
the live scene and maps the object ID to the selected cube and the existing goal.

The environment is reset with `defer_grasp=True`: this prepares the scene without
physical control. Planning and rejection happen before grasping. For an accepted
plan, `initialize_grasp(object_id)` grasps/lifts in the same scene without another
reset. Frozen SAC consumes the numerical state and produces XYZ carry actions.
Lowering, release, retreat and settling remain scripted. There is no fallback,
execution retry or training. A rejected task makes no control steps.

Run an individual instruction with the local Ollama service available:

```powershell
.\run.cmd run_instruction --executor rl --instruction "Place the blue cube on the green goal." --seed 61000 --output runs/llm_rl_single
```

The default RL checkpoint is `runs/carry_curriculum_40000/training/best_model.zip`.
Use `--plan-only` to inspect a task without executing it. Explicit `--plan-file`
replay remains labeled offline and is not reported as a real LLM test. The default
executor stays scripted for compatibility; `--executor rl` selects frozen SAC.

Run the small integration evaluation (new output directory required):

```powershell
.\run.cmd evaluate_llm_rl --output runs/my_llm_rl_evaluation
```

The fixed 24 English cases include 12 supported requests, 11 unsupported/ambiguous
requests and one empty input. Expected labels never enter model input or decide
whether an accepted plan executes. Thus wrong accepts and wrong-object choices
remain visible rather than being filtered by the test harness.

`results.json` contains the initial/final scene, complete LLM exchange and repair
records, structured plan, control count and numerical carry trajectory. Summary
metrics separate interpretation, physical success, correct rejection, input
rejection and API/format errors. Rejection checks require zero physical control
and unchanged scene state. Frozen model parameters and checkpoint hashes are
verified after inference. The script writes `REPORT.md` with next improvements.

## Correction roles

The repair contract records `correction_scope` as `none`, `object`, `destination`
or `both`. Object and destination references bind distinct source roles. Scope is
resolved from the grounded positions, and a conflicting model label is retained
in the audit instead of changing those references. A goal's color token cannot
be used as the moved object's color token.

For an explicit destination-only correction, `correction_roles.py` can bind a
complete single-task source clause and its target-only replacement directly.
It keeps the original object's exact text/positions, replaces destination evidence,
and records `inherited_object`, `replaced_destination`, `active_destination` and
`source_positions` in compiler checks. Explicit prior object corrections are
processed in order, so a later destination edit preserves the latest object.
Unknown colors are preserved; the normal scene validation still rejects them.
Ambiguous references, multiple objects, extra independent actions and unmatched
language do not enter this narrow binding rule. No scene availability or expected
test labels are used to derive role evidence.

The initial extraction prompt stays unchanged. The repair prompt distinguishes
inherited roles, with the same maximum of one repair and unchanged output budgets.
The policy, reward and execution horizon are unchanged.

Additional correction controls are in `configs/correction_scope_holdout.json`.
These become development regressions after being observed. Archived planner-only
regressions can be replayed with `run.cmd check_planner_regressions`; they do not
count as new physical or independent holdout successes.

`correction_checks.py` also binds a complete object-only replacement while
preserving the original destination, or a complete replacement of both roles.
Only an explicit source correction can cancel the old task. A source check for
additional independent placement requests prevents an omitted second move from
being accepted merely because the model returned one task. Relative or negated
target phrases do not enter the destination-only binding rule.

The final combined assessment is in `runs/correction_roles_verified/`: `end_to_end`
repeats the 24 integration cases, `controls` covers 16 role edits, `boundaries`
covers 8 target/ambiguity boundaries, and `planner_regressions` replays 100 archived
instructions/scenes without physical execution. Compiler checks retain source
offsets and resolved role scopes; raw LLM extraction remains in the exchange log.

Historical failed runs and source snapshots were removed. Regression inputs are
kept as source-only fixtures in `configs/planner_regression_cases.json`. Current
reports and successful videos remain available; old comparison generators were retired.
