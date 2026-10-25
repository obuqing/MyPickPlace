"""Small real-Llama / symbolic-plan / frozen-SAC integration evaluation."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import envs
import numpy as np
import torch
from stable_baselines3 import SAC
from algorithms.llm_planner import LLMPlanner, PlanningError
from algorithms.rl_plan_executor import execute_rl_plan
from envs.carry_skill import CarrySkillEnv
from validate_carry_generalization import used_scene_seeds


class AuditedEnv(CarrySkillEnv):
    def reset(self, **kwargs):
        self.control_steps = 0
        return super().reset(**kwargs)

    def _sim_step(self, action):
        self.control_steps += 1
        return super()._sim_step(action)


def parameter_digest(policy):
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.policy.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('runs/llm_rl_evaluation'))
    parser.add_argument('--model', type=Path, default=Path('runs/carry_curriculum_40000/training/best_model.zip'))
    parser.add_argument('--cases', type=Path, default=project/'configs/llm_rl_smoke_cases.json')
    parser.add_argument('--seed', type=int, default=80000)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    state = dict(status='running', phase='setup', completed=0, started=datetime.now().isoformat())
    def save():
        (root/'status.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    save()
    environment = None
    try:
        cases_data = args.cases.read_bytes()
        cases = json.loads(cases_data)
        assert len({c['id'] for c in cases}) == len(cases)
        used, audit = used_scene_seeds(Path('runs'))
        assert set(range(args.seed,args.seed+len(cases))).isdisjoint(used)
        planner = LLMPlanner(provider='ollama', model='llama3:8b', base_url='http://127.0.0.1:11434', prompt_version='two_stage')
        with urlopen(planner.base_url+'/api/tags', timeout=10) as response:
            llm = next(m for m in json.load(response)['models'] if m['name'] == planner.model)
        torch.set_num_threads(1)
        policy = SAC.load(args.model, device='cpu')
        original_digest = parameter_digest(policy)
        file_digest = hashlib.sha256(args.model.read_bytes()).hexdigest()
        frozen = root/'frozen'
        frozen.mkdir()
        (frozen/'cases.json').write_bytes(cases_data)
        (frozen/'initial_prompt.txt').write_text(planner.system_prompt, encoding='utf-8')
        from algorithms.intent_planning import INTENT_SYSTEM_PROMPT
        (frozen/'repair_prompt.txt').write_text(INTENT_SYSTEM_PROMPT, encoding='utf-8')
        hashes = {}
        for name in ('scripts/evaluate_llm_rl.py', 'algorithms/rl_plan_executor.py', 'algorithms/llm_planner.py',
                     'algorithms/intent_planning.py','algorithms/source_spans.py','algorithms/correction_roles.py','algorithms/correction_checks.py','envs/carry_skill.py',
                     'envs/pick_place.py','algorithms/scripted_pick_place.py'):
            data = (project/name).read_bytes()
            destination = frozen/'code'/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        manifest = dict(llm=llm, rl_model=str(args.model.resolve()), rl_sha256=file_digest,
                        rl_parameters_sha256=original_digest, code_sha256=hashes,
                        cases_sha256=hashlib.sha256(cases_data).hexdigest(), cases=len(cases),
                        seed_start=args.seed, audited_previous_seed_count=len(used), seed_overlap=[],
                        prompt_sha256=planner.prompt_sha256, generation_options=planner.generation_options,
                        repair_prompt_sha256=planner.repair_prompt_sha256,
                        max_repairs=1, temperature=0, inference_only=True, expected_labels_in_model_input=False,
                        execution_gated_by_expected_labels=False,
                        note='All accepted plans execute in simulation, including semantic mistakes; labels only score results.')
        (root/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        environment = AuditedEnv(audit_path=root/'initializations.jsonl')
        rows = []
        for index, case in enumerate(cases):
            state.update(phase='planning_and_execution', current_case=case['id'])
            save()
            environment.reset(seed=args.seed+index, options={'defer_grasp':True})
            scene = environment.current_scene()
            assert environment.control_steps == 0
            planner.exchange = {}
            row = dict(case, seed=args.seed+index, initial_scene=scene, plan=None, execution=None,
                       semantic_correct=False, correct_input_rejection=False)
            begin = time.perf_counter()
            try:
                plan = planner.plan(case['instruction'], scene)
                row.update(plan=plan.to_dict(), planning_seconds=time.perf_counter()-begin)
                # Scoring cannot alter which accepted plan is executed.
                row['semantic_correct'] = (plan.status == 'reject') if case['expected_object'] is None else (
                    plan.status == 'execute' and plan.object_id == case['expected_object'])
                if plan.status == 'execute':
                    execution_begin = time.perf_counter()
                    row['execution'] = execute_rl_plan(environment,plan,policy)
                    row['execution_seconds'] = time.perf_counter()-execution_begin
                    row['status'] = 'completed' if row['execution']['success'] else 'execution_failed'
                else:
                    row['status'] = 'rejected'
            except PlanningError as exc:
                row.update(status='planning_error', error=str(exc))
                if not case['instruction'].strip() and not planner.exchange:
                    row.update(status='input_rejected', correct_input_rejection=True, semantic_correct=True)
            row.setdefault('planning_seconds',time.perf_counter()-begin)
            row.update(llm_exchange=planner.exchange, control_steps=environment.control_steps,
                       final_scene=environment.current_scene())
            if row['status'] in ('rejected','input_rejected','planning_error'):
                assert environment.control_steps == 0 and row['initial_scene'] == row['final_scene']
            row['end_to_end_correct'] = bool(row['semantic_correct'] and (
                row['execution']['success'] if case['expected_object'] is not None and row['execution'] else
                row['status'] in ('rejected','input_rejected') if case['expected_object'] is None else False))
            rows.append(row)
            (root/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
            state['completed'] = len(rows)
            save()
            print(f"{index+1}/{len(cases)} {case['id']}: {row['status']} semantic={row['semantic_correct']} e2e={row['end_to_end_correct']} controls={row['control_steps']}",flush=True)
        assert parameter_digest(policy) == original_digest
        assert hashlib.sha256(args.model.read_bytes()).hexdigest() == file_digest
        for name, digest in hashes.items():
            assert hashlib.sha256((project/name).read_bytes()).hexdigest() == digest
        positives = [r for r in rows if r['expected_object'] is not None]
        negatives = [r for r in rows if r['expected_object'] is None]
        summary = dict(cases=len(rows), supported=len(positives), unsupported=len(negatives),
            semantic_correct=sum(r['semantic_correct'] for r in rows),
            end_to_end_correct=sum(r['end_to_end_correct'] for r in rows),
            supported_end_to_end_successes=sum(r['end_to_end_correct'] for r in positives),
            executed=sum(r['execution'] is not None for r in rows),
            physical_successes=sum(bool(r['execution'] and r['execution']['success']) for r in rows),
            correct_rejections=sum(r['status']=='rejected' for r in negatives),
            correct_input_rejections=sum(r['correct_input_rejection'] for r in negatives),
            false_accepts=sum(bool(r['plan'] and r['plan']['status']=='execute') for r in negatives),
            false_rejections=sum(bool(r['plan'] and r['plan']['status']=='reject') for r in positives),
            wrong_object_selections=sum(bool(r['plan'] and r['plan']['status']=='execute' and r['plan']['object_id']!=r['expected_object']) for r in positives),
            planning_errors=sum(r['status']=='planning_error' for r in rows),
            model_calls=sum(len(r['llm_exchange'].get('attempts',[])) for r in rows),
            repairs=sum(r['llm_exchange'].get('repair_attempts',0) for r in rows),
            mean_planning_seconds=float(np.mean([r['planning_seconds'] for r in rows])),
            frozen_parameters_verified=True, rejection_zero_control_steps_verified=True,
            statuses=dict(Counter(r['status'] for r in rows)))
        (root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        lines = ['# Llama to structured plan to frozen SAC: end-to-end smoke test','',
                 'Local llama3:8b, two-stage planner, at most one evidence repair; prompt hashes recorded in manifest.',
                 'Frozen step-40000 SAC policy. Grasp/lift and lowering/release are scripted; carry XYZ is learned.',
                 'Planning precedes all physical control. Accepted plans run in the same scene without reset.',
                 'Expected labels score results only and never gate execution. No fallback or execution retry.',
                 'Empty input is rejected before a model call; API/format errors are not scored as semantic rejections.',
                 'This frozen small development set is an integration test, not a general reliability estimate.','',
                 '```json',json.dumps(summary,indent=2),'```','',
                 '| Case | Expected | Plan | Outcome | Correct end to end | Control steps |',
                 '| --- | --- | --- | --- | --- | --- |']
        for r in rows:
            plan = r['plan']
            label = (plan['object_id'] if plan['status']=='execute' else 'reject') if plan else r['status']
            lines.append(f"| {r['id']} | {r['expected_object'] or 'reject'} | {label} | {r['status']} | {r['end_to_end_correct']} | {r['control_steps']} |")
        lines += ['', '## Next improvements','']
        if summary['false_accepts'] or summary['wrong_object_selections']:
            lines.append('Inspect extraction roles, correction scope and source evidence for false accepts/wrong targets before broadening execution. Keep the repair cap and verify changes on a fresh instruction set.')
        elif summary['false_rejections'] or summary['planning_errors']:
            lines.append('Inspect rejected supported cases and separate extraction/compilation issues from API errors. Add focused regressions, then a fresh instruction holdout; keep retries unchanged.')
        elif summary['supported_end_to_end_successes'] < len(positives):
            lines.append('Inspect failed initialization, carry and release separately; do not change the planner to hide physical failures.')
        else:
            lines.append('Broaden untouched English instructions and ambiguous/correction cases across more scene seeds. Keep the policy frozen, and distinguish input rejection, LLM interpretation and physical success in the user-facing executor.')
        (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        state.update(status='complete',phase='complete',finished=datetime.now().isoformat())
    except Exception as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}',finished=datetime.now().isoformat())
        raise
    finally:
        if environment is not None:
            environment.close()
        save()


if __name__ == '__main__':
    main()
