"""Evaluate source-only regression cases without requiring historical run data."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from algorithms.llm_planner import LLMPlanner, PlanningError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases',type=Path,default=Path(__file__).resolve().parents[1]/'configs/planner_regression_cases.json')
    parser.add_argument('--output',type=Path,default=Path('runs/planner_regressions'))
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    planner = LLMPlanner(provider='ollama',model='llama3:8b',prompt_version='two_stage',base_url='http://127.0.0.1:11434')
    rows = []
    frozen = json.loads(args.cases.read_text(encoding='utf-8'))
    (root/'cases.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    with urlopen(planner.base_url+'/api/tags',timeout=10) as response:
        model = next(m for m in json.load(response)['models'] if m['name']==planner.model)
    manifest = dict(model=model,prompt_sha256=planner.prompt_sha256,repair_prompt_sha256=planner.repair_prompt_sha256,
                    max_repairs=1,physical_execution=False,code_sha256={})
    project=Path(__file__).resolve().parents[1]
    for name in ('algorithms/llm_planner.py','algorithms/intent_planning.py','algorithms/source_spans.py','algorithms/correction_roles.py','algorithms/correction_checks.py','scripts/check_planner_regressions.py'):
        data=(project/name).read_bytes()
        destination=root/'frozen'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(data)
        manifest['code_sha256'][name]=hashlib.sha256(data).hexdigest()
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for case in frozen:
        row=dict(case,semantic_correct=False)
        planner.exchange={}
        try:
            plan=planner.plan(case['instruction'],case['initial_scene'])
            row['plan']=plan.to_dict()
            row['semantic_correct']=(plan.status=='reject') if case['expected_object'] is None else (plan.status=='execute' and plan.object_id==case['expected_object'])
        except PlanningError as exc:
            row['error']=str(exc)
        row['llm_exchange']=planner.exchange
        rows.append(row)
        (root/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        print(f"{len(rows)}/{len(frozen)} {case['id']}: {row['semantic_correct']}",flush=True)
    summary=dict(cases=len(rows),
                 current_correct=sum(r['semantic_correct'] for r in rows),
                 failures=[r['id'] for r in rows if not r['semantic_correct']],
                 false_accepts=sum(r['expected_object'] is None and r.get('plan',{}).get('status')=='execute' for r in rows),
                 wrong_objects=sum(r['expected_object'] is not None and r.get('plan',{}).get('status')=='execute' and r['plan']['object_id']!=r['expected_object'] for r in rows),
                 model_calls=sum(len(r['llm_exchange'].get('attempts',[])) for r in rows),finished=datetime.now().isoformat())
    for name,digest in manifest['code_sha256'].items():
        assert hashlib.sha256((project/name).read_bytes()).hexdigest()==digest
    (root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
