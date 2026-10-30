"""Fixed development suite; expected labels never control physical execution."""
import argparse
import json
from pathlib import Path
from .push_instruction import execute,write,ROOT


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cases',type=Path,default=Path(__file__).with_name('language_cases.json'))
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    cases=json.loads(args.cases.read_text(encoding='utf-8'))
    write(args.output/'cases.json',cases)
    rows=[]
    for i,case in enumerate(cases):
        # Intentionally pass only the instruction, never expected labels.
        result=execute(case['instruction'],args.output/case['id'],Path(__file__).resolve().parents[2]/'models/push/best_model.zip',
                       seed=530000+i,plan_only=args.plan_only)
        plan=result['plan']
        correct=plan['status']==case['expected']
        if case['expected']=='execute' and correct:
            correct=abs(plan['distance_m']-case['distance_m'])<1e-8
        rows.append(dict(id=case['id'],expected=case['expected'],plan_status=plan['status'],
                         planning_correct=correct,physical_success=result['physical_success'],
                         control_steps=result['control_steps'],scene_created=result['scene_created']))
        write(args.output/'results.json',rows)
        print(rows[-1],flush=True)
    summary=dict(real_model='llama3:8b',cases=len(rows),planning_correct=sum(r['planning_correct'] for r in rows),
                 api_errors=sum(r['plan_status']=='error' for r in rows),
                 correct_rejections=sum(r['expected']=='reject' and r['plan_status']=='reject' for r in rows),
                 physical_successes=sum(r['physical_success'] is True for r in rows),
                 physical_attempts=sum(r['scene_created'] for r in rows),
                 rejected_cases_zero_control=all(r['control_steps']==0 and not r['scene_created'] for r in rows if r['plan_status']=='reject'),
                 plan_only=args.plan_only,scope='Fixed development suite; not a general natural-language benchmark')
    write(args.output/'summary.json',summary); print(summary,flush=True)


if __name__=='__main__': main()
