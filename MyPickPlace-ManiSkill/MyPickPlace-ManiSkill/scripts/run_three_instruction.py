"""Opt-in three-color command; the default two-color entry point is untouched."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from stable_baselines3 import SAC
from envs.three_carry import ThreeCarryEnv
from algorithms.three_color_planner import ThreeColorPlanner
from algorithms.three_color_executor import execute_three_plan
from algorithms.llm_planner import PlanningError


class CommandEnv(ThreeCarryEnv):
    def __init__(self,**kwargs):
        self.control_steps=0
        super().__init__(**kwargs)

    def _sim_step(self,action):
        self.control_steps+=1
        return super()._sim_step(action)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--instruction',required=True)
    parser.add_argument('--seed',type=int,default=95000)
    parser.add_argument('--model',type=Path,default=Path('runs/carry_curriculum_40000/training/best_model.zip'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args()
    output=args.output or Path('runs/three_color_instructions')/datetime.now().strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True,exist_ok=False)
    log=dict(instruction=args.instruction,seed=args.seed,scene_variant='three_color',execution=None,status='starting')
    env,planner=None,None
    code=1
    try:
        torch.set_num_threads(1)
        planner=ThreeColorPlanner(provider='ollama',model='llama3:8b',base_url='http://127.0.0.1:11434',prompt_version='two_stage')
        env=CommandEnv(audit_path=output/'initializations.jsonl')
        env.reset(seed=args.seed,options={'defer_grasp':True})
        assert env.control_steps==0
        log['initial_scene']=env.current_scene()
        plan=planner.plan(args.instruction,log['initial_scene'])
        log['plan']=plan.to_dict()
        if plan.status=='reject':
            log['status']='rejected'
            code=2
        elif args.plan_only:
            log['status']='planned'
            code=0
        else:
            digest=hashlib.sha256(args.model.read_bytes()).hexdigest()
            policy=SAC.load(args.model,device='cpu')
            log['execution']=execute_three_plan(env,plan,policy)
            assert hashlib.sha256(args.model.read_bytes()).hexdigest()==digest
            log.update(status='completed' if log['execution']['success'] else 'execution_failed',
                       model=str(args.model.resolve()),model_sha256=digest)
            code=0 if log['execution']['success'] else 1
    except PlanningError as exc:
        log.update(status='planning_error',error=str(exc))
    finally:
        if planner is not None:
            log['llm_exchange']=planner.exchange
        if env is not None:
            log['control_steps']=env.control_steps
            env.close()
        (output/'run.json').write_text(json.dumps(log,indent=2),encoding='utf-8')
    print(json.dumps({k:log[k] for k in ('status','plan','error') if k in log},indent=2))
    print(f'Log: {output / "run.json"}')
    return code


if __name__=='__main__':
    raise SystemExit(main())
