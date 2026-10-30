"""Plan first, then execute frozen SAC only if validation accepts the task."""
import argparse
import hashlib
import json
from pathlib import Path
from .push_language import plan

ROOT=Path(__file__).resolve().parents[4]


def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def execute(instruction,output,checkpoint,seed=510000,plan_only=False,provider=None,llm_model=None,base_url=None):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    compiled,audit=plan(instruction,model=llm_model,base_url=base_url,provider=provider)
    write(output/'llm_exchange.json',audit); write(output/'plan.json',compiled)
    result=dict(instruction=instruction,plan=compiled,llm_used=bool(audit['attempts']),
                scene_created=False,control_steps=0,retries=0,execution_retries=0,
                extraction_repairs=max(0,len(audit['attempts'])-1),fallback=False,
                planning_status=compiled['status'],physical_success=None)
    if compiled['status']!='execute' or plan_only:
        result['execution_status']='plan_only' if plan_only and compiled['status']=='execute' else 'not_executed'
        write(output/'result.json',result)
        return result
    # Imports that can construct a simulator happen only after acceptance.
    import torch
    import imageio.v2 as imageio
    from stable_baselines3 import SAC
    from .push_skill import PushSkill,CONTRACT
    checkpoint=Path(checkpoint)
    config=json.loads((checkpoint.parent/'config.json').read_text())
    if config['contract']!=CONTRACT:
        result.update(execution_status='error',error='Checkpoint contract mismatch')
        write(output/'result.json',result); return result
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    torch.set_num_threads(1)
    model=SAC.load(checkpoint,device='cpu')
    env=PushSkill(distance_m=compiled['distance_m'],render_mode='rgb_array')
    result['scene_created']=True
    trace=[]
    try:
        obs,_=env.reset(seed=seed)
        result['initial_position_m']=env.start.tolist()
        result['target_position_m']=env.goal.tolist()
        while True:
            action=model.predict(obs,deterministic=True)[0]
            obs,reward,terminated,truncated,info=env.step(action)
            trace.append(dict(push_step=info['push_steps'],action=action.tolist(),
                              cube_position_m=env._position().tolist(),x_error_m=info['x_error_m'],
                              lateral_error_m=info['lateral_error_m']))
            if terminated or truncated: break
        result.update(execution_status='completed',physical_success=info['success'],metrics=info,
                      control_steps=env.physical_steps,seed=seed,checkpoint_sha256=digest,
                      checkpoint_unchanged=digest==hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        imageio.mimsave(str(output/'execution.mp4'),env.frames,fps=20)
        reader=imageio.get_reader(str(output/'execution.mp4'))
        try:
            result['decoded_video_frames']=reader.count_frames()
            assert result['decoded_video_frames']==env.physical_steps
        finally: reader.close()
        write(output/'push_trace.json',trace)
    except Exception as exc:
        result.update(execution_status='error',error=str(exc),control_steps=env.physical_steps)
    finally:
        env.close(); write(output/'result.json',result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--instruction',required=True)
    parser.add_argument('--model',type=Path,default=Path(__file__).resolve().parents[2]/'models/push/best_model.zip')
    parser.add_argument('--seed',type=int,default=510000)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args()
    result=execute(args.instruction,args.output,args.model,args.seed,args.plan_only)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
