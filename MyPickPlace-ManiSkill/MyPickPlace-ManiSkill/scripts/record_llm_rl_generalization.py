"""Two preselected new instructions/scenes, real Llama planning and full videos."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import textwrap

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from stable_baselines3 import SAC
from algorithms.llm_planner import LLMPlanner, PlanningError
from algorithms.rl_plan_executor import execute_rl_plan
from evaluate_llm_rl import parameter_digest
from validate_carry_generalization import RecordedCarryEnv, used_scene_seeds


CASES = [
    dict(color='red',seed=72001,instruction='Transport the red block onto the violet marker. Correction: select the green disk instead.'),
    dict(color='blue',seed=72039,instruction='Carry red into the green circle. Actually, put blue there. Correction: choose the green goal instead.'),
]


class CaptionWriter:
    def __init__(self,writer,instruction):
        self.writer=writer
        self.instruction=instruction
        self.plan='Planning before robot motion'
        self.phase='Initial scene'
        self.count=0
        self.font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
        self.small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)

    def append_data(self,frame):
        frame=Image.fromarray(np.asarray(frame,dtype=np.uint8))
        canvas=Image.new('RGB',(512,704),(20,25,36))
        canvas.paste(frame,(0,192))
        draw=ImageDraw.Draw(canvas)
        draw.text((12,9),'Llama 3 -> structured plan -> frozen SAC',font=self.font,fill=(240,245,255))
        y=38
        for line in textwrap.wrap(self.instruction,width=62):
            draw.text((12,y),line,font=self.small,fill=(225,230,240))
            y+=20
        draw.text((12,112),self.plan,font=self.small,fill=(120,210,255))
        draw.text((12,142),self.phase,font=self.small,fill=(180,240,160))
        draw.text((12,168),'One attempt | scripted grasp/release | learned XYZ carry',font=self.small,fill=(180,190,205))
        self.writer.append_data(np.asarray(canvas))
        self.count+=1


class CaptionedEnv(RecordedCarryEnv):
    def initialize_grasp(self,object_id=None):
        self.writer.phase='Scripted grasp and lift'
        result=super().initialize_grasp(object_id)
        self.writer.phase='Frozen SAC: XYZ transport'
        return result

    def finish_placement(self):
        self.writer.phase='Scripted lowering, release and settling'
        return super().finish_placement()


def main():
    root=Path('runs/llm_rl_generalization_videos')
    root.mkdir(parents=True,exist_ok=False)
    state=dict(status='running',started=datetime.now().isoformat(),completed=0)
    def save():
        (root/'status.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    save()
    try:
        used,_=used_scene_seeds(Path('runs'))
        previous_instructions=set()
        history_path=Path('runs/scene_history.json')
        history=json.loads(history_path.read_text()) if history_path.exists() else {}
        prior_hashes=set(history.get('instruction_sha256',[]))
        for path in Path('runs').rglob('results.json'):
            data=json.loads(path.read_text(encoding='utf-8-sig'))
            if isinstance(data,list):
                previous_instructions.update(r['instruction'] for r in data if isinstance(r,dict) and isinstance(r.get('instruction'),str))
        assert all(c['seed'] not in used and c['instruction'] not in previous_instructions
                   and hashlib.sha256(c['instruction'].encode()).hexdigest() not in prior_hashes for c in CASES)
        model_path=Path('runs/carry_curriculum_40000/training/best_model.zip')
        digest=hashlib.sha256(model_path.read_bytes()).hexdigest()
        torch.set_num_threads(1)
        policy=SAC.load(model_path,device='cpu')
        weights=parameter_digest(policy)
        planner=LLMPlanner(provider='ollama',model='llama3:8b',base_url='http://127.0.0.1:11434',prompt_version='two_stage')
        manifest=dict(cases=CASES,rl_model=str(model_path.resolve()),rl_sha256=digest,
            initial_prompt_sha256=planner.prompt_sha256,repair_prompt_sha256=planner.repair_prompt_sha256,
            previously_used_scene_seeds=len(used),prior_instruction_count=len(previous_instructions),
            new_scene_seeds=True,new_exact_instruction_wordings=True,max_repairs=1,no_execution_retries=True,
            selection='Both instructions and seeds frozen before inference; failures retained without replacement',
            scope='New wording and scene positions within the existing task/physics distribution')
        project=Path(__file__).resolve().parents[1]
        manifest['code_sha256']={}
        for name in ('scripts/record_llm_rl_generalization.py','algorithms/llm_planner.py','algorithms/intent_planning.py',
                     'algorithms/source_spans.py','algorithms/correction_roles.py','algorithms/correction_checks.py',
                     'algorithms/rl_plan_executor.py','envs/carry_skill.py','envs/pick_place.py','algorithms/scripted_pick_place.py'):
            data=(project/name).read_bytes()
            target=root/'frozen'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(data)
            manifest['code_sha256'][name]=hashlib.sha256(data).hexdigest()
        (root/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        rows=[]
        for case in CASES:
            row=dict(case,plan=None,execution=None,semantic_correct=False,success=False)
            path=root/(case['color']+'.mp4')
            with imageio.get_writer(str(path),fps=20) as video:
                captions=CaptionWriter(video,case['instruction'])
                environment=CaptionedEnv(captions,audit_path=root/(case['color']+'_initializations.jsonl'))
                try:
                    environment.reset(seed=case['seed'],options={'defer_grasp':True})
                    scene=environment.current_scene()
                    row['initial_scene']=scene
                    planner.exchange={}
                    for _ in range(10):
                        captions.append_data(environment.render())
                    try:
                        plan=planner.plan(case['instruction'],scene)
                        row['plan']=plan.to_dict()
                        captions.plan=('Plan: '+str(plan.object_id)+' -> '+str(plan.target_id)) if plan.status=='execute' else 'Plan: reject'
                        if plan.status=='execute':
                            row['execution']=execute_rl_plan(environment,plan,policy)
                            row['semantic_correct']=plan.object_id=='cube_'+case['color']
                            row['success']=bool(row['semantic_correct'] and row['execution']['success'])
                        else:
                            row['reason']=plan.reason
                    except PlanningError as exc:
                        row['error']=str(exc)
                    row.update(llm_exchange=planner.exchange,physical_frames=environment.frame_count,
                               final_scene=environment.current_scene())
                    captions.phase='Result: '+('SUCCESS' if row['success'] else 'FAILED / REJECTED (not replaced)')
                    for _ in range(20):
                        captions.append_data(environment.render())
                    row['frames']=captions.count
                    assert row['frames']==row['physical_frames']+30
                finally:
                    environment.close()
            reader=imageio.get_reader(str(path))
            try:
                row['decoded_frames']=reader.count_frames()
                assert row['decoded_frames']==row['frames']
                row['seconds']=row['frames']/20
                imageio.imwrite(str(root/(case['color']+'_preview.jpg')),reader.get_data(min(110,row['frames']-1)))
            finally:
                reader.close()
            row.update(video=str(path.resolve()),video_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            rows.append(row)
            (root/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
            state['completed']=len(rows)
            save()
            print(f"{case['color']} seed={case['seed']} success={row['success']} frames={row['frames']}",flush=True)
        assert parameter_digest(policy)==weights and hashlib.sha256(model_path.read_bytes()).hexdigest()==digest
        lines=['# Two new-wording / new-scene end-to-end videos','',
               'Real local Llama 3 planning followed by frozen SAC XYZ carry, with scripted grasp/lift and release.',
               'Instructions and scene seeds were selected before inference and absent from previous run records.',
               'Exactly one attempt per case; no replacement on failure. Captions show the instruction, structured plan and active controller.',
               'Same task/physics distribution; two videos are examples, not a generalization success-rate estimate.','']
        for row in rows:
            lines += [f"## {row['color']}",'',row['instruction'],'',
                      f"Seed: {row['seed']}; plan: {row['plan']}; success: {row['success']}; duration: {row['seconds']}s.",'',
                      f"[Video]({row['color']}.mp4)",'']
        lines += ['## Next improvements','',
                  'Evaluate an untouched batch of longer role corrections across more new seeds; test changed friction/mass separately before claiming physics robustness.']
        (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        state.update(status='complete',successes=sum(r['success'] for r in rows),finished=datetime.now().isoformat())
    except Exception as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()


if __name__=='__main__':
    main()
