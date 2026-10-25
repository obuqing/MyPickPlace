"""Isolated three-color experiment; frozen old SAC and preserved two-color scene."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import imageio.v2 as imageio
import numpy as np
import torch
from stable_baselines3 import SAC
from algorithms.llm_planner import Plan,SKILLS,PlanningError
from algorithms.three_color_planner import ThreeColorPlanner,validate_three_plan
from algorithms.three_color_executor import execute_three_plan
from algorithms.rl_plan_executor import execute_rl_plan
from envs.three_carry import ThreeCarryEnv
from envs.carry_skill import CarrySkillEnv
from envs.three_pick_place import COLORS
from evaluate_llm_rl import parameter_digest
from record_llm_rl_generalization import CaptionWriter
from validate_carry_generalization import used_scene_seeds

CASES=[
 dict(id='direct_red',instruction='Move the red cube to the green goal.',expected='cube_red'),
 dict(id='direct_blue',instruction='Move the blue cube to the green goal.',expected='cube_blue'),
 dict(id='direct_yellow',instruction='Move the yellow cube to the green goal.',expected='cube_yellow'),
 dict(id='correct_to_red',instruction='Move yellow to the green goal. Actually, use red instead.',expected='cube_red'),
 dict(id='correct_to_blue',instruction='Move red to the green goal. Actually, use blue instead.',expected='cube_blue'),
 dict(id='correct_to_yellow',instruction='Move blue to the green goal. Actually, use yellow instead.',expected='cube_yellow'),
 dict(id='yellow_goal_edit',instruction='Carry yellow to the purple target. Correction: use the green goal instead.',expected='cube_yellow'),
 dict(id='exclude_yellow',instruction='Leave yellow alone and put the blue cube onto the green disk.',expected='cube_blue'),
 dict(id='exclude_red_blue',instruction='Leave red and blue untouched; place the yellow cube in the green goal.',expected='cube_yellow'),
 dict(id='reject_missing_orange',instruction='Move the orange cube to the green goal.',expected=None),
 dict(id='reject_ambiguous',instruction='Move a cube to the green goal.',expected=None),
 dict(id='reject_all_three',instruction='Move the red, blue and yellow cubes to the green goal.',expected=None),
]


class ObservedThreeEnv(ThreeCarryEnv):
    def __init__(self,writer=None,**kwargs):
        self.writer=writer
        self.controls=0
        super().__init__(render_mode='rgb_array' if writer else None,**kwargs)

    def reset(self,**kwargs):
        self.controls=0
        return super().reset(**kwargs)

    def _sim_step(self,action):
        result=super()._sim_step(action)
        self.controls+=1
        if self.writer:
            self.writer.append_data(self.render())
        return result

    def initialize_grasp(self,object_id=None):
        if self.writer:
            self.writer.phase='Scripted grasp and lift (three physical cubes)'
        result=super().initialize_grasp(object_id)
        if self.writer:
            self.writer.phase='Frozen original SAC: XYZ carry'
        return result

    def finish_placement(self):
        if self.writer:
            self.writer.phase='Scripted release and settling'
        return super().finish_placement()


def explicit_plan(color):
    return Plan('execute','cube_'+color,'goal',tuple(SKILLS),'Fixed target for physical-only evaluation')


def main():
    root=Path('runs/three_color_trial')
    baseline=json.loads((root/'baseline/manifest.json').read_text(encoding='utf-8'))
    project=Path(__file__).resolve().parents[1]
    state=dict(status='running',phase='setup',completed=0,started=datetime.now().isoformat())
    def save():
        (root/'status.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    save()
    env=None
    try:
        proposed=set(range(90010,90020))|set(range(91000,91012))|set(range(92000,92003))|set(range(93000,93003))
        used,_=used_scene_seeds(Path('runs'))
        preflight=root/'diagnostics/preflight/manifest.json'
        old_seeds=set(json.loads(preflight.read_text())['physical_seeds']) if preflight.exists() else set()
        assert not ((proposed & used)-old_seeds)
        model_path=Path('runs/carry_curriculum_40000/training/best_model.zip')
        torch.set_num_threads(1)
        policy=SAC.load(model_path,device='cpu')
        weights=parameter_digest(policy)
        planner=ThreeColorPlanner(provider='ollama',model='llama3:8b',base_url='http://127.0.0.1:11434',prompt_version='two_stage')
        manifest=dict(physical_seeds=list(range(90010,90020)),llm_cases=CASES,video_seeds=[92000,92001,92002],
            original_scene_control_seeds=[93000,93001,93002],model=str(model_path.resolve()),
            retrained=False,adapter='Keep 52-value schema; nearest of two physical distractors goes into other_pose',
            initial_prompt_sha256=planner.prompt_sha256,repair_prompt_sha256=planner.repair_prompt_sha256,
            max_repairs=1,execution_retries=0,reset_contact_refresh_physics_steps=1,
            preflight_setup_diagnosis_retained=preflight.exists(),code_sha256={})
        for name in ('envs/three_pick_place.py','envs/three_carry.py','algorithms/three_color_planner.py',
                     'algorithms/three_color_executor.py','scripts/evaluate_three_colors.py'):
            data=(project/name).read_bytes()
            target=root/'experiment_source'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(data)
            manifest['code_sha256'][name]=hashlib.sha256(data).hexdigest()
        (root/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        rows=[]
        env=ObservedThreeEnv(audit_path=root/'physical_initializations.jsonl')
        state['phase']='three_color_physical'
        save()
        for color in COLORS:
            for seed in range(90010,90020):
                obs,_=env.reset(seed=seed,options={'defer_grasp':True})
                assert env.controls==0,'Scene preparation issued a control action'
                scene=env.current_scene()
                positions=[np.array(o['position']) for o in scene['objects']]
                separation=min(np.linalg.norm(a-b) for i,a in enumerate(positions) for b in positions[i+1:])
                assert separation>=.079999 and sum(v.size for v in obs.values())==52
                try:
                    plan=validate_three_plan(explicit_plan(color).to_dict(),scene)
                    execution=execute_three_plan(env,plan,policy)
                except PlanningError as exc:
                    execution=dict(success=False,initialization_success=False,move_success=False,
                        placement_success=False,reason='initial_scene_validation_failed',error=str(exc))
                row=dict(color=color,seed=seed,initial_scene=scene,execution=execution,minimum_initial_separation=float(separation),
                         reset_contact_refresh=env.base.unwrapped.last_reset_contact_refresh)
                rows.append(row)
                (root/'physical_results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
                state['completed']=len(rows)
                save()
                print(f"physical {len(rows)}/30 {color} seed={seed}: success={execution['success']} reason={execution['reason']}",flush=True)
        env.close()
        env=None
        llm_rows=[]
        env=ObservedThreeEnv(audit_path=root/'llm_initializations.jsonl')
        state['phase']='llm_to_three_color_execution'
        save()
        for index,case in enumerate(CASES):
            env.reset(seed=91000+index,options={'defer_grasp':True})
            assert env.controls==0,'Planning must precede physical control'
            scene=env.current_scene()
            planner.exchange={}
            row=dict(case,seed=91000+index,initial_scene=scene,execution=None,correct=False)
            try:
                plan=planner.plan(case['instruction'],scene)
                row['plan']=plan.to_dict()
                if plan.status=='execute':
                    row['execution']=execute_three_plan(env,plan,policy)
                    row['correct']=plan.object_id==case['expected'] and row['execution']['success']
                else:
                    row['correct']=case['expected'] is None
                    assert env.controls==0
            except PlanningError as exc:
                row['error']=str(exc)
            row.update(llm_exchange=planner.exchange,control_steps=env.controls)
            llm_rows.append(row)
            (root/'llm_results.json').write_text(json.dumps(llm_rows,indent=2),encoding='utf-8')
            print(f"llm {index+1}/12 {case['id']}: correct={row['correct']}",flush=True)
        env.close()
        env=None
        state['phase']='three_color_videos'
        save()
        videos=[]
        for index,color in enumerate(COLORS):
            instruction=f'Place the {color} cube onto the green goal.'
            path=root/(color+'.mp4')
            with imageio.get_writer(str(path),fps=20) as writer:
                captions=CaptionWriter(writer,instruction)
                env=ObservedThreeEnv(captions,audit_path=root/(color+'_video_initializations.jsonl'))
                env.reset(seed=92000+index,options={'defer_grasp':True})
                assert env.controls==0,'Video scene preparation issued a control action'
                for _ in range(10):
                    captions.append_data(env.render())
                planner.exchange={}
                video=dict(color=color,seed=92000+index,instruction=instruction,execution=None,success=False)
                try:
                    plan=planner.plan(instruction,env.current_scene())
                    video['plan']=plan.to_dict()
                    captions.plan=f'Plan: {plan.object_id} -> {plan.target_id}'
                    if plan.status=='execute':
                        video['execution']=execute_three_plan(env,plan,policy)
                        video['success']=plan.object_id=='cube_'+color and video['execution']['success']
                except PlanningError as exc:
                    video['error']=str(exc)
                video['llm_exchange']=planner.exchange
                captions.phase='Result: '+('SUCCESS' if video['success'] else 'FAILED / REJECTED')
                for _ in range(20):
                    captions.append_data(env.render())
                assert captions.count==env.controls+30
                video['frames']=captions.count
                env.close()
                env=None
            reader=imageio.get_reader(str(path))
            try:
                assert reader.count_frames()==video['frames']
                imageio.imwrite(str(root/(color+'_preview.jpg')),reader.get_data(min(110,video['frames']-1)))
            finally:
                reader.close()
            videos.append(video)
            (root/'videos.json').write_text(json.dumps(videos,indent=2),encoding='utf-8')
            print(f"video {color}: {video['success']}",flush=True)
        state['phase']='original_two_color_control'
        save()
        controls=[]
        original=CarrySkillEnv()
        try:
            for color in ('red','blue'):
                for seed in range(93000,93003):
                    original.reset(seed=seed,options={'defer_grasp':True})
                    execution=execute_rl_plan(original,explicit_plan(color),policy)
                    controls.append(dict(color=color,seed=seed,execution=execution))
        finally:
            original.close()
        (root/'two_color_controls.json').write_text(json.dumps(controls,indent=2),encoding='utf-8')
        assert weights==parameter_digest(policy)
        for name,digest in baseline['source_sha256'].items():
            assert hashlib.sha256((project/name).read_bytes()).hexdigest()==digest,'Original source changed: '+name
        for name,digest in baseline['protected_artifacts'].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,'Original artifact changed: '+name
        summary=dict(physical={},llm_correct=sum(bool(r['correct']) for r in llm_rows),llm_cases=12,
            llm_false_accepts=sum(c['expected'] is None and c.get('plan',{}).get('status')=='execute' for c in llm_rows),
            videos_successful=sum(bool(v['success']) for v in videos),two_color_successes=sum(bool(r['execution']['success']) for r in controls),
            two_color_attempts=len(controls),original_source_and_artifact_hashes_verified=True,retrained=False)
        for color in COLORS:
            group=[r['execution'] for r in rows if r['color']==color]
            summary['physical'][color]=dict(attempted=len(group),initializations=sum(r['initialization_success'] for r in group),
                carry_successes=sum(r['move_success'] for r in group),placements=sum(r['success'] for r in group),
                strict_successes=sum(r.get('strict_success',False) for r in group))
        (root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        lines=['# Isolated three-color experiment','',
            'Original MyPickPlace-v1 / MyCarrySkill-v1 and original model/replay/videos were preserved and hash-verified.',
            'Their source snapshot and a copied successful checkpoint are under baseline/.',
            'New env IDs: MyPickPlaceThree-v1 and MyCarryThree-v1. Red, blue and yellow all physically exist.',
            'Original SAC weights are frozen: the nearest non-target actor fills the original other_pose slot.',
            'The second distractor is physically simulated and audited but is not separately observable by the frozen network.',
            'Wrong-grasp checks inspect both non-target actors. No training, execution retry, or failure replacement.', '',
            'Preflight checks exposed an inherited reset that grasped before planning and stale CPU contacts after pose initialization. The isolated adapter now initializes only Gym RNG, and the isolated scene always refreshes one physics substep without a control action. Preparation is asserted to issue zero control actions; previous setup diagnostics are retained separately. Results below are a full evaluation after these fixes.', '',
            '| Color | Initialization | Carry | Full placement | Placement with both others <=1cm disturbance |',
            '| --- | --- | --- | --- | --- |']
        for color,s in summary['physical'].items():
            lines.append(f"| {color} | {s['initializations']}/10 | {s['carry_successes']}/10 | {s['placements']}/10 | {s['strict_successes']}/10 |")
        lines += ['',f"Real Llama end-to-end supported/rejection outcomes: {summary['llm_correct']}/12; false accepts: {summary['llm_false_accepts']}.",
            f"Original two-color physical controls: {summary['two_color_successes']}/6.", '',
            '[Red video](red.mp4) | [Blue video](blue.mp4) | [Yellow video](yellow.mp4)', '',
            '## Next improvements','']
        if any(s['placements']<10 for s in summary['physical'].values()):
            lines.append('Inspect failed grasp/carry/release traces and nearest-distractor switches. Replace the single-other adapter with an all-object observation encoder and train in mixed two/three-object scenes before claiming robust transfer.')
        else:
            lines.append('Broaden untouched scene seeds and closer/occluded layouts. For further scaling, use an all-object/set observation encoder rather than hiding the second distractor in a single-pose adapter; compare against the preserved two-object baseline.')
        if summary['llm_correct']<12:
            lines.append('Inspect wrong/rejected planner cases separately from physics; keep unsupported/multiple-object requests rejected without extra retries.')
        lines += ['','This is a small transfer experiment within fixed tabletop physics, not a robustness guarantee.']
        (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        state.update(status='complete',phase='complete',finished=datetime.now().isoformat())
    except Exception as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if env is not None:
            env.close()
        save()


if __name__=='__main__':
    main()
