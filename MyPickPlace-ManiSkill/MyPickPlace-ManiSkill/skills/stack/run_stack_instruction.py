"""Sentence -> Llama 3 plan -> checked plan -> robot stacks the cubes (StackThreeCube-v1).

    .\\run-skills.cmd run_stack_instruction --instruction "Build a tower: green at the bottom, then red, then blue." --seed 1 --output runs/tower1
    # Without Llama, give the plan yourself (for testing the robot side only):
    .\\run-skills.cmd run_stack_instruction --plan red:green,blue:red --seed 1 --output runs/tower_manual
    # ("red:green" = put red ON green. Do not use ">": Windows cmd treats it as file redirection.)
"""
import argparse
import json
import re
from pathlib import Path

import imageio.v2 as imageio
from stable_baselines3 import SAC

from .stack_language import plan as llm_plan
from .three_cube_executor import ThreeCubeStacker

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "stack" / "best_model.zip"


def parse_manual(text):
    steps = []
    for part in text.split(","):
        names = [x.strip().lower() for x in re.split(r"[:>/]", part)]
        if len(names) != 2 or not all(names):
            raise SystemExit(f'Bad --plan step "{part}". Use pick:onto, e.g. red:green,blue:red')
        steps.append({"pick": names[0], "onto": names[1]})
    return steps


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--instruction", help="Natural-language instruction for Llama 3")
    source.add_argument("--plan", help="Manual plan, e.g. red:green,blue:red (skips the LLM)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Trained stacking policy")
    parser.add_argument("--scripted-carry", action="store_true", help="Use the hand-written carry instead of SAC")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    if args.plan:
        checked, audit = {"status": "execute", "steps": parse_manual(args.plan), "reason": "manual plan"}, {"manual": True}
    else:
        checked, audit = llm_plan(args.instruction)
    (args.output / "plan.json").write_text(json.dumps({"plan": checked, "audit": audit}, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    print("Plan:", checked["status"], [f"{s['pick']} -> {s['onto']}" for s in checked["steps"]], "|", checked["reason"])
    if checked["status"] != "execute":
        print("Nothing executed.")
        return

    model = None if args.scripted_carry else SAC.load(args.model, device="cpu")
    env = ThreeCubeStacker(render_mode=None if args.no_video else "rgb_array")
    try:
        env.reset_scene(args.seed)
        result = env.run_plan(checked["steps"], model)
        result.update(instruction=args.instruction, seed=args.seed,
                      carry="scripted" if model is None else str(args.model))
        if not args.no_video:
            video = args.output / f"seed{args.seed}_{'success' if result['success'] else 'failed'}.mp4"
            imageio.mimsave(str(video), env.frames, fps=30)
            result["video"] = str(video)
    finally:
        env.close()
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for step in result["steps"]:
        print(f"  step {step['step'] + 1}: {step['pick']} -> {step['onto']}: {step['reason']}")
    print("SUCCESS" if result["success"] else "FAILED", "| results in", args.output)


if __name__ == "__main__":
    main()
