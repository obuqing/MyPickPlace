"""Step 6: Run a trained (frozen) stacking policy, or the scripted baseline, and save videos.

    .\run-skills.cmd run_trained_stack --model runs/stack_sac_20000/best_model.zip --episodes 5 --output runs/stack_demo
    .\run-skills.cmd run_trained_stack --scripted --episodes 5 --output runs/stack_scripted_demo
"""
import argparse
import json
from pathlib import Path

from stable_baselines3 import SAC

from .train_stack import evaluate


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=Path, help="Trained SAC checkpoint (.zip)")
    group.add_argument("--scripted", action="store_true", help="Run the hand-written baseline instead")
    parser.add_argument("--seed", type=int, default=50000)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    model = None if args.scripted else SAC.load(args.model, device="cpu")
    seeds = list(range(args.seed, args.seed + args.episodes))
    result = evaluate(model, seeds, video_dir=None if args.no_video else args.output)
    result["policy"] = "scripted" if args.scripted else str(args.model)
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for row in result["rows"]:
        print(f"seed {row['seed']}: {row['reason']}")
    print(f"Stacked {result['successes']}/{result['attempts']}. Results and videos in {args.output}")


if __name__ == "__main__":
    main()
