import argparse
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import envs
from algorithms.carry_evaluation import evaluate
from algorithms.carry_policy import ScriptedCarryPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("scripted", "sac", "random"), default="scripted")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per color")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-target-distance", type=float)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.policy == "sac":
        if args.model is None:
            parser.error("--model is required for SAC")
        from stable_baselines3 import SAC
        import torch
        torch.set_num_threads(1)
        policy = SAC.load(args.model, device="cpu")
    else:
        policy = ScriptedCarryPolicy() if args.policy == "scripted" else None
    output = args.output or Path("runs/carry_evaluations") / (args.policy + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    report = evaluate(policy, args.episodes, args.seed, output, max_target_distance=args.max_target_distance)
    print(report["summary"], flush=True)


if __name__ == "__main__":
    main()
