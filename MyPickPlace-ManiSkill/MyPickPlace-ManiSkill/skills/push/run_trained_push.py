"""Frozen SAC inference; no training, scripted fallback or execution retries."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from stable_baselines3 import SAC
from .train_push import evaluate, save
from .push_skill import CONTRACT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--distance-cm',type=float,default=10)
    parser.add_argument('--seed',type=int,default=320000)
    parser.add_argument('--episodes',type=int,default=1)
    parser.add_argument('--all-distances',action='store_true')
    parser.add_argument('--no-video',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if not 3 <= args.distance_cm <= 15:
        parser.error('Distance must be 3..15 cm')
    if not args.model.is_file():
        parser.error('Model file does not exist')
    if args.episodes < 1:
        parser.error('episodes must be positive')
    config_path = args.model.parent/'config.json'
    if not config_path.is_file() or json.loads(config_path.read_text())['contract'] != CONTRACT:
        parser.error('Checkpoint environment contract differs; use its saved source to reproduce that run')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    before = hashlib.sha256(args.model.read_bytes()).hexdigest()
    model = SAC.load(args.model,device='cpu')
    distances = CONTRACT['distances_m'] if args.all_distances else [args.distance_cm/100]
    result = evaluate(model,list(range(args.seed,args.seed+args.episodes)),distances,
                      video=None if args.no_video else args.output)
    result.update(policy='frozen SAC',model_sha256=before,
                  checkpoint_unchanged=before==hashlib.sha256(args.model.read_bytes()).hexdigest(),
                  llm_used=False,retries=0)
    save(args.output/'result.json',result)
    print(result)


if __name__ == '__main__':
    main()
