"""Reproducibility helpers: which scene seeds were already used, and policy weight digests."""

import hashlib
import json


def used_scene_seeds(runs):
    """Read actual scene audits/results, not just declared training RNG seeds."""
    seeds, files = set(), []
    history = runs / 'scene_history.json'
    if history.exists():
        seeds.update(json.loads(history.read_text(encoding='utf-8'))['scene_seeds'])
    for pattern in ('*initializations*.jsonl', 'results.json'):
        for path in runs.rglob(pattern):
            text = path.read_text(encoding='utf-8-sig')
            data = [json.loads(line) for line in text.splitlines() if line.strip()] if path.suffix == '.jsonl' else json.loads(text)
            if not isinstance(data, list):
                continue
            count = 0
            for row in data:
                if isinstance(row, dict) and isinstance(row.get('seed'), int):
                    seeds.add(row['seed'])
                    count += 1
            if count:
                files.append(dict(path=str(path.resolve()), records=count,
                                  sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return seeds, files


def parameter_digest(policy):
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.policy.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()
