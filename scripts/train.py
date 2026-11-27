"""Launch the existing trainer for a repository task."""

import argparse
from importlib import import_module
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
TRAINERS = {
    "pick_and_place": "scripts.train_carry_curriculum",
    "push": "panda_skills.push.train_push",
    "stack": "panda_skills.stack.train_stack",
    "peg_insertion": "scripts.train_peg_insertion",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False, add_help=False)
    parser.add_argument("-h", "--help", action="store_true", help="Show global or task-specific help")
    parser.add_argument("--task", choices=TRAINERS)
    args, trainer_args = parser.parse_known_args()
    if args.task is None:
        if args.help or len(sys.argv) == 1:
            parser.print_help()
            return 0
        parser.error("--task is required")
    if args.help:
        trainer_args.append("--help")
    trainer = TRAINERS[args.task]
    sys.path.insert(0, str(ROOT))
    return import_module(trainer).main(trainer_args)


if __name__ == "__main__":
    raise SystemExit(main())
