"""Canonical project locations (the package is installed editable by `uv sync`)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
CARRY_CHECKPOINT = CHECKPOINTS_DIR / "pick_and_place" / "best_model.zip"
PUSH_CHECKPOINT = CHECKPOINTS_DIR / "push" / "best_model.zip"
STACK_CHECKPOINT = CHECKPOINTS_DIR / "stack" / "best_model.zip"
PEG_INSERTION_CHECKPOINT = CHECKPOINTS_DIR / "peg_insertion" / "best.pt"
RUNS_DIR = PROJECT_ROOT / "runs"
CACHE_DIR = PROJECT_ROOT / ".cache"
