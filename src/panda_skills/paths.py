"""Canonical project locations (the package is installed editable by `uv sync`)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RUNS_DIR = PROJECT_ROOT / "runs"
CACHE_DIR = PROJECT_ROOT / ".cache"
