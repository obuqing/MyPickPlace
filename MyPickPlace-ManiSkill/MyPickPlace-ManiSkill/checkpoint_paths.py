"""Make the repository checkpoint paths available to the legacy launchers."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from panda_skills.paths import CARRY_CHECKPOINT, PUSH_CHECKPOINT, STACK_CHECKPOINT

__all__ = ["CARRY_CHECKPOINT", "PUSH_CHECKPOINT", "STACK_CHECKPOINT"]
