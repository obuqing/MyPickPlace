"""Registry of robot skills.

A *skill* is a named task the Panda can learn or execute: the Gym/ManiSkill env it runs
in plus the default env settings. Skills come either from this package (custom envs
in ``robot_skills.envs``) or straight from ManiSkill's built-in task library. Scripts and
trainers look skills up by name instead of hard-coding env ids, so adding a new skill is
one ``register_skill`` call (plus an env, if ManiSkill doesn't already provide it).
"""

from dataclasses import dataclass, field
from typing import Any, Literal

import gymnasium as gym


@dataclass(frozen=True)
class Skill:
    name: str
    env_id: str
    description: str
    source: Literal["robot_skills", "maniskill"]
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    """Defaults passed to ``gym.make``; callers may override any of them."""
    vectorized: bool = False
    """True for ManiSkill envs that support GPU-parallel ``num_envs`` > 1."""

    def ensure_registered(self):
        if self.source == "maniskill":
            import mani_skill.envs  # noqa: F401  (registers ManiSkill's built-in tasks)
        else:
            import robot_skills.envs  # noqa: F401  (registers this package's envs)

    def make(self, **overrides):
        self.ensure_registered()
        return gym.make(self.env_id, **{**self.env_kwargs, **overrides})


SKILLS: dict[str, Skill] = {}


def register_skill(skill: Skill) -> Skill:
    if skill.name in SKILLS:
        raise ValueError(f"Skill already registered: {skill.name}")
    SKILLS[skill.name] = skill
    return skill


def get_skill(name: str) -> Skill:
    try:
        return SKILLS[name]
    except KeyError:
        raise KeyError(f"Unknown skill {name!r}; available: {', '.join(sorted(SKILLS))}") from None


register_skill(Skill(
    name="pick_place",
    env_id="MyPickPlace-v1",
    description="Move the red or blue cube onto the green goal (two-cube tabletop scene).",
    source="robot_skills",
    env_kwargs=dict(obs_mode="state", control_mode="pd_ee_delta_pose"),
    vectorized=True,
))
register_skill(Skill(
    name="pick_place_three",
    env_id="MyPickPlaceThree-v1",
    description="Pick/place scene with red, blue and yellow cubes.",
    source="robot_skills",
    env_kwargs=dict(obs_mode="state", control_mode="pd_ee_delta_pose"),
    vectorized=True,
))
register_skill(Skill(
    name="carry",
    env_id="MyCarrySkill-v1",
    description="Scripted grasp/lift, learned XYZ transport to the goal, scripted release (SAC).",
    source="robot_skills",
))
register_skill(Skill(
    name="carry_three",
    env_id="MyCarryThree-v1",
    description="Carry skill in the three-cube scene, reusing the two-cube SAC policy.",
    source="robot_skills",
))
register_skill(Skill(
    name="peg_insertion",
    env_id="PegInsertionSide-v1",
    description="Grasp a peg and insert it sideways into a box hole (ManiSkill built-in, Panda + wrist cam).",
    source="maniskill",
    env_kwargs=dict(robot_uids="panda_wristcam", obs_mode="state", control_mode="pd_joint_delta_pos"),
    vectorized=True,
))
