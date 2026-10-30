# MyPickPlace: shared instruction entry

The push contribution is additive. All files present in Buqing Ou's commit
`630cc5a` are preserved unchanged. His `run.cmd`, `run_instruction.py`, and
dependency list remain the original versions. Use the separate `run-skills.cmd`
launcher and `run_skills.py` entry for the optional combined interface.

The existing pick-and-place pipeline and the distance-pushing skill share
`run_skills`. Each skill retains its own simulator, planner validation,
and policy. This integration does not compose both skills in one episode.

See [integration instructions](PUSH_INTEGRATION.md) for setup, commands,
interface fields, validation evidence, and limitations.

The original project lives in `MyPickPlace-ManiSkill/MyPickPlace-ManiSkill`.
Its existing documentation and carry training scripts remain available there.
