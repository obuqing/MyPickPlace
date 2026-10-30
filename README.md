# MyPickPlace: shared instruction entry

The existing pick-and-place pipeline and the distance-pushing skill share
`run_instruction`. Each skill retains its own simulator, planner validation,
and policy. This integration does not compose both skills in one episode.

See [integration instructions](PUSH_INTEGRATION.md) for setup, commands,
interface fields, validation evidence, and limitations.

The original project lives in `MyPickPlace-ManiSkill/MyPickPlace-ManiSkill`.
Its existing documentation and carry training scripts remain available there.
