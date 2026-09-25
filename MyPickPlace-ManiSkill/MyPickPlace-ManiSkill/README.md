# MyPickPlace-v1 for ManiSkill

A minimal custom ManiSkill pick-and-place project designed as a clean base for
connecting a later RL, imitation-learning, planning, or optimization framework.

## Task

- Robot: Franka Panda
- Object: red cube
- Goal: green circular tabletop region
- Default observation: state
- Default controller: `pd_ee_delta_pose`
- Success: cube center is inside the target region, cube is released, robot is static
- Episode limit: 200 control steps

## 1. Create an environment

Recommended:

```bash
conda create -n mypickplace python=3.11 -y
conda activate mypickplace
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If your CUDA/PyTorch setup needs a specific build, install PyTorch for your
machine first, then install the remaining requirements.

## 2. Run from the project root

```bash
python scripts/inspect_env.py
```

This opens the scene and prints the action/observation interfaces.

Then run the smoke test:

```bash
python scripts/test_env.py
```

The smoke test deliberately uses random actions. Its purpose is to verify that
reset, stepping, reward, termination, and rendering all work. It is not expected
to solve pick-and-place.

## 3. Environment creation

```python
import gymnasium as gym
import mani_skill.envs
import envs

env = gym.make(
    "MyPickPlace-v1",
    robot_uids="panda",
    num_envs=1,
    obs_mode="state",
    control_mode="pd_ee_delta_pose",
    reward_mode="dense",
    render_mode="human",
)

obs, info = env.reset(seed=0)
```

Importing `envs` is important because that executes the `@register_env`
decorator for `MyPickPlace-v1`.

## 4. Algorithm boundary

Put your method in `algorithms/` and keep this interface:

```python
class MyAlgorithm:
    def reset(self):
        ...

    def act(self, observation):
        action = ...
        return action
```

The rollout code stays generic:

```python
obs, info = env.reset()
algorithm.reset()

while True:
    action = algorithm.act(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    if bool((terminated | truncated).item()):
        break
```

This is intentional. Your algorithm should not access `env.unwrapped.cube`,
`env.unwrapped.agent`, or other simulator internals. Everything the algorithm
needs should come through observations.

## 5. Project layout

```text
MyPickPlace-ManiSkill/
├── envs/
│   ├── __init__.py
│   └── pick_place.py
├── algorithms/
│   ├── __init__.py
│   └── policy.py
├── scripts/
│   ├── inspect_env.py
│   ├── test_env.py
│   └── run_policy.py
├── configs/
│   └── pick_place.yaml
├── requirements.txt
└── README.md
```

## 6. Recommended development order

1. Run `inspect_env.py`.
2. Run `test_env.py`.
3. Inspect the exact state/action dimensions on your installed ManiSkill build.
4. Replace `RandomPolicy` with your algorithm.
5. Only after state-based control works, add RGB-D observations.
6. Then add object/physics/camera randomization if your research requires it.

## Notes

This project follows the current ManiSkill custom-task API and the structure of
the official PickCube task. ManiSkill is actively developed, so if a future
release changes an API, compare `envs/pick_place.py` with the PickCube task from
the installed version.
