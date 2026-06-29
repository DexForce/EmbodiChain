# NeuralPlanner

````{admonition} Experimental
:class: warning

`NeuralPlanner` is an experimental feature. The API, checkpoint format, and
default parameters may change without a deprecation cycle. It is currently
validated only on the Franka Panda checkpoint shipped for neural motion
generation.
````

`NeuralPlanner` is a learning-based motion generation backend for end-effector
waypoint trajectories. It rolls out a trained waypoint-conditioned APG policy
from the current arm joint state toward one or more target TCP poses.

Unlike `ToppraPlanner`, this planner is not a time-parameterization solver for
joint waypoints. It is intended for `MoveType.EEF_MOVE` inputs and returns a
policy rollout as joint positions plus the corresponding forward-kinematics
poses.

## Features

- **End-effector waypoint planning**: Accepts
  `PlanState(move_type=MoveType.EEF_MOVE, xpos=...)` waypoints.
- **MotionGenerator integration**: Select it with
  `MotionGenCfg(planner_cfg=NeuralPlannerCfg(...))`.
- **Atomic action integration**: EEF-based atomic actions route through
  `TrajectoryBuilder.plan_arm_traj()` and use the neural backend when the active
  planner type is `neural` or `neural_refine`.
- **Checkpoint-backed inference**: Loads a trained transformer waypoint
  checkpoint and runs it in evaluation mode.

## Usage

Pre-trained checkpoints are hosted on HuggingFace and can be downloaded with
`download_neural_planner_checkpoint()`. The repository is gated, so the process
must have access to an authenticated token through `HF_TOKEN` or
`huggingface-cli login` when no local checkpoint is provided.

If you already have `franka.pt`, pass it explicitly or place it at:

```text
~/.cache/embodichain_data/checkpoints/dexforce/neural_motion_generator/franka/franka.pt
```

You can also set `EMBODICHAIN_NEURAL_PLANNER_CHECKPOINT=/path/to/franka.pt`.

```python
from embodichain.data.assets import download_neural_planner_checkpoint
from embodichain.lab.sim.planners import (
    MotionGenCfg,
    MotionGenOptions,
    MotionGenerator,
    MoveType,
    NeuralPlannerCfg,
    PlanState,
)

checkpoint_path = download_neural_planner_checkpoint()

motion_gen = MotionGenerator(
    cfg=MotionGenCfg(
        planner_cfg=NeuralPlannerCfg(
            robot_uid=robot.uid,
            checkpoint_path=checkpoint_path,
            control_part="main_arm",
        )
    )
)

result = motion_gen.generate(
    target_states=[
        PlanState(move_type=MoveType.EEF_MOVE, xpos=waypoint)
        for waypoint in waypoints
    ],
    options=MotionGenOptions(
        control_part="main_arm",
        start_qpos=start_qpos,
    ),
)
```

## Atomic Actions

When the active planner is `neural`, EEF-based atomic actions such as
`MoveEndEffector`, `PickUp`, `MoveHeldObject`, `Place`, and the down phase of
`Press` call the neural planner through `MotionGenerator.generate()`. The raw
policy rollout is then resampled to the fixed waypoint count requested by the
action config.

When the active planner is `neural_refine`, the atomic-action trajectory builder
uses the neural rollout and appends a final IK-refined waypoint before
resampling. Joint-space actions and joint-space return phases, such as
`MoveJoints` and the return phase of `Press`, continue to use joint
interpolation.

## Limitations

- Only `MoveType.EEF_MOVE` waypoint inputs are supported.
- Current transformer checkpoints assume a compatible arm/checkpoint pair; the
  default checkpoint targets Franka Panda 7-DoF.
- Multi-env atomic actions are handled by calling the neural planner once per
  environment, not by a batched policy rollout.
- The planner does not enforce TOPPRA-style velocity or acceleration
  constraints.
- Collision checking and self-collision constraints are not provided by this
  backend.
- Planning failure is explicit; the neural planner does not silently fall back
  to IK interpolation or TOPPRA.

## Examples

Run the standalone planner example:

```bash
python examples/sim/planners/neural_planner.py --headless --device cuda
```

Use a local checkpoint without HuggingFace:

```bash
python examples/sim/planners/neural_planner.py --headless --device cuda --checkpoint-path /path/to/franka.pt
```

The example downloads the checkpoint automatically on first run only when no
local checkpoint is provided or found in the default cache path.

For deciding whether NMG can replace the IK interpolation path in atomic
actions, compare several success notions:

- `action_success`: the atomic action produced a trajectory.
- `strict_pose_success`: the final TCP pose is within the script's strict
  threshold, defaulting to 1 mm and 0.05 rad.
- `all_waypoint_strict_success`: every target waypoint is reached by some
  trajectory sample within the strict threshold. Use this as the primary planner
  quality signal for multi-waypoint NMG comparisons.
- `nmg_threshold_success`: the final TCP pose is within the NMG waypoint
  threshold, defaulting to 5 cm and 0.3 rad.
- downstream task success: the simulated task outcome after trajectory replay.

Use `strict_pose_success`, `final_pos_error`, `final_rot_error`, and the
downstream task outcome rather than `action_success` alone.

## Benchmarks

Use the Franka benchmark in three layers:

1. Demo-matched planner benchmark: mirrors this example's fixed start qpos and
   compact relative TCP offsets.
2. Reachable-FK planner benchmark: uses a broader bank of FK-generated reachable
   target poses.
3. Atomic-action benchmark: runs the Franka `PickUp -> Place -> MoveEndEffector`
   integration path with planner-only and optional physical replay modes.

Run the first two layers with `franka_planner.py`:

```bash
PYTHONPATH="$PWD" conda run -n embodichain040 python -m scripts.benchmark.robotics.nmg.franka_planner \
  --device cuda \
  --planner all \
  --trial_source demo_offsets \
  --neural_checkpoint franka.pt \
  --sample_interval 120
```

```bash
PYTHONPATH="$PWD" conda run -n embodichain040 python -m scripts.benchmark.robotics.nmg.franka_planner \
  --device cuda \
  --planner all \
  --trial_source fk_bank \
  --neural_checkpoint franka.pt \
  --sample_interval 120
```

Run the downstream atomic-action layer separately:

```bash
PYTHONPATH="$PWD" conda run -n embodichain040 python -m scripts.benchmark.robotics.nmg.franka_pick_place \
  --device cuda \
  --planner all \
  --mode planner \
  --neural_checkpoint franka.pt \
  --object sugar_box \
  --support_surface ground
```

For visual inspection, run the physical layer with `--open_window`. The
`attached` object replay mode is useful when you want to inspect the planned
held-object transform without requiring a successful contact grasp:

```bash
PYTHONPATH="$PWD" conda run -n embodichain040 python -m scripts.benchmark.robotics.nmg.franka_pick_place \
  --device cuda \
  --planner ik_interpolate \
  --mode physical \
  --neural_checkpoint franka.pt \
  --object sugar_box \
  --support_surface ground \
  --object_replay_mode attached \
  --replay_control target \
  --open_window
```
