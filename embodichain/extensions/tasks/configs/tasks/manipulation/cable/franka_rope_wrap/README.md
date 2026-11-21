# Franka Rope Wrap

A Franka FR3 picks up the free end of a rope from the floor, wraps it around a
single post, tightens it and holds the wrap with the gripper closed. The task
uses Newton/DexUni with one CUDA environment and DexSim `0.6.0`.

## Run and record

From the EmbodiChain repository root, run the configured Task Program:

```bash
python -m embodichain run-task \
  --gym_config embodichain/extensions/tasks/configs/tasks/manipulation/cable/franka_rope_wrap/task.franka.yaml \
  --headless --max_episodes 1 --record_trajectory \
  --trajectory_save_dir ./output/franka_rope_wrap
```

The deployment defaults to `no-render`. The robot asset is derived from the
pinned `FrankaV4.zip` bundle with flat finger pads. To populate the asset
cache ahead of an offline run, use `python -m embodichain data download --name
FrankaRopePads`. Set `EMBODICHAIN_DATA_ROOT` before launching if you use a different
cache directory.

The packaged [`cable.usda`](cable.usda) contains 97 centerline points and an
8 mm rope diameter. [`env.yaml`](env.yaml) fixes segment zero at the floor
clamp and supplies the contact and rod parameters. The fingers grip by physical
contact, and the arm executes the planned position and velocity targets. The
rope follows the resulting forces throughout the episode.

## Acceptance

The [program](task_program/program.yaml) uses `simulation.cable_pickup` to
approach, grip and lift the live cable. The other six stages use `MoveJoints` to
go from home through the wrap and the final two-second hold. The default
36-second control sequence uses 2,160 steps
at 60 Hz. The [evaluation graph](task_program/evaluation.yaml) requires all 19
checks to pass, including floor pickup, grip retention, material slip, bilateral
contact, measured winding, contact arc, clearance, anchor stability and the
final hold. The standard trajectory recorder saves successful episodes for
analysis. Replay a cable trajectory from reset with the same Gym configuration
and `run-task --replay --replay_mode dynamic --replay_trajectory <path>`.
