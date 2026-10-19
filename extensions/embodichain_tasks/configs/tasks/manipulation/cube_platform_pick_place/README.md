# Cube Platform Pick and Place

This configuration-defined UR5 task picks a 4 cm native Cube and places it on a
16 cm square native Cube platform. Loading its deployment registers the common
`EmbodiedEnv`; no task-specific Python environment is required.

| Configuration | Responsibility |
| --- | --- |
| `task.ur5.yaml` | Runnable ID, reusable UR5 embodiment, execution policy and one-episode collection target |
| `env.yaml` | Default physics backend, cube/platform geometry, initial states and recording settings |
| `task_program/program.yaml` | Pick, lift check, semantic Place, settling and upright check |
| `task_program/integration.yaml` | Logical cube/platform bindings and parent-relative support pose |

From the repository root, with the SDK and its dependencies installed:

```bash
embodichain run-task --gym-config \
  extensions/embodichain_tasks/configs/tasks/manipulation/cube_platform_pick_place/task.ur5.yaml
```

The default support frame is 0.047 m above the platform origin: 0.025 m platform
half-height + 0.020 m cube half-height + 0.002 m release clearance.

`examples/gym/asset_generalization/run.py` reads these same files and produces
4 cm/5 cm batches with the matching placement height and fresh deployment IDs.
Its measured acceptance additionally checks the full transformed cube footprint
and bottom-to-platform contact height before saving a trajectory. See
`examples/gym/asset_generalization/README.md` for that collection workflow.
