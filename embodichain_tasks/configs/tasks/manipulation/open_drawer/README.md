# Open Drawer

The Task Program performs one registered `slide` call on the drawer handle. Its
configuration follows the same task-first layout as the other manipulation
examples:

```text
envs/default.yaml       # Default physics scene
envs/newton.yaml        # Newton physics scene and solver settings
task.franka.yaml                # Franka Task Program deployment
task.ur5.yaml                   # UR5 deployment and expansion reference
expansion/open_drawer.yaml     # task-facing expansion overrides
task_program/{program,integration}.yaml
```

Both task deployments share the program and choose the physical backend with
`--physics`. The selected environment file owns its backend-specific physics
settings; the task files do not duplicate scene or solver fields.

The UR5 deployment references `expansion/open_drawer.yaml` through
`expansion.config`. That file selects the shared Task Program expansion
policy, permits a small joint residual only during the free pull phase, and
requests three trajectory variants.

Run the configured expansion through the formal task entry point:

```bash
embodichain run-task \
  --gym-config embodichain_tasks/configs/tasks/manipulation/open_drawer/task.ur5.yaml \
  --disable-sensor \
  --device cpu --headless --seed 7 \
  --output-dir /tmp/open-drawer-expansion \
  --dataset-dir /tmp/open-drawer-lerobot
```

Open Drawer disables visual expansion, so `--disable-sensor` is valid here.
The dataset recorder is owned by `envs/default.yaml` and can be
filtered for a dry run with `--filter-dataset-saving` (or
`--filter_dataset_saving`). Use `--expansion-profile` or
`--expansion-candidate-indices` to override the referenced expansion values.
