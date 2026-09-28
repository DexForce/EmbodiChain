# Open Drawer

The Task Program performs one registered `slide` call on the drawer handle.
The generation deployment uses the episode-level profile schema with three
trajectory variants. A small joint residual is authorized only for the free
pull phase; the approach, reach, grasp, and release phases remain fixed.

Run the configured expansion through the formal task entry point:

```bash
embodichain run-task \
  --gym-config embodichain_tasks/configs/tasks/manipulation/open_drawer/task.ur5.generation.yaml \
  --disable-sensor \
  --device cpu --headless --seed 7 \
  --output-dir /tmp/open-drawer-generation \
  --dataset-dir /tmp/open-drawer-lerobot
```

The task-bound `generation` block selects the shared Task Program policy and
supplies the Open Drawer source/phase binding plus candidate indices `0`, `1`,
and `2`. CLI generation options can override those values.

The dataset recorder is owned by `env.yaml` and can be disabled for a run with
`--filter-dataset-saving` (or `--filter_dataset_saving`).
