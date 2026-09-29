# Repeated Pick and Place

The Task Program repeats cube transfers between two authored target poses. The
configuration directory separates the physical environment, robot deployments,
and expansion settings:

```text
envs/default.yaml       # Default physics scene
envs/newton.yaml        # Newton physics scene and solver settings
task.franka.yaml                # Franka Task Program deployment
task.ur5.yaml                   # UR5 deployment and expansion reference
expansion/repeated_pick_place.yaml
                                # task-facing expansion runtime and overrides
task_program/{program,integration}.yaml
```

Both task deployments use the same Task Program and select the backend through
one environment mapping. The launcher chooses the backend with `--physics`; the
selected environment file owns its `physics` and optional `physics_config`.
Requesting an unavailable backend fails before simulator construction. A task
with one `environment.component` uses `--physics` only to confirm that file's
backend.

| Deployment | Robot component | Physics | Configuration |
|---|---|---|---|
| franka | franka_panda | default or newton | `task.franka.yaml` |
| ur5 | ur5_dh_pgi_140_80 | default or newton | `task.ur5.yaml` |

Inspect the available deployments:

```bash
embodichain show-task embodichain_tasks:repeated_pick_place
embodichain run-env \
  --gym-config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.franka.yaml
```

## Configured trajectory expansion

`task.ur5.yaml` references `expansion/repeated_pick_place.yaml` through its
`expansion.config` field. The expansion file contains the Task Program policy,
scene/affordance/trajectory/visual overrides, runtime recorder events, and the
collection target. The task config remains the only file that selects the
robot and environment.

Run the complete 64-recipe batch schedule through the formal task entry point:

```bash
embodichain run-task \
  --gym-config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml \
  --num-envs 16 --arena-space 2.5 --device cpu --headless --seed 7 \
  --output-dir /tmp/repeated-pick-place-expansion \
  --dataset-dir /tmp/repeated-pick-place-lerobot
```

`--expansion-profile` replaces the profile selected by the task expansion
file. Collection uses `target_episodes` as the final data count and `num_envs`
as batch capacity. The default sequential policy collects 64 rows in four
batches of sixteen. For a small explicit selection, use
`--expansion-recipe-indices 0 16 32 48`; the legacy
`--expansion-candidate-indices` spelling remains accepted. Explicit recipe
selection requires the list length to equal `collection.target_episodes`.
The recorder is owned by the selected environment component and can be
filtered for a dry run with `--filter-dataset-saving`.

The Python showcase remains available for visual debugging and uses the same
`task.ur5.yaml` deployment and referenced expansion config:

```bash
python examples/sim/motion/repeated_pick_place_expansion_combined.py \
  --task-config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml \
  --output-dir /tmp/repeated-pick-place-expansion

python examples/sim/motion/repeated_pick_place_expansion_showcase.py \
  --task-config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml \
  --candidate-indices 0 16 32 48 --num_envs 16 --seed 7 \
  --device cpu --headless --output-dir /tmp/repeated-pick-place-showcase-m4
```

The expansion runtime adds a reset event for the selected candidate and a
synchronous 4×4 camera grid for sixteen environments. Combined expansion with
visual variation requires the configured sensors; use `--disable-sensor` only
with a expansion override that disables visual capture.

The task-local `catalog.yaml` names the two robot deployments; expansion is a
capability of the UR5 deployment rather than a third deployment entry.
