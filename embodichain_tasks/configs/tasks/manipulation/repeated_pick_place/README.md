# Repeated Pick and Place

The Task Program repeats cube transfers between two authored target poses. The
configuration directory separates the physical environment, robot deployments,
and generation settings:

```text
envs/default.yaml       # Default physics scene
envs/newton.yaml        # Newton physics scene and solver settings
task.franka.yaml                # Franka Task Program deployment
task.ur5.yaml                   # UR5 deployment and generation reference
generation/repeated_pick_place.yaml
                                # task-facing generation runtime and overrides
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

## Configured trajectory generation

`task.ur5.yaml` references `generation/repeated_pick_place.yaml` through its
`generation.config` field. The generation file contains the Task Program policy,
scene/affordance/trajectory/visual overrides, runtime recorder events, and the
candidate batch starts. The task config remains the only file that selects the
robot and environment.

Run the complete 64-recipe batch schedule through the formal task entry point:

```bash
embodichain run-task \
  --gym-config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml \
  --num-envs 16 --arena-space 2.5 --device cpu --headless --seed 7 \
  --output-dir /tmp/repeated-pick-place-generation \
  --dataset-dir /tmp/repeated-pick-place-lerobot
```

`--generation-profile` replaces the profile selected by the task generation
file. `--generation-candidate-indices` replaces its candidate starts; the
configured starts `0`, `16`, `32`, and `48` cover all 64 recipes on sixteen
rows. The recorder is owned by the selected environment component and can be
filtered for a dry run with `--filter-dataset-saving`.

The Python showcase remains available for visual debugging and uses the same
`task.ur5.yaml` deployment and referenced generation config:

```bash
python examples/sim/motion/repeated_pick_place_generation_combined.py \
  --task-config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml \
  --output-dir /tmp/repeated-pick-place-generation

python examples/sim/motion/repeated_pick_place_generation_showcase.py \
  --task-config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml \
  --candidate-indices 0 16 32 48 --num_envs 16 --seed 7 \
  --device cpu --headless --output-dir /tmp/repeated-pick-place-showcase-m4
```

The generation runtime adds a reset event for the selected candidate and a
synchronous 4×4 camera grid for sixteen environments. Combined generation with
visual variation requires the configured sensors; use `--disable-sensor` only
with a generation override that disables visual capture.

The task-local `catalog.yaml` names the two robot deployments; generation is a
capability of the UR5 deployment rather than a third deployment entry.
