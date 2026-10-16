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

The Franka and UR5 deployments use the same Task Program and select the backend through
one environment mapping. The launcher chooses the backend with `--physics`; the
selected environment file owns its `physics` and optional `physics_config`.
Requesting an unavailable backend fails before simulator construction. A task
with one `environment.component` uses `--physics` only to confirm that file's
backend.

| Deployment | Robot component | Physics | Configuration |
|---|---|---|---|
| franka | franka_panda | default or newton | `task.franka.yaml` |
| ur5 | ur5_dh_pgi_140_80 | default or newton | `task.ur5.yaml` |
| tianji_marvin | tianji_marvin | default | `task.tianji_marvin.yaml` |

Inspect the available deployments:

```bash
embodichain show-task embodichain_tasks:repeated_pick_place
embodichain run-env \
  --gym-config extensions/embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.franka.yaml
```

## Configured trajectory expansion

`task.ur5.yaml` references `expansion/repeated_pick_place.yaml` through its
`expansion.config` field. The expansion file contains the Task Program policy,
scene/affordance/trajectory/visual overrides, runtime recorder events, and the
collection target. The task config remains the only file that selects the
robot and environment.

The trajectory override declares both the joint-space amplitude
(`trajectory.spatial.joint_offset_scale`) and the deduplication tolerance
(`trajectory.spatial.dedup_tolerance`), both as fractions of each joint's
declared range. The tolerance has to stay below the amplitude: at or above it
every proposal counts as a copy of the reference trajectory, and each recipe
would keep the unmodified nominal motion instead of the requested variant.

Run the complete 64-recipe batch schedule through the formal task entry point:

```bash
embodichain run-task \
  --gym-config extensions/embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml \
  --device cpu --headless --seed 7 \
  --output-dir /tmp/repeated-pick-place/expansion \
  --dataset-dir /tmp/repeated-pick-place/datasets
```

`--expansion-profile` replaces the profile selected by the task expansion
file. Collection uses `target_episodes` as the final data count and `num_envs`
as batch capacity. The default sequential policy collects 64 rows in four
batches of sixteen. For a small explicit selection, set the matching target
and use `--max_episodes 4 --expansion-recipe-indices 0 16 32 48`; the legacy
`--expansion-candidate-indices` spelling remains accepted. Explicit recipe
selection requires the list length to equal `collection.target_episodes`.
The selected environment backend provides the same LeRobot recorder for
Default and Newton; the embodiment supplies the recorder's robot metadata.
The recorder writes structured data below `/tmp/repeated-pick-place/datasets`,
the Expansion manifest below `/tmp/repeated-pick-place/expansion`, and the
debug camera videos below `/tmp/repeated-pick-place/cameras`. Each path can be
overridden by its owning config or CLI option (`--dataset-dir` and
`--output-dir` cover the dataset and manifest); the camera path is owned by the
Expansion declaration. The recorder can be filtered for a dry run with
`--filter-dataset-saving`.

The Python showcase remains available for visual debugging and uses the same
`task.ur5.yaml` deployment and referenced expansion config. Both commands write
below the repository's `output/` directory unless `--output-dir` overrides it:

```bash
python examples/sim/motion/repeated_pick_place_expansion_combined.py \
  --task-config extensions/embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml

python examples/sim/motion/repeated_pick_place_expansion_showcase.py \
  --task-config extensions/embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml \
  --candidate-indices 0 16 32 48 --num_envs 16 --seed 7 \
  --device cpu --headless
```

The expansion runtime adds a reset event for the selected candidate and a
synchronous 4×4 camera grid for sixteen environments. Combined expansion with
visual variation requires the configured sensors; use `--disable-sensor` only
with a expansion override that disables visual capture.

The expansion manifest reports the collection boundary separately from the
recipe schedule. `target_episodes` and `committed_episodes` describe the final
dataset size; `batch_count` describes parallel execution; `attempts` includes
failed retries; and the prepare, commit, and discard reset counts describe
reset boundaries. These counters let a run be compared across different
`num_envs` values without treating a batch or a reset as an episode.

The task-local `catalog.yaml` names the robot deployments; the UR5 deployment
also owns its expansion capability.

## Tianji Marvin tabletop demo

The Marvin deployment uses `envs/tabletop.default.yaml`, a 750 mm tabletop and
a 50 mm cube. Its left arm makes three transfers between two tabletop targets;
the right arm retains its initial target. Both arms inherit the shared compact
ready pose from the Marvin embodiment. The tabletop program and Marvin
integration select verified Pick/Place effects, with stable-object checks and
a 40 mm final position tolerance after every placement. The 25 mm release
clearance has its own attachment/release hysteresis, so an open hand and the
measured drop onto the support are observed before retracting.

Run without creating a training dataset:

```bash
python -m embodichain run-env \
  --gym-config extensions/embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.tianji_marvin.yaml \
  --headless --viser --renderer hybrid --seed 0 --max_episodes 1 --filter-dataset-saving
```

Use `--no-headless` without `--viser` for the native simulation window. The
shared Marvin embodiment owns the gripper-frame and drive calibration. The
robot preset preserves source inertials. This embodiment explicitly supplies
simulation-calibrated mass/inertia overlays with an assumed 1000× scale of the
pinned URDF entries; its source units are not manufacturer-confirmed. The
simulation retains gravity and physical gripper contacts.

Sensor-enabled runs expose `cam_left_wrist` and `cam_right_wrist`, each with
640×360 RGB and aligned ideal depth. The robot preset owns their
D405-equivalent nominal optics and mesh-estimated mounts; these are simulation
defaults, not hardware calibration. Add `--disable-sensor` to skip acquisition.
Dataset recording remains controlled separately by `--filter-dataset-saving`.
