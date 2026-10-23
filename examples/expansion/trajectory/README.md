# Trajectory augmentation showcases

## Pure trajectory: task-space deformation and segment stitching

Generate two CPU-only figures without constructing a simulator:

```bash
python examples/expansion/trajectory/pure_trajectory_augmentation_showcase.py \
  --output-dir /tmp/embodichain-pure-trajectory
```

The first figure applies four bounded task-space residuals to one analytic tool
path, projects each target back to qpos with batched trajectory IK, and makes
the shared locked endpoints visible. The second builds a phase-indexed segment
bank from three demonstrations, composes Approach from A, Transport from B and
Retreat from C, and highlights the explicitly planned minimum-jerk bridge.
``summary.json`` records IK residuals, selected source lineage and bridge phase
IDs. Add ``--show`` to open the Matplotlib windows after rendering.

This example deliberately contains no image observations, affordance labels or
physical execution. Collision, dynamics and task-success validation remain
required before its outputs could be used as expert trajectories.

### Task-space deformation and trajectory IK

![Endpoint-preserving task-space trajectory variants](assets/task_space_deformation.png)

The black curve is the nominal trajectory. Colored curves use different local
random streams while sharing the exact black endpoint samples. The right panel
shows the corresponding qpos returned by trajectory IK.

### Phase-aligned multi-demo stitching

![Phase-aligned segment bank and stitched trajectory](assets/segment_stitching.png)

The composite takes Approach from demo A, Transport from demo B, and Retreat
from demo C. The red dashed path and shaded time interval are the explicit
minimum-jerk bridge inserted at the discontinuous B/C boundary.

## Task-environment trajectory augmentation

Run the PickUp → Place task scene as a 3×3 parallel simulation: environment 1
replays the nominal trajectory and environments 2–9 replay eight distinct
endpoint-preserving task-space deformations:

```bash
python examples/expansion/trajectory/task_environment_augmentation_showcase.py \
  --output-dir /tmp/embodichain-task-augmentation \
  --num_envs 9 \
  --device cuda \
  --headless
```

The showcase uses the existing Atomic Action task scene and MotionGenerator
planning path. It converts the shared plan to the production
``TaskSpaceTrajectory`` contract, bends the lift/transport/retract paths in the
horizontal plane, and projects each target through batched trajectory IK. The
grasp and release waypoints remain exact. Generation fails instead of silently
reusing a row if nine distinct IK-valid variants cannot be produced. The output
directory receives the nine-way joint/path plots plus a wide-view
``place_auto_play.mp4`` showing all arenas at once.

Useful overrides include:

```bash
--trajectory_variants 9
--task_space_translation_m 0.04
--task_space_rotation_rad 0.0
--variant_seed 7
```

Pass ``--trajectory_variant_mode joint_space`` to return to the earlier
``joint_residual`` / ``via_points`` / ``nullspace_residual`` / retiming demo.

This is a physical execution showcase. It does not claim confirmed dataset
commits; use the expansion coordinator and EpisodeSink flow for collection.
