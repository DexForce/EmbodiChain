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

Run the PickUp → Place task scene with one nominal execution and several
trajectory variants:

```bash
python examples/expansion/trajectory/task_environment_augmentation_showcase.py \
  --output-dir /tmp/embodichain-task-augmentation \
  --num_envs 4 \
  --device cuda \
  --headless
```

The showcase uses the existing Atomic Action task scene and MotionGenerator
planning path. It keeps contact and release waypoints fixed while varying the
authorized free motion using joint residuals, via points, and timing profiles.
The output directory receives the joint-trajectory and tool-path plots plus
``place_auto_play.mp4`` produced by the tutorial runtime.

Useful overrides include:

```bash
--trajectory_variants 4
--spatial_methods joint_residual via_points
--no_redundancy
--variant_seed 7
```

This is a physical execution showcase. It does not claim confirmed dataset
commits; use the expansion coordinator and EpisodeSink flow for collection.
