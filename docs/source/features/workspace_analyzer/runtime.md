# Runtime Workspace Sampling

The `embodichain.lab.sim.workspace` package can reuse a cached workspace during
environment resets. Runtime sampling selects a reachable joint configuration
from the cache and asks the target `Robot` to recompute forward kinematics for
each environment. This accounts for the current robot-base pose instead of
reusing the analyzer's environment-zero pose.

Workspace samples are kinematically reachable candidates. They do not guarantee
that a collision-free trajectory exists from the robot's current state.

## Configure a robot workspace

Workspace caches are configured per robot control part:

```python
from embodichain.lab.sim.workspace import RobotWorkspaceCfg

workspace_cfg = {
    "left_arm": RobotWorkspaceCfg(
        cache_path="/path/to/cache-entry",
        strategy="voxel_uniform",
        voxel_size=0.03,
    )
}
```

The cache path may point to the entry directory containing `results.npz` and
`meta.json`, or directly to `results.npz`.

In a YAML robot configuration:

```yaml
robot:
  robot_type: DexforceW1
  workspace_cfg:
    left_arm:
      cache_path: /path/to/cache-entry
      strategy: voxel_uniform
      voxel_size: 0.03
```

## Sample reachable poses

Use `Robot.sample_reachable_pose()` to obtain full end-effector poses and their
aligned joint configurations:

```python
samples = robot.sample_reachable_pose(
    name="left_arm",
    env_ids=env_ids,
    num_samples=1,
    strategy="voxel_uniform",
    position_bounds=(
        [0.25, -0.35, 0.65],
        [0.75, 0.35, 0.90],
    ),
    max_attempts=32,
)

eef_pose = samples.eef_pose  # (B, K, 4, 4), local arena frame
qpos = samples.qpos          # (B, K, arm_dof)
valid = samples.valid        # (B, K)
```

Available strategies:

- `point_uniform`: sample cached entries uniformly.
- `voxel_uniform`: sample Cartesian voxels uniformly, then select an entry
  inside each voxel. This avoids over-weighting dense areas produced by
  joint-space sampling.

When runtime bounds reject every candidate for an environment, its result has
`valid=False` and `indices=-1`.

## Randomize an object in an environment

The `sample_rigid_object_pose_from_workspace` event functor consumes workspace
positions during reset:

```yaml
env:
  events:
    randomize_cube_workspace:
      func: sample_rigid_object_pose_from_workspace
      mode: reset
      params:
        robot_cfg:
          uid: robot
          control_parts: [left_arm]
        entity_cfg:
          uid: cube
        position_bounds:
          - [0.25, -0.35, 0.65]
          - [0.75, 0.35, 0.90]
        reference_height: 0.72
        max_attempts: 32
```

Only environments with a valid sample are updated. Failed environments retain
their previous object pose.

## Package layout

```text
embodichain/lab/sim/workspace/
├── runtime.py       # RobotWorkspace and WorkspaceSample
├── cfg.py           # RobotWorkspaceCfg
├── analyzer.py      # WorkspaceAnalyzer
├── caches/
├── configs/
├── constraints/
├── metrics/
├── samplers/
└── visualizers/
```

Runtime APIs are lightweight exports. Analyzer APIs are loaded lazily so
importing `Robot` does not load the analyzer or create a circular dependency.
