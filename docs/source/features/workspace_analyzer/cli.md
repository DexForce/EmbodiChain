# Workspace Analyzer CLI

The `embodichain analyze-workspace` command analyzes a robot's reachable
workspace from either a **predefined EmbodiChain robot** or a **URDF/USD asset**.
It selects an analysis mode and core parameters, **caches the reachable
workspace data** to disk for reuse by other applications (e.g. environment data
generation), and visualizes the result in the simulation window.

## Quick start

```bash
# Predefined robot: control parts + solver come from the preset, so only
# --control-part (optional) is needed.
embodichain analyze-workspace \
    --robot franka_panda --mode joint_space --num-samples 20000

embodichain analyze-workspace \
    --robot cobotmagic --control-part left_arm \
    --mode cartesian_space --bounds -0.5 0.5 -0.5 0.5 0.6 1.5

# Generic URDF asset: requires --ee-link (and --joints for arms with grippers)
embodichain analyze-workspace \
    --asset /path/to/panda.urdf \
    --ee-link fr3_hand_tcp --joints "fr3_joint[1-7]" \
    --mode joint_space --num-samples 20000
```

`--robot` and `--asset` are mutually exclusive; exactly one is required,
including when previewing an existing cache.

## Predefined robots (`--robot`)

Use `--robot NAME` to analyze one of EmbodiChain's built-in robots
(`embodichain/lab/sim/robots/`). The preset already defines the URDF, control
parts, and kinematics solver, so `--ee-link` and `--joints` are **not** needed --
just pick the control part to analyze with `--control-part` (optional; defaults
to `left_arm`/`right_arm` or the first available part).

| Argument | Description |
|----------|-------------|
| `--robot NAME` | Predefined robot. Choices: `franka_panda`, `cobotmagic`, `dexforce_w1`, `ur`. |
| `--robot-params JSON` | JSON dict of variant overrides, e.g. `{"robot_type":"ur5"}` or `{"version":"v021","arm_kind":"industrial"}`. |
| `--control-part NAME` | Control part to analyze (e.g. `arm`, `left_arm`, `right_arm`). Optional; auto-selected if omitted. |

```bash
embodichain analyze-workspace \
    --robot dexforce_w1 \
    --robot-params '{"version":"v021","arm_kind":"industrial"}' \
    --control-part left_arm --mode joint_space --num-samples 20000
```

Available control parts per robot: `franka_panda` (`arm`, `hand`), `cobotmagic`
(`left_arm`, `left_eef`, `right_arm`, `right_eef`), `dexforce_w1`
(`left_arm`, `right_arm` for industrial), `ur` (`arm`).

## Generic asset (`--asset`)

The robot asset is loaded with `SimulationManager.add_robot`. The
end-effector link (`--ee-link`) and control-part joints (`--joints`) are
required to build the kinematics solver used for FK/IK.

| Argument | Description |
|----------|-------------|
| `--asset PATH` | Robot asset (`.urdf` / `.usd` / `.usda` / `.usdc`). |
| `--urdf PATH` | URDF for the kinematics solver. Required for USD assets; defaults to `--asset` when it is a URDF. |
| `--ee-link NAME` | End-effector link name (FK/IK target). **Required with `--asset`.** |
| `--joints SPEC` | Comma-separated joint names or a regex for the control part (e.g. `fr3_joint[1-7]`, `joint1,joint2`). Default: all joints (`.*`). |
| `--root-link NAME` | Root/base link for the solver. If omitted, the URDF root is used. |
| `--solver {pytorch,pinocchio,pink}` | Kinematics solver (default: `pytorch`, works for any URDF). |
| `--tcp TX TY TZ RX RY RZ` | Tool center point: translation (m) + rotation (xyz euler, deg). |
| `--init-qpos F...` | Initial joint positions for the control part (also used as the IK reference pose). Applies to both `--robot` and `--asset`. |
| `--fix-base / --no-fix-base` | Fix the robot base (default: fixed). |

```{attention}
USD assets cannot be parsed by the kinematics solver (pytorch-kinematics).
When using a USD asset, also pass `--urdf` pointing at the matching URDF. For
arms with a gripper, specify the arm joints via `--joints` to avoid a
joint-count mismatch with the solver's serial chain.
```

## Analysis modes

`--mode {joint_space, cartesian_space, plane_sampling}` (default: `joint_space`)

- **`joint_space`** - Sample joint configurations within limits, compute FK, and
  collect the reachable end-effector points.
- **`cartesian_space`** - Sample Cartesian positions, compute IK, and record
  which points are reachable. Use `--bounds XMIN XMAX YMIN YMAX ZMIN ZMAX` to
  restrict the sampling region (omitted: computed from joint-space FK).
- **`plane_sampling`** - Sample on a 2D plane and verify reachability via IK.
  Configure the plane with `--plane-normal NX NY NZ`, `--plane-point X Y Z`,
  and optionally `--plane-bounds UMIN UMAX VMIN VMAX`.

Common parameters: `--num-samples`, `--ik-samples-per-point` (Cartesian/plane
IK seeds per point), `--sampler {random,sobol,halton,lhs,uniform,gaussian}`,
`--seed`, `--batch-size`, `--joint-limits-scale`.

```bash
# Cartesian reachability with explicit bounds
embodichain analyze-workspace \
    --asset /path/to/panda.urdf --ee-link fr3_hand_tcp --joints "fr3_joint[1-7]" \
    --mode cartesian_space --bounds -0.8 0.8 -0.8 0.8 0.0 1.5 \
    --ik-samples-per-point 5 --num-samples 50000
```

## Caching results

Analysis results are cached to disk using a readable
`robot name + parameters + hash` key. The name exposes the robot variant,
control part, mode, sampler, sample count, and seed; the short hash covers all
remaining inputs such as bounds and IK settings. Repeated runs with identical
inputs reuse the reachable workspace without recomputing.

| Argument | Description |
|----------|-------------|
| `--cache-dir PATH` | Cache root. Default: `~/.cache/embodichain_data/robot_workspace`. |
| `--no-cache` | Disable caching. |
| `--force-recompute` | Recompute even if a cached entry exists. |
| `--output PATH` | Export a copy of the results to a user path. |
| `--export-format {npz,pkl,json}` | Export format for `--output` (default: `npz`). |

After a run, the CLI prints the cache entry path, e.g.:

```
Results cached at: ~/.cache/embodichain_data/robot_workspace/urrobot__robot_type-ur5__part-arm__mode-joint_space__sampler-random__samples-20000__seed-42__4c0a3a3190d7
```

Each entry is a directory containing:

- `results.npz` - the workspace arrays: `workspace_points`, `reachable_points`,
  `all_points`, `joint_configurations`, `success_rates`, `reachability_mask`.
- `meta.json` - mode, sample counts, metrics, analysis time, and the input
  metadata used to compute the cache key.

### Previewing a cached workspace

To re-visualize an already-computed workspace without recomputing, pass
`--preview-cache` together with the corresponding `--robot` or `--asset`.
EmbodiChain loads the robot and cached workspace into the same simulation
window.

```bash
# By cache entry directory (the path printed after a run)
embodichain analyze-workspace \
    --robot ur --robot-params '{"robot_type":"ur5"}' \
    --preview-cache ~/.cache/embodichain_data/robot_workspace/<cache-name>

# By results.npz file directly
embodichain analyze-workspace \
    --robot franka_panda --preview-cache /path/to/results.npz

# By cache key (looked up under --cache-dir)
embodichain analyze-workspace \
    --robot franka_panda --preview-cache <cache-name>
```

Reachable points are shown green and unreachable points red (Cartesian/plane
modes); pass `--hide-unreachable` to show only the reachable points. Use
`--vis-type`, `--point-size`, etc. to control the rendering.

To preview the cached workspace and robot in a headless Viser browser instead
of the native window:

```bash
embodichain analyze-workspace \
    --robot franka_panda \
    --preview-cache <cache-name> \
    --viser
```

### Loading cached data from other applications

Other applications can load the cached reachable workspace directly with NumPy,
without re-running the analyzer:

```python
import json
import numpy as np
from pathlib import Path

entry = Path(
    "~/.cache/embodichain_data/robot_workspace/<cache-name>"
).expanduser()
data = np.load(entry / "results.npz")
meta = json.loads((entry / "meta.json").read_text())

reachable = data["reachable_points"]   # (M, 3) reachable Cartesian positions
configs = data["joint_configurations"] # (M, num_joints) IK solutions
print(meta["mode"], meta["num_reachable"], "/", meta["num_samples"])
```

For environment randomization, prefer the runtime API so sampled joint
configurations are converted to poses using each environment's current robot
base:

```python
from embodichain.lab.sim.workspace import RobotWorkspace

workspace = RobotWorkspace.from_cache(entry, device="cuda")
indices = workspace.sample_indices(16, strategy="voxel_uniform")
candidate_qpos = workspace.qpos[indices]
```

See [Runtime Workspace Sampling](runtime.md) for Robot and event-functor
integration.

To look up an entry by its inputs from Python, use the analyzer's cache key:

```python
from embodichain.lab.sim.workspace.caches import (
    ResultsCache, compute_cache_key,
)
# metadata = analyzer._build_cache_key_metadata(num_samples)  # same inputs
# key = compute_cache_key(metadata)
# results = ResultsCache(cache_dir).load(key)
```

Use `embodichain workspace-cache list` / `info` / `clean` / `size` to manage
the lower-level sampling-session caches.

## Visualization

After computation, the workspace is drawn in the simulation window (reachable
points green, unreachable red in Cartesian/plane modes). The window stays open
until `Ctrl+C`.

Use `--viser` to publish the robot and workspace to a browser while running the
simulation headlessly:

```bash
embodichain analyze-workspace \
    --robot ur --robot-params '{"robot_type":"ur5"}' \
    --mode joint_space --num-samples 20000 \
    --viser --viser-port 8080
```

Viser currently renders the workspace as a point cloud. Other `--vis-type`
values fall back to point-cloud rendering in Viser while retaining their
existing behavior in the native window.

| Argument | Description |
|----------|-------------|
| `--vis-type {point_cloud,voxel,sphere,axis}` | Visualization type (default: `point_cloud`). |
| `--point-size`, `--voxel-size` | Rendering sizes. |
| `--viser` | Open a headless browser visualization containing the robot and workspace. |
| `--viser-point-size` | Viser workspace point size in world units (default: `0.01`). |
| `--viser-host`, `--viser-port`, `--viser-fps` | Viser server and update settings. |
| `--hide-unreachable` | Show only reachable points in Cartesian/plane modes. |
| `--no-visualize` | Skip visualization. |
| `--headless` | Run without the native window; Viser remains available. |
| `--sim-device`, `--renderer`, `--width`, `--height` | Simulation/render settings. |
