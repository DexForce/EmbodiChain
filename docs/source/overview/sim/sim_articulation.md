# Articulation

```{currentmodule} embodichain.lab.sim
```

An {class}`~objects.Articulation` joins rigid links with joints. Use it for
mechanisms such as doors, drawers, and hinges. For a robot that needs named
control parts and part-specific kinematic solvers, use {doc}`sim_robot`.

## Configuration

Declare an {class}`~cfg.ArticulationCfg` with `sim.add_articulation(cfg)`.

| Field | Meaning |
| :--- | :--- |
| `fpath` or `source` | A local URDF path, or an authored USD articulation selected with `UsdSourceCfg`. |
| `uid`, `init_pos`, `init_rot` | Scene identity and initial arena-local placement. A homogeneous `init_local_pose` takes precedence over position and Euler angles. |
| `root_props` | Root settings such as fixed base and self-collision. Defaults are fixed base and disabled self-collision. |
| `asset_physics_mode` | `"preserve"` retains source link and joint physics. `"overlay"` applies explicit physical overrides. Root policy and scale are configured separately. |
| `init_qpos` | Initial positions in source-resolved DOF order. |
| `qpos_limits` | Position-limit overrides by joint name or regex, or as a `(dof, 2)` array. |
| `joint_drive_props` | Sparse overrides of source or backend joint drive properties. |
| `attrs`, `link_attrs` | Shared rigid-body properties and per-link overrides. |
| `build_pk_chain` | Defaults to `True` for FK and Jacobians. Set `False` to load a supported simulation topology that cannot be represented as a kinematic tree. |

## Setup & Initialization

Run this headless example from the repository root. The bundled hinge URDF
contains its link geometry and joint limits. An explicit position drive
commands the hinge from its measured state.

```python
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import ArticulationCfg, JointDrivePropertiesCfg, RenderCfg

sim = SimulationManager(
    SimulationManagerCfg(
        headless=True,
        device="cpu",
        render_cfg=RenderCfg(renderer="no-render"),
    )
)
try:
    articulation = sim.add_articulation(
        ArticulationCfg(
            uid="hinged_arm",
            fpath="examples/sim/sensors/state/hinge.urdf",
            init_pos=(0.0, 0.0, 0.3),
            asset_physics_mode="overlay",
            joint_drive_props=JointDrivePropertiesCfg(
                drive_type="force",
                target_mode="position",
                stiffness=40.0,
                damping=4.0,
            ),
        )
    )
    sim.prepare()
    qpos = articulation.get_qpos()
    target = qpos + 0.2  # radians for the revolute joint
    articulation.set_qpos(target, target=True)
    sim.update(step=1, render_final_step=False)
finally:
    sim.destroy(exit_process=False)
    SimulationManager.flush_cleanup_queue()
```

### USD Import

Set `source_prim_path` to choose one articulation from the USD source. A
scene can contain several jointed mechanisms, so select the exact prim.

```python
from embodichain.lab.sim.cfg import ArticulationCfg, UsdSourceCfg

door_cfg = ArticulationCfg(
    uid="door",
    source=UsdSourceCfg(file_path="workcell.usda", source_prim_path="/World/Door"),
    asset_physics_mode="preserve",
)
```

Declare constrained USD assets before preparation on the Default backend.
Use `sim.add_usd()` to import a scene with constraints between distinct
objects or between an object and the world. See {doc}`sim_assets` for scene
import and placement guidance.

## Articulation Class

`E = articulation.num_instances`, `J = articulation.dof`, and
`L = len(articulation.link_names)`. Joint tensors follow `joint_names`.
Link tensors follow `link_names`.

| Read | Shape | Meaning |
| :--- | :--- | :--- |
| `get_local_pose()` | `(E, 7)` | Root pose relative to the environment origin, in `xyz + xyzw` order. |
| `get_link_pose(link_name)` | `(E, 7)` | Pose of one named link in the local arena. Use `to_matrix=True` for homogeneous matrices. |
| `get_qpos()`, `get_qvel()` | `(E, J)` | Current joint positions and velocities. Pass `target=True` to read the commanded targets. |
| `body_data.mass` | `(E, L)` | Mass for every link. |
| `get_qpos_limits()` | `(E, J, 2)` | Effective lower and upper position limits. |

Clone state views that you need to keep across subsequent reads.

### Joint Control

Use `set_qpos(..., target=True)` and `set_qvel(..., target=True)` for motion
through the configured drives. Step the simulation after writing the
targets so the solver can act on them. With `target=False`, the same methods
write the current state directly. Use `set_qf()` to apply generalized
joint effort where supported.

### Combined state writes

Use `Articulation.set_state()` (also inherited by `Robot`) to write selected
root and joint state in one operation. Every supplied joint tensor shares the
selected environment and joint order.

```python
env_ids = [0]
joint_ids = [0]
qpos = articulation.get_qpos()[env_ids][:, joint_ids].clone()
articulation.set_state(
    env_ids=env_ids,
    joint_ids=joint_ids,
    qpos=qpos,
    target_qpos=qpos,
)
```

Optional fields include `qvel`, `target_qvel`, `qf`, `root_pose`, and
`root_velocity`. With `clear_dynamics=True`, the write clears velocities,
forces, external wrenches, and supported solver history, and holds the final
positions where no target is supplied. Explicit fields override these
reset defaults. Newton/MJWarp requires all articulations in each affected
solver world when clearing solver history. Use
`sim.reset_objects_state(env_ids=...)` for an episode reset across the scene.

### Root velocity writes

`set_root_velocity(velocity, env_ids=...)` accepts world-frame root-link
velocity, with shape `(selected_envs, 6)`. Put linear velocity in m/s first
and angular velocity in rad/s second. The input rows follow `env_ids`.

### Pose Control

Use `set_local_pose(pose, env_ids=...)` to reposition the root. Pass
`(selected_envs, 7)` pose vectors in `xyz + xyzw` order or
`(selected_envs, 4, 4)` matrices. Named link poses can then be read through
`get_link_pose()`.

### State Reset

`articulation.reset(env_ids=...)` restores the selected instances' initial
poses, joint state, and captured properties. In code that owns the entire
episode, call `sim.reset_objects_state(env_ids=...)` to coordinate the reset
with the engine and the rest of the scene.

## Physics and Drives

Fixed-base and self-collision settings are portable. The Default backend
also accepts root sleep and solver iteration settings in `root_props`.

### Per-link physics (`link_attrs`)

In overlay mode, `attrs` applies rigid-body properties to every link.
`link_attrs` applies a sparse {class}`~cfg.LinkPhysicsOverrideCfg` on top of
that baseline. Each group matches source link names by regex. Assign each
link to at most one group.

```python
from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    LinkPhysicsOverrideCfg,
    RigidBodyMaterialCfg,
    RigidBodyPhysicsCfg,
)

cfg = ArticulationCfg(
    uid="hinged_arm",
    fpath="examples/sim/sensors/state/hinge.urdf",
    asset_physics_mode="overlay",
    link_attrs={
        "contact_link": LinkPhysicsOverrideCfg(
            link_names_expr=["arm"],
            attrs=RigidBodyPhysicsCfg(
                material_props=RigidBodyMaterialCfg(static_friction=0.8),
            ),
        ),
    },
)
```

Use `set_link_physical_attr()` and `get_link_physical_attr()` for supported
runtime reads and partial overrides.

### Source mass properties

With URDF sources, `MassPropertiesCfg.recompute_inertia=True` allows the
collision geometry to replace an authored inertia tensor. By default, an
overlay that changes a drive or other unrelated field preserves valid source
mass, inertia, and COM. A positive mass override can retain the authored
tensor. To use density as an override when a valid source tensor exists,
enable recomputation. If the source inertia is invalid and collision geometry
is available, both backends can derive a fallback tensor.

### Reading current mass properties

`articulation.body_data.read_physical_properties()` returns mass, principal
inertia, and local center-of-mass pose in the public link order. Their
shapes are `(E, L)`, `(E, L, 3)`, and `(E, L, 7)`. Clone the returned
buffers for a snapshot across reads.

`set_inertia()` keeps the current principal frame. `set_com_pose()` keeps
the principal moments while changing their orientation and COM position.
The corresponding body-frame inertia matrix is `R diag(I) Rᵀ`.
`get_newton_link_properties()` exposes that matrix and the COM position.
Reset restores the captured inertia and COM frame for the selected instances.

### Drive Configuration

Use {class}`~cfg.JointDrivePropertiesCfg` in overlay mode to configure gains,
limits, friction, and targets. Omitted fields preserve their source or
backend values.

| Field | Meaning |
| :--- | :--- |
| `stiffness`, `damping` | Position and velocity gains. For force drives, units follow linear force or rotary torque. |
| `max_effort`, `max_velocity` | Authored effort and speed limits. Enforcement depends on the solver. |
| `friction`, `armature` | Passive friction and added joint-space inertia, subject to solver support. |
| `drive_type` | `"force"`, `"acceleration"` (Default only), or `"none"`. |
| `target_mode` | `"none"` / 0, `"position"` / 1, `"velocity"` / 2, `"position_velocity"` / 3, or `"effort"` / 4. |

Without an explicit target mode, an active force or acceleration drive uses
position and velocity targets. Default implements target modes through
effective gains. Newton uses native modes where supported and gain-based
fallbacks elsewhere. Non-MuJoCo Newton position mode assumes a zero velocity
target.

Use `get_joint_drive()` to inspect stiffness, damping, effort limit,
velocity limit, friction, and armature in that order. Each tensor has shape
`(E, J)` for a full read. `get_joint_drive_type()` reports the resolved
drive types. Use `set_joint_drive()` for supported runtime updates and
validate the motion response on the solver you selected.

### Joint Position Limits

Set `qpos_limits` on the configuration to replace source limits before
preparation. A dictionary selects joints by name or regex, for example
`qpos_limits={"hinge": [-1.2, 1.2]}`. An array supplies one row per joint
in source-resolved DOF order. After preparation, read the effective limits
with `get_qpos_limits()` or change supported limits with
`set_qpos_limits()`. `set_qpos()` clips to effective limits by default.

Use `SolverCfg.user_qpos_limits` to constrain planning or IK inside the
physical limits. See {doc}`sim_robot` for part-specific limits.

## Kinematics

With `build_pk_chain=True`, `compute_fk()` evaluates the configured
kinematic tree at the supplied joint positions. For example, on the
prepared hinge above, compute the `arm` transform relative to `base`.

```python
transform = articulation.compute_fk(
    qpos=articulation.get_qpos(),
    root_link_name="base",
    end_link_name="arm",
)
```

For full-chain FK, pass `qpos_joint_names` when the input tensor follows a
named order that differs from the chain. For measured simulation link poses,
call `get_link_pose()`.

### Atomic Skill Geometry Adapter

`get_link_vert_face()`, `compute_fk()`, and `get_parent_joint_chain()`
expose link geometry and kinematics. For sampled affordance geometry, use
`sample_initial_articulation_geometry()` from
`embodichain.lab.sim.atomic_skills`. The adapter returns a typed result.
Convert it with `to_object_geometry()` at the `ObjectSemantics.geometry`
boundary.

### Visual Appearance

Set `link_visuals` by exact source link name to customize appearance at
declaration. For a prepared scene with a native renderer,
`get_visual_material_inst()` returns one dictionary per environment,
keyed by link name. Use `get_existing_visual_material()` for per-segment
edits and `restore_visual_material()` to restore the authored assignments.
`reset()` also restores those assignments for the selected environments.

## Related Topics

- {doc}`sim_robot` for named control parts, robot presets, FK, and IK.
- {doc}`Robot Articulation tutorial </tutorial/robot_articulation>` for a
  complete rendered example.
- {doc}`sim_manager/physics/newton/runtime_qualification` for runtime and
  solver-specific support.
