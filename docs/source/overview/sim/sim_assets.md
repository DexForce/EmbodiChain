# Simulation Assets

```{currentmodule} embodichain.lab.sim
```

Simulation assets in EmbodiChain use typed Python configuration classes. A
configuration describes the source or geometry, initial placement, and relevant
physics and appearance settings. The object returned by the
{class}`SimulationManager` represents that asset at runtime. For a replicated
asset, one UID identifies the batch and the first dimension of its state
selects the environment.

## Choose an asset type

| Use case | Configuration | Manager method |
| :--- | :--- | :--- |
| {doc}`Rigid body <sim_rigid_object>` with collision | {class}`~cfg.RigidObjectCfg` | `add_rigid_object()` |
| {doc}`Group of independent rigid bodies <sim_rigid_object_group>` | {class}`~cfg.RigidObjectGroupCfg` | `add_rigid_object_group()` |
| {doc}`Jointed mechanism <sim_articulation>` such as a door or drawer | {class}`~cfg.ArticulationCfg` | `add_articulation()` |
| {doc}`Robot <sim_robot>` with control parts and kinematics | {class}`~cfg.RobotCfg` | `add_robot()` |
| {doc}`Visual object <sim_visual_object>` for scenery or props | {class}`~cfg.VisualObjectCfg` | `add_visual_object()` |
| {doc}`Soft solid <sim_volume_deformable_object>` or {doc}`cloth <sim_surface_deformable_object>` with nodal state | {class}`~cfg.VolumeDeformableObjectCfg`, {class}`~cfg.SurfaceDeformableObjectCfg` | `add_deformable_object()` |
| {doc}`Cable <sim_cable_object>` such as a rope, wire, or hose | {class}`~cfg.CableObjectCfg` | `add_cable_object()` |
| {doc}`Fluid <sim_fluid_object>` sampled from an initial fill | {class}`~cfg.fluid.FluidObjectCfg` | `add_fluid_object()` |
| {doc}`Direct light <sim_light>` in a rendered scene | {class}`~cfg.LightCfg` | `add_light()` |
| [Sensor](#sensors) for images, contacts, or other observations | {doc}`Sensor configuration <sensors/index>` | `add_sensor()` |

The manager provides `get_asset(uid)` for lookup by runtime UID. Lights and
sensors have their own configuration and interfaces alongside scene objects.
See {doc}`sim_manager/physics/index` and {doc}`sensors/backend_support` for
the backend and device requirements.

## Declare, prepare, and step

Declare the initial assets through the manager, then call `sim.prepare()`
before reading physics state or joint metadata. Preparation binds physics
objects in place and publishes the initial render state. Use
`sim.update(step=1)` to advance a physics step.
Declare the initial physical topology before the first preparation. Later
additions or removals depend on the asset and backend.

This headless example declares one logical rigid object in two environments
and reads both poses after preparation.

```python
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import RenderCfg, RigidObjectCfg
from embodichain.lab.sim.shapes import CubeCfg

sim = SimulationManager(
    SimulationManagerCfg(
        headless=True,
        device="cpu",
        num_envs=2,
        render_cfg=RenderCfg(renderer="no-render"),
    )
)
try:
    cube = sim.add_rigid_object(
        RigidObjectCfg(
            uid="cube",
            shape=CubeCfg(size=[0.08, 0.08, 0.08]),
            init_pos=(0.0, 0.0, 0.2),
        )
    )
    sim.prepare()
    poses = cube.get_local_pose()  # (2, 7), ordered as xyz + xyzw
    sim.update(step=1, render_final_step=False)
finally:
    sim.destroy(exit_process=False)
    SimulationManager.flush_cleanup_queue()
```

### Read the scene inventory

The startup table counts registered UIDs under `Scene assets`. A batch of
replicated bodies contributes one logical asset, and a rigid object group
contributes one entry for the group. `Particle deformables` covers soft
solids and cloth. `Cable objects` and `Fluid objects` have their own rows.

`startup_summary="full"` adds asset UIDs and per-light type and scope to the
inventory. Sensor detail rows include the sensor class and, for cameras, the
resolution and enabled outputs. Select `startup_summary="off"` to disable the
EmbodiChain summary.

## Objects

Most individual object configurations inherit the placement fields in
{class}`~cfg.ObjectBaseCfg`. A rigid object group supplies placement on each
member configuration.

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `uid` | `None` | Set a unique identifier for the asset in its manager. |
| `init_pos` | `(0.0, 0.0, 0.0)` | Initial position in metres. Replicated procedural objects use the same arena-local placement in each environment. |
| `init_rot` | `(0.0, 0.0, 0.0)` | Initial Euler angles in degrees. |
| `init_local_pose` | `None` | Explicit 4×4 initial transform. It takes precedence over `init_pos` and `init_rot`. |

Pose vectors use `[x, y, z, qx, qy, qz, qw]`. Arena-local poses are relative
to each environment's origin. The world-frame state fields for particles and
cable segments already include their environment origins.

## Rigid Object

Use {class}`~cfg.RigidObjectCfg` for a single physical body. A dynamic body
responds to forces and contacts. A kinematic body follows commanded motion,
and a static body stays fixed.

| Parameter | Description |
| :--- | :--- |
| `shape` or `source` | Supply a procedural or file-backed shape, or select one USD body with `UsdSourceCfg`. |
| `attrs` | Physical properties grouped in `RigidBodyPhysicsCfg`. |
| `body_type` | `"dynamic"`, `"kinematic"`, or `"static"`. Procedural bodies default to dynamic. |
| `body_scale` | Procedural bodies default to `(1.0, 1.0, 1.0)`. USD bodies in preserve mode retain the source scale. |
| `shape.collision` | For mesh shapes, selects collision geometry such as convex hull, convex decomposition, triangle mesh, or SDF. The default is a single convex hull for procedural mesh colliders. |

Use `CompoundShapeCfg` when one rigid body needs several collision shapes.
Use a rigid object group when each piece needs its own pose and body state.

### Rigid Body Physics

{class}`~cfg.RigidBodyPhysicsCfg` keeps physical settings in optional groups.
An unset field leaves the source asset or backend default intact, which makes
the same configuration usable as either a complete procedural definition or a
sparse USD/URDF overlay.

| Group | Type | Contents |
| :--- | :--- | :--- |
| `mass_props` | `MassPropertiesCfg` | Mass, density, inertia, and COM pose. |
| `rigid_props` | `DefaultRigidBodyPropertiesCfg` | Rigid-body behavior such as damping, CCD, and solver iterations. |
| `collision_props` | `CollisionPropertiesCfg` | Collision enablement, contact/rest offsets, and concrete-backend contact properties. |
| `material_props` | `RigidBodyMaterialCfg` | Restitution, friction, and concrete-backend material properties. |

COM quaternions in configuration use `xyzw`. The Spawn adapter converts to the
native backend order only when it writes an engine descriptor.

Mesh cooking is configured through `MeshCfg.collision`. Its `approximation`
selects the collision representation, and fields such as `max_hulls` and
`sdf_resolution` are validated against that choice.

See {doc}`sim_rigid_object` for pose, collision, and dynamics operations, and
{doc}`Create Scene </tutorial/create_scene>` for a runnable example.

## Rigid Object Groups

{class}`~cfg.RigidObjectGroupCfg` builds a group from named rigid-object
configurations or initializes members from a directory using a template. Group
operations can address both environments and members. For example,
`group.get_local_pose()` returns `(environments, members, 7)`. See
{doc}`sim_rigid_object_group` for construction and selection examples.

## Articulations and robots

An `Articulation` represents links connected by joints. Use
{class}`~cfg.ArticulationCfg` for mechanisms that need link poses, joint state,
or drive targets. `Robot` extends this interface with named control parts and
solver integration. See {doc}`sim_articulation` and {doc}`sim_robot`.

## Visual objects

Use {class}`~cfg.VisualObjectCfg` for mesh scenery, primitive props, or Gaussian
splat visuals. A visual object has a pose and visibility. Choose a rigid
object if the same asset also needs a physics body and colliders. For a
physical object with custom appearance, use `RigidObjectCfg.visuals` or
`ArticulationCfg.link_visuals` to replace its rendered geometry while keeping
the physical model. See {doc}`simulation configuration </api_reference/embodichain/embodichain.lab.sim.cfg>`
for visual composition and source examples. See {doc}`sim_visual_object`
for setup, selected poses, visibility, and materials.

## Deformable objects

Volume, surface, and cable assets can change shape during simulation. Volume
and surface assets share a node-based state interface. Cables use a separate
object and configuration because their state includes the orientation and
angular velocity of each segment.

| Asset | Simulated geometry | World-frame state after `prepare()` |
| :--- | :--- | :--- |
| {doc}`Volume deformable <sim_volume_deformable_object>` | Tetrahedral volume mesh | `data.nodal_state_w` has shape `(environments, nodes, 6)` with position and velocity per node. |
| {doc}`Surface deformable <sim_surface_deformable_object>` | Triangle surface mesh | `data.nodal_state_w` uses the same node layout. |
| Cable | Ordered centreline represented by capsule segments | `data.segment_state_w` has shape `(environments, segments, 13)` with position, `xyzw` orientation, linear velocity, and angular velocity per segment. |

This terminology follows [Isaac Lab 3.0's overview of deformables](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/concepts/deformables.html).
The requirements below describe EmbodiChain.

### Volume and surface deformables

Both are declared with `add_deformable_object()` before preparation. They
require Newton on CUDA, with gradients disabled and a solver that supports
particle sets. Use `set_nodal_state()` to write world-frame positions and
velocities. Node indices refer to simulation particles. A separate render
mesh can have different vertices and indexing. See
{doc}`sim_manager/physics/newton/runtime_qualification` for solver support.

### Cable objects

Declare {class}`~cfg.CableObjectCfg` with ordered centreline points and a
radius through `sim.add_cable_object()`. Cables require the Newton backend
with DexUni on CUDA and gradients disabled. Declare them before `prepare()`.
They expose per-segment state and loads through `data.segment_state_w`,
`set_force()`, and `set_torque()`. For a full cable episode reset, call
`sim.reset_objects_state()` so the selected environments and attached
physical assets reset together.

In a Newton/DexUni scene, declare the cable and then prepare the scene.

```python
from embodichain.lab.sim.cfg import CableObjectCfg

cable = sim.add_cable_object(
    CableObjectCfg(
        uid="wire",
        points=[(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.2, 0.0, 0.0)],
        radius=0.005,
        init_pos=(0.0, 0.0, 0.25),
    )
)
sim.prepare()
state = cable.data.segment_state_w  # (environments, 2, 13)
```

The current cable reset path requires separate scenes for cables and fluid,
cloth, or soft-body particles. See {class}`~objects.cable.CableObject` for state
writes, loads, and reset behavior. See {doc}`sim_cable_object` for a complete
setup example, attachments, and state layouts.

## Fluid objects

Use {class}`~cfg.fluid.FluidObjectCfg` with `sim.add_fluid_object()` before
preparation. `FluidSpawnCfg` defines the initial particle sample from a volume
or explicit points. Density and particle spacing are shared across the scene's
fluids. Particle count is fixed after preparation.
`fluid.data.particle_state_w` returns world-frame positions and velocities
with shape `(environments, particles, 6)`.

Fluids require Newton on CUDA with DexUni's DFSPH component and gradients
disabled. Keep liquid and cloth or soft-body particles in separate scenes.
Use `renderer="no-render"` and `FluidRenderCfg(mode="none")` for physics-only
runs. Native liquid rendering requires `fast-rt` or `rt`. The
[fluid tutorial](https://github.com/dexforce/EmbodiChain/blob/main/scripts/tutorials/sim/create_fluid.py)
demonstrates setup, stepping, particle reads, and reset with a container.
See {doc}`sim_fluid_object` for source recipes and selected particle-state
writes.

## Import assets from USD

Use {class}`~cfg.UsdSourceCfg` in a supported object configuration to select
one object with `source_prim_path`. For a rigid object or articulation,
`asset_physics_mode="preserve"` keeps authored physics, and `"overlay"` lets
the configuration override selected properties. `VisualObjectCfg` uses the
selected object's appearance.

Use `sim.add_usd(UsdImportCfg(...))` for a scene or subtree. Set
`source.root_prim_path` to select the subtree. The import result exposes
object handles by their source keys and the manager registers them by runtime
UID. Call `sim.prepare()` before reading their state. The USD import recipes
section in {doc}`simulation configuration </api_reference/embodichain/embodichain.lab.sim.cfg>`
includes a complete example.

## Visual Materials

### Configuration

{class}`~material.VisualMaterialCfg` defines rendered appearance through
physically based rendering (PBR) properties. The defaults below apply when
no preset is selected. With `VisualMaterialCfg.from_preset(...)`, omitted
properties inherit their preset values.

| Parameter | Type | Default without a preset | Description |
| :--- | :--- | :--- | :--- |
| `uid` | `str \| None` | `None` | Optional shared material identity. The manager generates a name when omitted. |
| `base_color` | `list` | `[0.5, 0.5, 0.5, 1.0]` | Base color/diffuse color (RGBA). |
| `metallic` | `float` | `0.0` | Metallic factor (0.0 = dielectric, 1.0 = metallic). |
| `roughness` | `float` | `0.7` | Surface roughness (0.0 = smooth, 1.0 = rough). |
| `emissive` | `list` | `[0.0, 0.0, 0.0]` | Emissive color (RGB). |
| `emissive_intensity` | `float` | `1.0` | Emissive intensity multiplier. |
| `base_color_texture` | `str` | `None` | Path to base color texture map. |
| `metallic_texture` | `str` | `None` | Path to metallic map. |
| `roughness_texture` | `str` | `None` | Path to roughness map. |
| `normal_texture` | `str` | `None` | Path to normal map. |
| `ao_texture` | `str` | `None` | Path to ambient occlusion map. |
| `ior` | `float` | `1.5` | Index of refraction for ray tracing materials. |

### Visual Material and Visual Material Instance

A visual material is defined using the {class}`~material.VisualMaterialCfg` class. It is a material template that can create multiple instances with independent parameters.

A {class}`~material.VisualMaterialInst` can come from either of two sources:

- {meth}`~material.VisualMaterial.create_instance` creates a new dexsim material instance from an EmbodiChain material template.
- {meth}`~material.VisualMaterialInst.from_existing` wraps a dexsim material instance parsed from an asset without copying or replacing it.

When a rigid object, articulation, soft object, or cloth object is bound,
EmbodiChain inspects its render body and wraps any existing material. For an
instance without material, list-based assets return `None` and articulations
omit the link from the material dictionary. For render bodies with multiple
mesh segments, this API exposes the first valid material as the representative
instance. Rigid objects and articulations also provide
`get_existing_visual_material()` for per-segment access.

`get_existing_visual_material()` retains every original dexsim `MaterialInst` and creates a separate working `MaterialInst` from the first segment's existing material template. Randomizers can therefore modify and attach the working instance while keeping the original instance available for restoration. This creates an instance, not a new `VisualMaterial` template.

For batched simulation, `set_visual_material()` creates an instance per environment by default. Pass `shared=True` to reuse one instance across environments.

Rigid objects, articulations, soft objects, and cloth objects retain their
initial per-segment material assignments. Call `restore_visual_material()`
to restore them directly. Each asset also restores them for the selected
environments during its own `reset()`.

### Code

In a scene with a native renderer, declare the rigid object and call
`sim.prepare()` before changing its material instance.

```python
from embodichain.lab.sim import VisualMaterialCfg

obj = sim.add_rigid_object(object_cfg)
sim.prepare()

# Create a visual material with base color white and low roughness.
mat = sim.create_visual_material(
    cfg=VisualMaterialCfg(
        base_color=[1.0, 1.0, 1.0, 1.0],
        roughness=0.05,
    )
)

# Set the material to a rigid object.
obj.set_visual_material(mat)

# Get the material registered for each environment.
mat_inst = obj.get_visual_material_inst()

# Modify one instance without changing the other environments.
if mat_inst[0] is not None:
    mat_inst[0].set_base_color([1.0, 0.0, 0.0, 1.0])
```

To edit an authored material in place, use this instead of assigning a new
material after `sim.prepare()`.

```python
asset_inst = obj.get_visual_material_inst()[0]
if asset_inst is not None:
    asset_inst.set_roughness(0.4)
```

`VisualMaterialInst.set_base_color_texture()` accepts a file path, a tensor, or a pre-created dexsim `Texture`. Reusing a pre-created texture avoids uploading the same image on every material update.

See {doc}`sim_manager/rendering/materials` for presets, texture maps, and
advanced PBR parameters.

## Lights

Use {class}`~cfg.LightCfg` and `sim.add_light()` for direct lights. These
require a native renderer and work in headless scenes with offscreen rendering.
The manager configures environment emission and environment maps separately
through `set_emission_light()` and `set_indirect_lighting()`. See
{doc}`sim_light` for creation, global scope, and selected light updates.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `light_type` | `Literal` | `"point"` | Light type: ``"point"``, ``"sun"``, ``"direction"``, ``"spot"``, ``"rect"``, or ``"mesh"``. |
| `color` | `tuple` | `(1.0, 1.0, 1.0)` | RGB color of the light source. |
| `intensity` | `float` | `30.0` | Intensity of the light source in watts/m^2. |
| `enable_shadow` | `bool` | `True` | Whether the light casts shadows. |
| `radius` | `float` | `10.0` | Falloff radius (only for ``"point"``). |
| `direction` | `tuple` | `(0.0, 0.0, -1.0)` | Direction vector for directional types (``"sun"``, ``"direction"``, ``"spot"``, ``"rect"``, ``"mesh"``). |
| `spot_angle_inner` | `float` | `30.0` | Inner cone angle in degrees (only for ``"spot"``). |
| `spot_angle_outer` | `float` | `45.0` | Outer cone angle in degrees (only for ``"spot"``). |
| `rect_width` | `float` | `1.0` | Width of rectangular area light (only for ``"rect"``). |
| `rect_height` | `float` | `1.0` | Height of rectangular area light (only for ``"rect"``). |
| `mesh_path` | `str` | `""` | Asset path for mesh-based emissive lights (only for ``"mesh"``). |

```{note}
`"sun"` and `"direction"` each create one scene-wide instance and use
`Light.set_direction()` for orientation. Point, spot, rect, and mesh lights
are replicated per environment.
```

## Sensors

Declare cameras, stereo cameras, contact sensors, or registered sensor types
with `sim.add_sensor()`. Each sensor exposes observations through its own
interface. Cameras can be mounted on articulation links, with parent links
resolved during preparation. See {doc}`sensors/index` for configuration,
outputs, and attachment examples.

```{toctree}
:maxdepth: 1
:hidden:

sim_rigid_object.md
sim_rigid_object_group.md
sim_surface_deformable_object.md
sim_volume_deformable_object.md
sim_articulation.md
sim_robot.md
sim_visual_object.md
sim_cable_object.md
sim_fluid_object.md
sim_light.md
```
