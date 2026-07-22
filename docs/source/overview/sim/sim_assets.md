# Simulation Assets

```{currentmodule} embodichain.lab.sim
```

Simulation assets in EmbodiChain are configured using Python dataclasses. This approach provides a structured and type-safe way to define properties for physics, materials and objects in the simulation environment. 

## Visual Materials

### Configuration

The {class}`~material.VisualMaterialCfg` class defines the visual appearance of objects using Physically Based Rendering (PBR) properties.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `uid` | `str` | `"default_mat"` | Unique identifier for the material. |
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

When a rigid object, articulation, soft object, or cloth object is constructed, EmbodiChain inspects its render body and automatically wraps an existing material. If no material exists, list-based assets return `None`, while an articulation omits that link from its material dictionary. For render bodies with multiple mesh segments, this compatibility API exposes the first valid material as the representative instance. Rigid objects and articulations additionally provide `get_existing_visual_material()` for per-segment access.

`get_existing_visual_material()` retains every original dexsim `MaterialInst` and creates a separate working `MaterialInst` from the first segment's existing material template. Randomizers can therefore modify and attach the working instance while keeping the original instance available for restoration. This creates an instance, not a new `VisualMaterial` template.

For batched simulation, `set_visual_material()` creates an instance per environment by default. Pass `shared=True` to reuse one instance across environments.

Rigid objects, articulations, soft objects, and cloth objects retain their construction-time per-segment material assignments. Call `restore_visual_material()` explicitly to restore them; each asset's `reset()` method performs the same restoration for the selected environments.

### Code 

```python
# Create a visual material with base color white and low roughness.
mat: VisualMaterial = sim.create_visual_material(
    cfg=VisualMaterialCfg(
        base_color=[1.0, 1.0, 1.0, 1.0],
        roughness=0.05,
    )
)

# Set the material to a rigid object.
object: RigidObject
object.set_visual_material(mat)

# Get the material registered for each environment.
mat_inst: list[VisualMaterialInst | None] = object.get_visual_material_inst()

# Modify one instance without changing the other environments.
if mat_inst[0] is not None:
    mat_inst[0].set_base_color([1.0, 0.0, 0.0, 1.0])
```

To modify a material already contained in a loaded asset, no replacement call is needed:

```python
object: RigidObject = sim.add_rigid_object(cfg=object_cfg)
asset_inst = object.get_visual_material_inst()[0]
if asset_inst is not None:
    asset_inst.set_roughness(0.4)
```

`VisualMaterialInst.set_base_color_texture()` accepts a file path, a tensor, or a pre-created dexsim `Texture`. Reusing a pre-created texture avoids uploading the same image on every material update.


## Objects

All objects inherit from {class}`~cfg.ObjectBaseCfg`, which provides common properties.

**Base Properties**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `uid` | `str` | `None` | Unique identifier. |
| `init_pos` | `tuple` | `(0.0, 0.0, 0.0)` | Position of the root in simulation world frame. |
| `init_rot` | `tuple` | `(0.0, 0.0, 0.0)` | Euler angles (in degrees) of the root. |
| `init_local_pose` | `np.ndarray` | `None` | 4x4 transformation matrix (overrides `init_pos` and `init_rot`). |

## Rigid Object

Configured via {class}`~cfg.RigidObjectCfg`.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `shape` | `ShapeCfg` | `ShapeCfg()` | Shape configuration (e.g., Mesh, Box). |
| `attrs` | `RigidBodyAttributesCfg` | `RigidBodyAttributesCfg()` | Physical attributes. |
| `body_type` | `Literal` | `"dynamic"` | "dynamic", "kinematic", or "static". |
| `max_convex_hull_num` | `int` | `1` | Max convex hulls for decomposition (CoACD). |
| `sdf_resolution` | `int` | `0` | Resolution for signed distance field. In most cases, a resolution of around 250 produces good results; resolutions exceeding 1000 are rarely necessary.|
| `body_scale` | `tuple` | `(1.0, 1.0, 1.0)` | Scale of the rigid body. |

### Rigid Body Attributes

The {class}`~cfg.RigidBodyAttributesCfg` class defines physical properties for rigid bodies.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `mass` | `float` | `1.0` | Mass in kg. Set to 0 to use density. |
| `density` | `float` | `1000.0` | Density in kg/m^3. |
| `angular_damping` | `float` | `0.7` | Angular damping coefficient. |
| `linear_damping` | `float` | `0.7` | Linear damping coefficient. |
| `max_depenetration_velocity` | `float` | `10.0` | Maximum depenetration velocity. |
| `sleep_threshold` | `float` | `0.001` | Threshold below which the body can go to sleep. |
| `enable_ccd` | `bool` | `False` | Enable continuous collision detection. |
| `contact_offset` | `float` | `0.002` | Contact offset for collision detection. |
| `rest_offset` | `float` | `0.001` | Rest offset for collision detection. |
| `enable_collision` | `bool` | `True` | Enable collision for the rigid body. |
| `restitution` | `float` | `0.0` | Restitution (bounciness) coefficient. |
| `dynamic_friction` | `float` | `0.5` | Dynamic friction coefficient. |
| `static_friction` | `float` | `0.5` | Static friction coefficient. |

For Rigid Object tutorial, please refer to the [Create Scene](https://dexforce.github.io/EmbodiChain/tutorial/create_scene.html) tutorial.

## Rigid Object Groups

{class}`~cfg.RigidObjectGroupCfg` allows initializing multiple rigid objects, potentially from a folder.


### Lights

Configured via `LightCfg`. Supports six light types matching the dexsim rendering backend.

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

.. attention::
    ``"sun"`` and ``"direction"`` are **global scene lights** (infinite-distance directional light
    sources). They are created as a single instance on the root environment, not batched per
    environment. Use :meth:`Light.set_direction` instead of :meth:`Light.set_local_pose` for
    these types.


```{toctree}
:maxdepth: 1

sim_rigid_object.md
sim_rigid_object_group.md
sim_cloth.md
sim_soft_object.md
sim_articulation.md
sim_robot.md
```
