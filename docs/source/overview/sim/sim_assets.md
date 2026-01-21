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
| `roughness` | `float` | `0.5` | Surface roughness (0.0 = smooth, 1.0 = rough). |
| `emissive` | `list` | `[0.0, 0.0, 0.0]` | Emissive color (RGB). |
| `emissive_intensity` | `float` | `1.0` | Emissive intensity multiplier. |
| `base_color_texture` | `str` | `None` | Path to base color texture map. |
| `metallic_texture` | `str` | `None` | Path to metallic map. |
| `roughness_texture` | `str` | `None` | Path to roughness map. |
| `normal_texture` | `str` | `None` | Path to normal map. |
| `ao_texture` | `str` | `None` | Path to ambient occlusion map. |
| `ior` | `float` | `1.5` | Index of refraction for ray tracing materials. |

### Visual Material and Visual Material Instance

A visual material is defined using the {class}`~material.VisualMaterialCfg` class. It is actually a material template that can be used to create multiple instances with different parameters.

A visual material instance is created from a visual material using the method {meth}`~material.VisualMaterial.create_instance()`. User can set different properties for each instance. For details API usage, please refer to the [VisualMaterialInst](https://dexforce.github.io/EmbodiChain/api_reference/embodichain/embodichain.lab.sim.html#embodichain.lab.sim.material.VisualMaterialInst) documentation.

For batch simualtion scenarios, when user set a material to a object (eg, a rigid object with `num_envs` instances), the material instance will be created for each simulation instance automatically. 

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

# Get all material instances created for this object in the simulation. If `num_envs` is N, there will be N instances.
mat_inst: List[VisualMaterialInst] = object.get_visual_material_inst()

# We can then modify the properties of each material instance separately.
mat_inst[0].set_base_color([1.0, 0.0, 0.0, 1.0])  
```


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

Configured via `LightCfg`.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `light_type` | `Literal` | `"point"` | Type of light (currently only "point"). |
| `color` | `tuple` | `(1.0, 1.0, 1.0)` | RGB color. |
| `intensity` | `float` | `50.0` | Intensity in watts/m^2. |
| `radius` | `float` | `1e2` | Falloff radius. |


```{toctree}
:maxdepth: 1

sim_soft_object.md
sim_articulation.md
sim_robot.md
```