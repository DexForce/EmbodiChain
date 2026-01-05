# Simulation Assets

Simulation assets in EmbodiChain are configured using Python dataclasses. This approach provides a structured and type-safe way to define properties for physics, materials, objects and sensors in the simulation environment.

## Physics Configuration

The `PhysicsCfg` class controls the global physics simulation parameters.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `gravity` | `np.ndarray` | `[0, 0, -9.81]` | Gravity vector for the simulation environment. |
| `bounce_threshold` | `float` | `2.0` | The speed threshold below which collisions will not produce bounce effects. |
| `enable_pcm` | `bool` | `True` | Enable persistent contact manifold (PCM) for improved collision handling. |
| `enable_tgs` | `bool` | `True` | Enable temporal gauss-seidel (TGS) solver for better stability. |
| `enable_ccd` | `bool` | `False` | Enable continuous collision detection (CCD) for fast-moving objects. |
| `enable_enhanced_determinism` | `bool` | `False` | Enable enhanced determinism for consistent simulation results. |
| `enable_friction_every_iteration` | `bool` | `True` | Enable friction calculations at every solver iteration. |
| `length_tolerance` | `float` | `0.05` | The length tolerance for the simulation. Larger values increase speed. |
| `speed_tolerance` | `float` | `0.25` | The speed tolerance for the simulation. Larger values increase speed. |

## Materials

### Visual Materials

The `VisualMaterialCfg` class defines the visual appearance of objects using Physically Based Rendering (PBR) properties.

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
| `material_type` | `str` | `"BRDF"` | material type. |

## Objects

All objects inherit from `ObjectBaseCfg`, which provides common properties.

**Base Properties (`ObjectBaseCfg`)**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `uid` | `str` | `None` | Unique identifier. |
| `init_pos` | `tuple` | `(0.0, 0.0, 0.0)` | Position of the root in simulation world frame. |
| `init_rot` | `tuple` | `(0.0, 0.0, 0.0)` | Euler angles (in degrees) of the root. |
| `init_local_pose` | `np.ndarray` | `None` | 4x4 transformation matrix (overrides `init_pos` and `init_rot`). |

## Rigid Object

Configured via `RigidObjectCfg`.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `shape` | `ShapeCfg` | `ShapeCfg()` | Shape configuration (e.g., Mesh, Box). |
| `attrs` | `RigidBodyAttributesCfg` | `RigidBodyAttributesCfg()` | Physical attributes. |
| `body_type` | `Literal` | `"dynamic"` | "dynamic", "kinematic", or "static". |
| `max_convex_hull_num` | `int` | `1` | Max convex hulls for decomposition (CoACD). |
| `body_scale` | `tuple` | `(1.0, 1.0, 1.0)` | Scale of the rigid body. |

### Rigid Body Attributes

The `RigidBodyAttributesCfg` class defines physical properties for rigid bodies.

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

## Soft Object

Configured via `SoftObjectCfg`.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `voxel_attr` | `SoftbodyVoxelAttributesCfg` | `...` | Voxelization attributes. |
| `physical_attr` | `SoftbodyPhysicalAttributesCfg` | `...` | Physical attributes. |
| `shape` | `MeshCfg` | `MeshCfg()` | Mesh configuration. |

### Soft Body Attributes

Soft bodies require both voxelization and physical attributes.

**Voxel Attributes (`SoftbodyVoxelAttributesCfg`)**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `triangle_remesh_resolution` | `int` | `8` | Resolution to remesh the softbody mesh before building physx collision mesh. |
| `triangle_simplify_target` | `int` | `0` | Simplify mesh faces to target value. |
| `simulation_mesh_resolution` | `int` | `8` | Resolution to build simulation voxelize textra mesh. |
| `simulation_mesh_output_obj` | `bool` | `False` | Whether to output the simulation mesh as an obj file for debugging. |

**Physical Attributes (`SoftbodyPhysicalAttributesCfg`)**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `youngs` | `float` | `1e6` | Young's modulus (higher = stiffer). |
| `poissons` | `float` | `0.45` | Poisson's ratio (higher = closer to incompressible). |
| `dynamic_friction` | `float` | `0.0` | Dynamic friction coefficient. |
| `elasticity_damping` | `float` | `0.0` | Elasticity damping factor. |
| `material_model` | `SoftBodyMaterialModel` | `CO_ROTATIONAL` | Material constitutive model. |
| `enable_kinematic` | `bool` | `False` | If True, (partially) kinematic behavior is enabled. |
| `enable_ccd` | `bool` | `False` | Enable continuous collision detection. |
| `enable_self_collision` | `bool` | `False` | Enable self-collision handling. |
| `mass` | `float` | `-1.0` | Total mass. If negative, density is used. |
| `density` | `float` | `1000.0` | Material density in kg/m^3. |


### Articulations & Robots

Configured via `ArticulationCfg` and `RobotCfg` (which inherits from `ArticulationCfg`).

These configurations are typically loaded from URDF or MJCF files.

### Lights

Configured via `LightCfg`.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `light_type` | `Literal` | `"point"` | Type of light (currently only "point"). |
| `color` | `tuple` | `(1.0, 1.0, 1.0)` | RGB color. |
| `intensity` | `float` | `50.0` | Intensity in watts/m^2. |
| `radius` | `float` | `1e2` | Falloff radius. |

### Markers

Configured via `MarkerCfg` for debugging and visualization.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `marker_type` | `Literal` | `"axis"` | "axis", "line", or "point". |
| `axis_size` | `float` | `0.002` | Thickness of axis lines. |
| `axis_len` | `float` | `0.005` | Length of axis arms. |
| `line_color` | `list` | `[1, 1, 0, 1.0]` | RGBA color for lines. |

### Rigid Object Groups

`RigidObjectGroupCfg` allows initializing multiple rigid objects, potentially from a folder.
