# Sensor Backend Support

EmbodiChain selects physics with `physics: default` or `physics: newton` in Gym
configuration. The `default` backend runs through DexSim. DexUni is a solver
path within Newton. Select it with `solver_type="dexuni"` or use the default
Newton solver configuration. See {doc}`../sim_manager/physics/index` for
backend selection and the separate renderer configuration.

**Supported** means EmbodiChain provides the acquisition path for that
configuration. **Conditional** means availability or the requested output depends
on the solver, device, scene, or SDK. **Unsupported** means the acquisition mode
is unavailable on that backend or solver path.

| Sensor or acquisition mode | `default` | Newton (other solvers) | Newton (DexUni) | Requirements and output limits |
| :--- | :--- | :--- | :--- | :--- |
| {doc}`camera` and {doc}`stereo_camera` | Supported | Supported | Supported | Enable a DexSim renderer. Headless simulation can use offscreen rendering. |
| `JointEncoder` | Supported | Supported | Supported | Requires a robot or articulation that exposes joint position and velocity. |
| `FrameTransform` | Supported | Supported | Supported | Reads relative poses for physical assets or articulation links. |
| `PVA` / `IMU` (finite difference) | Supported | Supported | Supported | Estimates acceleration from completed state samples. Check `acceleration_valid` after initialization or reset. |
| `PVA` / `IMU` (native acceleration) | Unsupported | Conditional | Unsupported | Requires a dynamic rigid body on non-differentiable CUDA MuJoCo Warp with RNE enabled. `acceleration_source="auto"` falls back to finite differences when native acquisition is unavailable. |
| `JointWrench` | Unsupported | Conditional | Unsupported | Requires non-differentiable CUDA MuJoCo Warp with RNE and `NewtonPhysicsCfg.enable_joint_wrenches=True`. Check `wrench_valid` before reading the joint reaction. |
| {doc}`contact_sensor` | Supported | Conditional | Conditional | Requires a contact report from the active runtime. Available impulse components depend on the report. |
| `ContactResultant` | Conditional | Conditional | Conditional | Reports contact presence and separate normal and friction forces. Read each force with its validity flag. |
| `ContactGrid` | Supported | Conditional | Conditional | Bins contact counts and reported impulses. Use the component validity flags and the sample's completeness and impulse duration. |
| `TerrainScan` | Supported | Supported | Supported | Casts a grid of rays against supported geometry and reports distances, positions, and hit flags. |
| `RayLidar` | Supported | Supported | Supported | Casts geometric rays against supported static or moving targets. |
| `RayDepthCamera` | Supported | Supported | Supported | Reports axial depth and radial distance from geometric rays. The query can run with rendering disabled. |
| `VLP16` | Supported | Supported | Supported | The bundled `embodichain.extensions.sensors` extension provides the nominal 16-channel `RayLidar` preset. |
| `TactileIndentation` (geometric depth) | Supported | Supported | Supported | Uses probes on a planar, curved, or explicit grid. Targets require supported, closed, consistently wound geometry. |
| `TactileIndentation` (RGB) | Supported | Supported | Supported | Enable `rgb` to shade the geometric depth sample. The calculation can run with rendering disabled. |
| `TactileIndentation` (shear and marker displacement) | Supported | Supported | Supported | Enable `shear` to estimate normal load, shear, and marker displacement from sampled indentation and relative motion. |
| `TactileIndentation` (native hydroelastic surface) | Unsupported | Conditional | Unsupported | Enable `surface` with a DexSim SDK that exposes completed hydroelastic contact surfaces. Requires an active external Newton collision pipeline with `output_contact_surface=True`. |

## Contact report validity

On CPU, the `default` backend reports the magnitude of the total impulse.
A normal force requires a valid normal impulse. CPU MuJoCo does not supply
device contact buffers. DexUni can report both force-bearing and geometry-only
rows in the same scene. Check `contact_capabilities`, component validity flags,
and report completeness before consuming forces. See {doc}`contact_sensor`
for field semantics and {doc}`../sim_manager/physics/newton/runtime_qualification`
for Newton solver and device combinations.

Native hydroelastic output reports face counts and separation for the latest
completed contact surface. The runtime must report surface availability after
scene preparation. Check `surface_complete` before using projected surface data.

See the {doc}`/api_reference/public_api` for sensor classes and configuration
fields.
