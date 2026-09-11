# Default Physics Backend

```{currentmodule} embodichain.lab.sim
```

The Default backend provides CPU and Direct GPU simulation through DexSim.
Its default solver is displayed as Constraint Dynamics.
Select it with {class}`~cfg.DefaultPhysicsCfg`, or `physics: default` in a
Gym configuration. See {doc}`index` for the shared capability matrix,
device selection, and time-step definitions.

## Minimal simulation

This example declares one falling cube, prepares the scene, and advances
100 physics steps. Set `device="cuda:0"` to use the Direct GPU path on a
compatible installation. CPU physics still uses the normal DexSim runtime;
rendering requirements are described in {doc}`/quick_start/install`.

```python
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import DefaultPhysicsCfg, RigidObjectCfg
from embodichain.lab.sim.shapes import CubeCfg


def run() -> None:
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            physics_cfg=DefaultPhysicsCfg(device="cpu", physics_dt=0.01),
        )
    )
    try:
        cube = sim.add_rigid_object(
            RigidObjectCfg(
                uid="cube",
                shape=CubeCfg(size=[0.1, 0.1, 0.1]),
                body_type="dynamic",
                init_pos=(0.0, 0.0, 0.5),
            )
        )
        sim.prepare()
        for _ in range(100):
            sim.update(step=1)
    finally:
        sim.destroy(exit_process=False)


try:
    run()
finally:
    # Flush after run() releases its local scene/object references.
    SimulationManager.flush_cleanup_queue()
```

To run this same scene with Newton, import `NewtonPhysicsCfg` and replace the
`physics_cfg` value with the configuration in {doc}`newton`. Asset
declaration, preparation, and explicit stepping retain the same structure.

## Scene parameters

| Parameter | Default | Meaning and tuning guidance |
| :--- | :--- | :--- |
| `physics_dt` | `0.01` | Physics-step duration in seconds; coordinate changes with the control period. |
| `device` | `"cpu"` | CPU execution; `"cuda:0"` selects Direct GPU execution. |
| `gravity` | `[0.0, 0.0, -9.81]` | World-frame acceleration in m/s². |
| `bounce_threshold` | `2.0` | Relative normal-speed threshold in m/s below which contacts do not bounce. |
| `enable_ccd` | `False` | Scene-level continuous collision detection; participating rigid bodies must also enable CCD. |
| `length_tolerance` | `0.05` | Representative scene length in metres, usually near the characteristic object size. |
| `speed_tolerance` | `0.25` | Representative scene speed in m/s; contributes to internal thresholds with the length scale. |
| `gpu_memory` | `GPUMemoryCfg()` | GPU capacities below; applies only to Default CUDA execution. |

Set the tolerance scales before constructing the manager. These represent the
scene's scale; increasing them is not a general accuracy or performance control.

The current integration uses Constraint Dynamics with PCM, disables enhanced
determinism, and evaluates friction on every solver iteration. These choices
are not selectable fields of `DefaultPhysicsCfg`. The startup summary reports
the runtime's selected solver as Constraint Dynamics or PGS. Do not copy an upstream
framework's `solver_type` setting into this configuration.

### CCD and articulation properties

For a body that needs continuous collision detection, set both
`DefaultPhysicsCfg(enable_ccd=True)` and
`attrs.rigid_props=DefaultRigidBodyPropertiesCfg(enable_ccd=True)` on its
rigid-body physics configuration. Validate the intended CPU/GPU path with a
fast-moving-body example; the scene switch alone does not opt every body in.

Articulation root properties are configured separately with
`ArticulationRootPropertiesCfg`. Root `min_position_iters` and
`min_velocity_iters` must be specified together; these and `sleep_threshold`
are Default-only. Root iteration settings apply before GPU buffer preparation.
They are distinct from per-link `DefaultRigidBodyPropertiesCfg` settings.
See {doc}`../../sim_articulation` for source preservation and joint drives.

## GPU memory capacities

Some Default GPU buffers have fixed capacities. Overflow can produce warnings,
missing contacts, or invalid simulation results. Size them against the largest
scene and worst-case randomized contact configuration, including all arenas.

| `gpu_memory` field | Default | Capacity / response to overflow |
| :--- | :--- | :--- |
| `temp_buffer_capacity` | `2**24` | Bytes of temporary pinned-host storage; increase for linear allocator overflow. |
| `max_rigid_contact_count` | `2**19` | Rigid contact records; increase for contact-buffer overflow. |
| `max_rigid_patch_count` | `2**18` | Contact patches; increase for patch-buffer overflow. |
| `heap_capacity` | `2**26` | Initial GPU and pinned-host heap bytes. |
| `found_lost_pairs_capacity` | `2**25` | Broad-phase found/lost pair records. |
| `found_lost_aggregate_pairs_capacity` | `2**10` | Found/lost aggregate-pair records. |
| `total_aggregate_pairs_capacity` | `2**10` | Aggregate-pair records in the GPU pipeline. |

These fields have no effect on Default CPU or Newton. Newton contact and
constraint budgets belong to its selected solver or collision pipeline, so
capacity values cannot be transferred directly between backends.

## Contacts and validation

`ContactSensor` works on Default CPU and Direct GPU. The two paths differ in
impulse and static-actor identity reporting; use the public query metadata and
`contact_capabilities` described in {doc}`../../sensors/contact_sensor`. Do not treat GPU and CPU
contact rows as an identical manifold representation.

For a new task, check settling, impact response, joint tracking, and contact
sensor output before increasing the environment count. Use
{doc}`migration` for symptom-based tuning and backend comparisons.
