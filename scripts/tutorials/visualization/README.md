<!--
Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
SPDX-License-Identifier: Apache-2.0
-->

# Viser visualization tutorial

`viser_scene.py` demonstrates EmbodiChain's Viser integration without
modifying DexSim. It runs a headless simulation and publishes:

- batched rigid-object meshes and poses;
- rigid-object groups;
- per-link articulation meshes and poses;
- low-frequency soft-body and cloth deformation;
- camera frustums and low-frequency RGB previews;
- read-only Gizmo frames or opt-in interactive transform controls;
- coordinate frame, target pose, trajectory, and sampled point-cloud overlays;
- environment and overlay visibility controls.

Run it locally:

```bash
python scripts/tutorials/visualization/viser_scene.py --port 8080
```

Open `http://127.0.0.1:8080`. The example runs until Ctrl+C. For an automated
smoke run, use:

```bash
python scripts/tutorials/visualization/viser_scene.py \
  --steps 10 --no-realtime --port 8080
```

All applicable simulation examples and tutorials now support the unified flag,
either directly or through their shared launcher. The dedicated
`viser_scene.py` tutorial remains a lower-level runtime example and uses its
own `--host`/`--port` options:

```bash
python scripts/tutorials/sim/create_scene.py --viser
python examples/sim/solvers/srs_solver.py --viser
python examples/sim/workspace/analyze_joint_workspace.py --viser
embodichain run-env --gym_config path/to/task.ur5.yaml --viser
```

Use `--viser-host`, `--viser-port`, `--viser-fps`, `--viser-image-fps`,
`--viser-soft-body-fps`, and `--viser-env-ids` to override the standard
server and sampling settings. `--viser` automatically runs the simulation
headlessly and enables trusted clients to drag configured Gizmos; `--headless`
by itself does not enable Viser.
`embodichain run-env --viser` captures camera previews after every environment
step by default; pass `--viser-image-fps` to restore wall-clock rate limiting.

For camera frustums and low-frequency RGB preview:

```bash
python scripts/tutorials/sim/create_sensor.py --viser
```

The **Cameras** panel selects an environment and camera independently from
scene visibility. It also provides switches for the selected camera frustum
and RGB preview. RGB rendering uses `VisualizationCfg.sensor_image_fps`
(2 FPS by default) and a latest-frame queue so image traffic cannot accumulate
latency.

The matching object tutorials can be launched directly:

```bash
python scripts/tutorials/sim/create_rigid_object_group.py --viser
python scripts/tutorials/sim/create_softbody.py --viser
python scripts/tutorials/sim/create_cloth.py --viser
```

The atomic-action tutorials receive the same options through
`tutorial_utils.py`. Gym tutorials pass the generated visualization
configuration into their environment configuration. Gizmo examples accept
`--viser`, which uses a headless simulation with browser-native transform
controls. Omit `--viser` to use the DexSim native window instead.
Application launchers only need to check `--headless` before calling
`open_window()`; the simulation manager safely skips the native window while
Viser is configured. It also rejects Viser startup while the native window is
already open.

Cloth uses its welded physical surface topology. DexSim does not currently
expose the PhysX soft-body collision topology, so the soft-body preview uses
a convex-hull surface over the live collision vertices. It follows deformation
but intentionally omits concave render-mesh details.

## Remote access

Keep Viser bound to loopback on a server and forward it through SSH:

```bash
ssh -N -L 8080:127.0.0.1:8080 user@worker-host
```

Then open `http://127.0.0.1:8080` locally. A deployed service should instead
place the worker port behind its authenticated gateway. Do not expose a worker
Viser port directly to the public internet.

## Current boundary

The runtime reports captured, published, dropped, and rejected frame counts,
along with capture/upload time and approximate payload bytes. Static geometry
is content-addressed and uploaded only when the scene manifest changes.
Simulation-managed topology revisions incrementally add, reuse, and remove
Viser mesh handles while preserving the server state used by reconnecting
clients.

Depth/mask preview, endpoint registration, and authenticated command handling
remain later work. Lights and rigid constraints do not own scene meshes.
Programmatic deployments with `allow_commands=False` export Gizmos as read-only
frames. Common `--viser` launchers use browser-native transform controls by
default. Commands are queued and applied by
`SimulationManager.update_gizmos()` on the simulation thread.
