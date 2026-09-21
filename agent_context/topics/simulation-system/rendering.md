# Rendering, attachment and startup diagnostics

Read this for physics/render synchronization, native-window/offscreen image
processing and readiness reporting. Return to the
[simulation overview](simulation-system.md).

## Rendering does not advance physics

`NewtonPhysicsCfg.sync_to_renderer=None` preserves DexSim's consumer-aware
per-step policy. Camera rendering and reset observations explicitly synchronize
on demand; correctness must not depend on continuous render sync in headless
training. `SimulationManager.sync_render_state()` and backend render hooks
publish state without advancing physics time. Marker publication while paused
uses `capture_visualization(force=True)`.

Viser forces headless mode and is mutually exclusive with the native window.
An empty server may start before assets are declared; it must not finalize
Spawn. Closed windows do not imply disabled offscreen cameras. Browser export
and native controls have their own owners:
[visualization](../sim-visualization/sim-visualization.md) and
[native gizmos](../sim-visualization/native-gizmos.md).

## Parented cameras

`sensors/attachment.py:resolve_parent_nodes()` accepts an unambiguous link name
or `<asset_uid>/<link_name>`, using the articulation's public render-node query.
The manager owns CameraGroup plus one native camera view per Arena; attachment
reparents those views and extrinsics remain local to the link. Lights have no
corresponding parent-attachment contract.

`prepare()` detaches cameras before rebuilt parents are destroyed, then
reattaches every configured camera once per committed topology revision.
Completion is marked only after all attachments succeed, so partial failures
retry. The camera registry is the sole attachment-intent store. Detailed
sensor behavior belongs to [sensors](../sensor-system/sensor-system.md).

## Renderer, denoising and reconstruction configuration

`cfg/simulation.py:RenderCfg.apply_to_dexsim_config()` translates rendering,
`DenoisingCfg`, `DLSSCfg` and `NRDCfg` into WorldConfig after automatic renderer
resolution. `DenoisingCfg` independently selects the window and offscreen
pipelines from `off`, `optix`, `dlss` and `nrd`; native RR/SR variants and NRD
method variants are not part of the EmbodiChain public contract. Retain the
complete configuration during headless startup for offscreen cameras or a later
window.

The actual camera/window owns output dimensions; compatibility target fields
must not resize it. Internal dimensions/upsample ratio affect FastRT/OfflineRT
windows; hybrid/offscreen sizes derive from output size and quality.
`frame_time_delta_ms` is render cadence, not physics/control cadence.
`gym/utils/gym_utils.py:config_to_cfg()` decodes all three nested mappings.
Scalar validation runs at construction and conversion after mutable edits.
Configuration tests do not prove GPU/NGX/NRD runtime support, which initializes
on a rendered frame. Read exact renderer and algorithm defaults in the config
source.

## Startup summaries

`sim/_startup_summary.py` collects read-only rows. Manager engine output follows
construction; one scene snapshot follows successful update/render/window-open
with a prepared scene. Pending Newton graph capture defers that snapshot until
an update resolves capture. `prepare()` itself does not print; diagnostics must
not step or finalize to manufacture readiness. `_ready_spawn_topology_revision`
is invalidated at preparation entry and published only after full success.

Gym defers manager output and emits one combined summary at initialization
completion. `startup_summary` controls verbosity; `dexsim_startup_info` controls
native startup information independently of Newton Warp-log suppression.
Warnings/errors remain visible. Device names reuse existing Warp metadata;
diagnostics must not initialize PyTorch CUDA. Requested/resolved Newton solver
and graph status are separate; pending must never be reported as captured.

## Change sites and validation

Use `sim_manager.py` for synchronization/readiness and window sequencing,
`physics/newton.py` for render bridging, `cfg/simulation.py` for conversion,
`sensors/attachment.py` for parent lookup, and `_startup_summary.py` for reporting.
Focused tests: `tests/sim/test_sim_manager.py`, `tests/sim/test_startup_summary.py`,
`tests/sim/test_cfg.py`, `tests/sim/sensors/test_attachment.py`, and
`tests/gym/utils/test_gym_utils.py`.
