# Rendering, attachment and startup diagnostics

Read this for physics/render synchronization, native-window/DLSS configuration
and readiness reporting. Return to the [simulation overview](simulation-system.md).

## Rendering does not advance physics

`SimulationManager.render_frame()` owns a read-only consumption phase after
physics, interval events or reset writes. An open native window, nonempty camera
groups, due simulation-time recording and due Viser captures share one state
publication in that phase. Empty camera groups alone are not consumers; an open
window remains a consumer even with `NewtonPhysicsCfg.sync_to_renderer=False`.
Viser decides whether a frame is due before invoking its publication callback.

`update()` renders after each substep by default. Gym uses
`update(render_final_step=False)` and enters `render_frame()` around observations
after interval events, so the final camera, recording and Viser reads share the
post-event state. Standalone callers combining those consumers can use the same
sequence. Reset completes all writes before a new frame with forced Viser capture.
An independent camera/capture call still publishes fresh state. No publication
cache survives a frame, so paused edits and direct writes between frames do not
need mutation counters. State edits inside a read-only frame require an explicit
`sync_render_state()` before further reads; that API always publishes.

The Newton backend step hook suppresses automatic DexSim publication during
manager-owned updates to avoid publishing twice. The manager honors explicit
`sync_to_renderer=True` even without consumers; `None`/`False` publish only for
consumers. Raw DexSim `World.update()` retains its configured automatic policy.
Render-thread recording only consumes already-published state and must never
call the blocking physics-to-render bridge. Marker publication while paused uses
`capture_visualization(force=True)` without advancing physics.

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

## Renderer and DLSS configuration

`cfg/simulation.py:RenderCfg.apply_to_dexsim_config()` translates rendering and
`DLSSCfg` into WorldConfig after automatic renderer resolution. Forward the
DLSS master switch even when false, and retain config during headless startup
for offscreen cameras or a later window. Native window/offscreen behavior can
differ; settings not authored by EmbodiChain retain DexSim defaults.

The actual camera/window owns output dimensions; compatibility target fields
must not resize it. Internal dimensions/upsample ratio affect FastRT/OfflineRT
windows; hybrid/offscreen sizes derive from output size and quality.
`frame_time_delta_ms` is render cadence, not physics/control cadence.
`gym/utils/gym_utils.py:config_to_cfg()` decodes nested DLSS mappings. Scalar
validation runs at construction and conversion after mutable edits. Configuration
tests do not prove GPU/NGX support, which initializes on a rendered frame.
Read exact renderer/quality defaults in the config source.

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
