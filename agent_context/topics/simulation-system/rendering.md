# Rendering, attachment and startup diagnostics

Read this for physics/render synchronization, native-window/offscreen image
processing and readiness reporting. Return to the
[simulation overview](simulation-system.md).

## NoRender initialization

`RenderCfg(renderer="no-render")` selects `Renderer.NORENDER` and requires
headless mode. Task environments preflight native render demand before World
construction: enabled Camera/StereoCamera subclasses, built-in recording or
nonempty validation cameras, the original window intent, Viser, forced Newton
render synchronization, and custom environment demand hooks reserve
rendering. Existing sensor/visual filters run before preflight; disabled dataset
managers do not contribute demand. Contact-only environments do not require it.
Camera demand includes `RobotCfg.sensors` from the selected backend preset,
even when the independent Gym `sensor` list is empty. This selection and
disabled-sensor filtering run before the World is constructed.

`BaseEnv._requires_native_renderer()` is the extension hook. Its result reaches
`SimulationManager(..., requires_native_renderer=...)` before Gym temporarily
sets headless for scene setup. With auto, no global default override and demand
explicitly false, the manager selects NoRender without probing the GPU. Explicit
renderer selections and global defaults retain precedence. The default demand
`None` preserves standalone GPU selection. The run-task launcher scopes
its native auto preference through `utility/render_utils.py`; this preference is
consulted only after NoRender demand selection and explicit global overrides,
and is restored on launcher exit. Custom camera/visual code, including
cameras added after startup, must reserve rendering before construction; no
runtime renderer switching or speculative constructors are used.

DexSim keeps its process renderer after World cleanup. Subsequent auto Worlds
reuse an initialized native renderer, including camera-free Worlds. A process
initialized with NoRender cannot later create native render consumers: the
manager rejects that combination before constructing another World and asks
for a new process with rendering reserved from its first environment. Process
configuration, rather than the per-World requested renderer, owns this check.

Native-renderer scene-node reservations may exceed NoRender's 131,071-node
limit. The manager caps the effective NoRender World capacity at that limit
while preserving the authored native-renderer reservation. This also applies
to explicit NoRender selections and global defaults. Smaller reservations are
retained, and physical replication must still fit the effective capacity.

The manager skips native background, light and visual-material setup, keeps
physical ground, and skips Newton render-state publication. Gym additionally
skips configured lights/IBL and built-in native visual effects; background
objects and collision geometry remain. DexSim uses its existing package and
device-free NoRender engine. Native camera and window operations require a
native renderer. Viser retains native rendering because its current exporter
reads visual meshes. Checkpoint evaluation with `--viewer` defaults to Hybrid
when the saved training configuration uses NoRender.

## Default scene presentation

`SimulationManager.set_default_background()` authors the native ground material
before Spawn declares the plane, using the bundled `PlaneDark` color and
roughness textures. Viser owns its separate browser grid and does not mirror
native lights or PBR materials.

`set_default_global_lighting()` initializes only environment emission; it does
not create direct lights or change existing scene lights. Standalone callers
declare direct lights through `SimulationManager.add_light()`. Task deployments
own them through `light.direct`; `light.indirect` can override the initial
environment emission or select IBL. Each explicitly declared light controls
its own shadows through `LightCfg.enable_shadow`. NoRender skips native visual
setup while retaining physical ground.

`RenderCfg` owns final RGB tone mapping and fixed exposure for native windows
and offscreen cameras. Tone mapping is disabled by default; explicit
`tone_mapping_enabled=True` enables the tone-mapping curve.

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
Default publishes native transforms with `World.update(0.0)`, without advancing
physics. Pose-editing tools and display initialization use this publication
boundary; they must not use positive-duration updates merely to refresh a view.

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

## Renderer, denoising and reconstruction configuration

`cfg/rendering.py:RenderCfg.apply_to_dexsim_config()` translates rendering,
`DenoisingCfg`, `DLSSCfg` and `NRDCfg` into WorldConfig after automatic renderer
resolution. `DenoisingCfg` independently selects the window and offscreen
pipelines from `off`, `optix`, `dlss` and `nrd`; `dlss` maps to DLSS Ray
Reconstruction and `nrd` maps to standalone NRD RELAX. Native SR/REBLUR
variants are not part of the EmbodiChain public contract. Retain the
complete configuration during headless startup for offscreen cameras or a later
window.

`RenderCfg` owns the general ray-tracing bounce limits, including the depth
at which Russian Roulette termination is permitted. `NRDCfg` separately owns
the indirect continuation budget for NRD; matching general bounce limits
does not imply matching light-transport budgets across denoising modes.

The actual camera/window owns output dimensions; compatibility target fields
must not resize it. Internal dimensions/upsample ratio affect FastRT/OfflineRT
windows; hybrid/offscreen sizes derive from output size and quality.
`frame_time_delta_ms` is render cadence, not physics/control cadence.
`gym/utils/gym_utils.py:config_to_cfg()` decodes all three nested mappings.
Scalar validation runs at construction and conversion after mutable edits.
Configuration tests do not prove GPU/NGX/NRD runtime support, which initializes
on a rendered frame. Read exact renderer and algorithm defaults in the config
source.

`VisualMaterialCfg` optical settings apply to direct material creation and
shape-owned materials. Spawn carries transmission in its material descriptor;
rigid, visual and deformable facades apply explicit volume parameters to every
native material when binding. The captured defaults retain these properties
through material reset; absent settings preserve imported native values.

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
`physics/newton.py` for render bridging, `cfg/rendering.py` for conversion,
`sensors/attachment.py` for parent lookup, and `_startup_summary.py` for reporting.
Focused tests: `tests/sim/test_sim_manager.py`, `tests/sim/test_startup_summary.py`,
`tests/sim/test_cfg.py`, `tests/sim/sensors/test_attachment.py`, and
`tests/gym/utils/test_gym_utils.py`.
