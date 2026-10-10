# Visualization Markers Implementation Plan

Historical first-stage record; current environment batching, attachment and
delivery status is maintained in
[the follow-up plan](2026-09-18-markers-batch-attachment.md).

> **For agentic workers:** Use superpowers:subagent-driven-development. Track work in the ledger; do not commit independently.

**Goal:** Deliver issue #129's first working marker abstraction on DexSim and Viser.
**Architecture:** One validated marker group state produces detached mesh overlays. Existing manager and visualization runtime own lifecycle and publishing. DexSim's debug path owns sensor isolation.
**Tech Stack:** Python, NumPy, Viser, DexSim C++/pybind.
**Spec:** docs/superpowers/specs/2026-09-18-visualization-markers-design.md

## Global Constraints

- No physics stepping or Spawn finalization in marker operations.
- Public quaternions xyzw; protocol wxyz. Validate before mutation.
- Apache headers, configclass, annotations, public exports; Black 26.3.1.
- Use existing runtime/queues; no renderer thread shared mutations.
- Keep legacy draw_marker and point/trajectory APIs compatible.

## Task 1: Marker state and geometry

Files: visualization/markers/{cfg,group,_geometry,__init__}.py; tests/visualization/test_markers.py.
Produces MarkerPrototypeCfg, MarkerGroupCfg, MarkerGroup and snapshots using Task 2's MeshMarkerOverlay.
- [x] Add behavior tests for update/counts/validation/rotation/visibility and confirm missing implementation fails.
- [x] Implement detached state, geometry cache, callback-driven publishing and lifecycle.
- [x] Run focused tests.

## Task 2: Overlay transport and Viser

Files: visualization/protocol.py, backends/viser.py, scene_exporter.py, __init__.py; corresponding tests.
Produces MeshMarkerOverlay and SceneOverlays.meshes per spec.
- [x] Write snapshot/Viser update and environment visibility tests and verify failures.
- [x] Add protocol validation/copying/byte accounting; preserve meshes through exporter.
- [x] Render meshes with alpha and reusable handles; remove stale entries.
- [x] Run protocol/exporter/backend tests.

## Task 3: Native rendering and manager integration

Files: visualization/markers/_native.py, sim/sim_manager.py, sim/cfg/viewer.py; tests.
DexSim API dependency is independently investigated and implemented in an isolated sibling worktree.
- [x] Add manager lifecycle tests without physics.
- [x] Integrate add_marker_group, get/remove/clear and snapshots; implement native adapter using the verified engine API.
- [x] Preserve legacy handles and fix legacy arena name collisions.
- [x] Verify native creation/updates/cleanup and offscreen default isolation where runtime permits.

## Task 4: Delivery and review

- [x] Add runnable example and API coverage entries; update affected context.
- [x] Run focused CPU tests and renderer smoke checks, black, API/context gates.
- [x] Independent review; address actionable findings and rerun affected tests.
- [x] Report exact delivered scope, validation, engine dependency and remaining limitations.

## Delivery and validation (2026-09-18)

Implementation is retained uncommitted in two isolated worktrees:
- EmbodiChain: `codex/visualization-markers`, sibling `EmbodiChain-markers`.
- DexSim: `codex/debug-markers`, sibling `dexsim-markers`.

Both original main worktrees remain clean. Native support depends on rebuilding
and distributing the DexSim change exposing `Arena.create_debug_mesh`; an
unmodified DexSim 0.5.0 package does not expose this capability. The version pin
was not changed to an unpublished release. Viser works without that capability.

Validated with the `open` Python environment:
- Visualization and manager CPU coverage: 248 passed across two runs. Three
  unrelated existing COM descriptor tests fail against this environment's
  DexSim API (`RigidBodyPhysicsDesc.com_quaternion` is absent); the same failure
  was reproduced in unmodified main. They are not regressions from this change.
- Private native build succeeded; six DexSim tests passed across Hybrid,
  FastRT and OfflineRT, including sensor isolation, RGBA, pose/scale/visibility,
  actor removal, repeated private-material retirement and arena cleanup.
- EmbodiChain's real native integration test passed with `--run-gpu`, using the
  private native build, without modifying the installed package.
- API documentation gate: 2089/2089 exports documented.
- API documentation/context tests: 48 passed; context map check passed.
- Full-tree Black 26.3.1 formatting and diff whitespace checks passed.
- Headless Viser example ran through creation, animation, hide/show,
  clear/repopulate and clean shutdown. A real native seven-shape gallery was
  also rendered for visual inspection.
- Full Sphinx build was not verified: `sphinxcontrib.mermaid` is unavailable.

Independent review found and resolved two boundary defects: marker publication
must not finalize pending Spawn declarations, and removed native markers must
retire their private material registration. Added regressions failed before
those fixes and passed after. Numerical overflow/underflow is rejected before
state publication. The final scoped review had no remaining actionable findings.

Context decision: updated sim-visualization and DexSim rendering-system. Reviewed
simulation-system and robot-system overviews: their ownership, lifecycle and
quaternion guidance remains accurate; the viewer comment correction does not
change robot behavior, so these topics need no edit.

Deferred by design: GPU instancing, new streaming point-cloud machinery, arbitrary
USD import and text. Existing point-cloud/polyline and legacy axis APIs remain.
