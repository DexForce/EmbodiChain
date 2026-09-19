# Marker batching and attachment delivery

The user explicitly approved removing marker arena_index, defaulting to
SimulationManager.num_envs, selected env_ids updates, world scope, attachment,
and delivery of a DexSim batch-feature issue plus the existing native MR.

## Global Constraints

- Preserve legacy draw_marker and existing point/polyline APIs.
- No marker mutation may step physics, prepare/finalize Spawn, or force an
  unprepared physics synchronization. Pose reads and native mutation stay on
  the simulation thread; Viser uses detached snapshots on its existing worker.
- Public quaternions are xyzw; protocol boundary is wxyz.
- Validate before mutation; failed validation/publication preserves prior state.
- Native rendering uses the existing verified debug-mesh capability; its true
  native batch/instancing successor is tracked separately in the requested issue.
- Existing user approval is the design authority; no repeat approval gate.

## Task 1: Environment batches and attachment

Own markers/cfg.py, group.py, optional private marker helpers, sim_manager.py,
tests/visualization/test_markers.py, tests/sim/test_sim_manager.py and
tests/sim/test_marker_rendering.py. Do not edit docs/example/benchmark or DexSim.

Replace new MarkerGroupCfg.arena_index with scope='env'|'world', default 'env'.
Manager supplies all environment origins and num_envs; world groups have a
single world-space batch. Standalone groups default to one environment.
update accepts (E,M,3)/(E,M,4)/(E,M) arrays, with E=len(env_ids) when selected;
the existing (M,...) shorthand remains valid for a single selected environment.
env_ids must be unique, integral, in bounds; empty selection is a validated no-op.
No accidental broadcasting of ambiguous arrays. Prototype geometry/style is
shared; state is isolated by environment. A selected environment's count may
change without altering other environments (per-environment state is acceptable).
Expose count as total logical instances and counts per environment; document it.
Stable snapshot IDs include environment and marker index. Origins applied once.
Selective set_visibility and clear preserve untouched environments; remove
releases the entire group. New markers default visible unless their group/env
visibility says otherwise. Count changes reset omitted fields only for that env.

Add attach(parent: str, *, link_name: str|None=None, env_ids=None) and
detach(*, env_ids=None, keep_world_pose=True). Parent identifies registered
RigidObject, Robot or Articulation by uid; optional link_name selects a robot/
articulation link, absent means root. Existing marker poses become offsets in
the parent frame on attach. Pose composition handles rotations and origins.
Use an owner-supplied pose resolver querying existing public domain pose APIs,
not borrowed native nodes. Resolve live assets so topology rebuilds do not retain
stale native handles. Attachment requires prepared/bound targets but never
prepares them. Explicitly reject world-scope attach until a source-env selector
is specified; this avoids silently choosing an environment.
Auto-refresh attachment transforms after host simulation updates and explicit
render-state synchronization, before camera recording/capture. Pure export must
not mutate physics. Attach/detach updates publish immediately without stepping.
Detach preserves current world pose by default. Removing a target must not
dereference stale handles; detach affected groups before manager-owned deletion,
or use another explicitly tested safe policy with a clear observable error.

Tests first: default num_envs, origins once, full/selected update, selected count
changes and clear/hide, invalid/duplicate env_ids, atomicity, RGBA, rotations,
root/link attachment, motion following, detach, replacement/removal and no
implicit prepare/step. Preserve earlier native and Viser tests. Run targeted
tests in /home/dex/miniconda3/envs/open/bin/python. No commit until controller.

## Task 2: Examples, documentation, benchmark and verification

Update example for environment-shaped data and show selected-env update and
attachment. Update API/context docs and supersede earlier single-arena spec.
Add a focused marker benchmark following project benchmark skill: environment/
instance scale, full versus selected update/snapshot costs, CPU and GPU memory,
three-table Markdown report, clearly distinguish CPU snapshot work from actual
native/Viser rendering and transport. Run and retain measured evidence.
Run Black 26.3.1, API/context gates, docs build, focused CPU and native integration
tests. Resolve or independently track pre-existing COM descriptor test failures.

## Task 3: DexSim issue and MR

Create a Chinese feature issue in internal Engine/dexsim for native batch marker
creation/update/removal, per-env selections, shared prototypes, RGBA, sensor
isolation, scene/spawn ownership and existing-thread batch publication. Include
measurable benchmark and lifecycle acceptance; do not claim the current scalar
debug-mesh implementation completes it. Check duplicates and use existing labels.
Commit/push the verified existing native change and create MR against default
dev, following repository skill/template and worker/Runner gates. Link as a
prerequisite (Refs, not Closes) to the batch issue. Follow current-head pipeline
and report actual results or external blocker. Never self-merge or publish a
release. The user authorized issue/MR creation, not release admission.

## Task 4: Independent review and handoff

Review the complete final changes, resolve findings and rerun affected checks.
Report remote URLs, final validation, engine dependency and any residual scope.
Keep isolated worktrees and do not delete user work.

## Validation and delivery evidence

- Native batch issue created as DexSim #227; scalar debug-mesh MR is !1423,
  commit `fdd747e0a`. It is a prerequisite and does not close the batch issue.
- Environment batch/attachment implementation and selected-update tests complete.
- CPU marker/visualization/manager suite: 282 passed, three known pre-existing
  COM schema cases deselected. API coverage remains 2089/2089.
- Private native validation: two passed and one explicit dependency skip. The
  skipped registered-articulation attachment case is blocked before attachment:
  current DexSim `RigidBodyPhysicsDesc` removed `com_quaternion` and accepts a
  body-frame inertia tensor, while existing EmbodiChain Spawn code still uses
  the earlier descriptor. No descriptor shim was used to claim qualification.
- Root/link following, replacement, safe removal and no implicit prepare/step
  have focused manager-level coverage. Full native registered-asset qualification
  requires resolving the pre-existing physics schema mismatch independently.
- Four-environment headless Viser example completed creation, selected mutations,
  world-scope frame and cleanup.
- CPU benchmark (64 environments, 128 box markers each, three repeats): full
  state update 1.188 ms, selected 16-environment update 0.313 ms, detached mesh
  snapshot 198.331 ms. Before removing eager snapshot creation from validation,
  both update operations were approximately 200 ms. These are CPU measurements,
  not native renderer timings or GPU instancing claims. Snapshot publication
  remains proportional to total mesh instances; native batching is tracked above.

Physics compatibility follow-up must migrate COM/inertia as one contract:
convert principal moments/orientation to the body-frame tensor, adapt source
snapshots and USD lowering, and validate Default/Newton and nontrivial rotations.
Silently dropping `com_quaternion` would produce incorrect inertia, so it is not
an acceptable marker-scoped workaround. Actual affected code includes
`spawn/descriptors.py:_compile_rigid_physics`, `spawn/source.py` and `spawn/usd.py`.

No release was authorized or published. The native version requirement stays a
capability check until an engine release contains the MR; changing a pin to an
unpublished version would make installation fail. GPU instancing, arbitrary USD
marker import, text and new streaming point-cloud machinery remain separate
follow-up scope from this approved batch/attachment delivery.

Final review also corrected automatic attachment publication: it now updates
native overlays without forcing browser captures, skips unchanged parent poses,
and preserves the existing browser cadence. Simulation counters advance with
completed physics steps even when visual refresh fails; automatic failures are
logged and rolled back. Manual mutations still publish synchronously. Regression
tests cover capture counts across multiple groups and injected failures.

The Sphinx dummy build completed with 720 warnings; this is not a warning-free
documentation result. API coverage and 48 focused documentation/context checks
passed. An isolated temporary dependency directory supplied the missing Mermaid
extension without changing project runtime dependencies.

PR validation exposed a pre-existing architecture evidence checkout gap: the
snapshot pins a pre-squash commit outside current branch history. The docs test
workflow now explicitly fetches that exact revision when absent, preserving all
exact-source assertions and the committed snapshot. A fresh bare repository
reproduced absence and verified the fetch resolves the required source paths;
workflow actionlint passed. The final-head Sphinx repeat completed with 1023
warnings (the earlier run above reported 720).
