# Generative simulation

Scene Engine turns images into editable scenes and portable exports; the
SimReady Asset Engine ingests and generates individual assets, including
articulated assets. Neither output is by itself a complete runnable Gym task
deployment.

## Entry points and resolution

Paths below are relative to `embodichain/gen_sim/` unless qualified.

| Request | Owner |
|---|---|
| Unified command dispatch | `embodichain/cli/main.py`: `scene-engine`, `scene-randomization`, `preview-scene`, `simready` |
| Generate/edit CLI modes | `scene_engine/cli/start.py` |
| Image → scene | `scene_engine/pipeline/generate.py`: `generate_scene_from_image()` |
| Edit portable scene | `scene_engine/pipeline/edit.py`: `edit_scene()` |
| Portable scene contract | `scene_engine/pipeline/utils/scene_exporter.py`, `scene_importer.py` |
| Grouped layouts from an existing export | `scene_engine/cli/randomization.py`, `scene_engine/randomization/generation.py` |
| Baseline cache, saved runs and pose playback | `scene_engine/randomization/baseline.py`, `persistence.py`, `loading.py` |
| Whole-scene USD package | `scene_engine/pipeline/utils/scene_usd.py`: `build_scene_usd()`, direct preview loading; `usd_scene.py`: entity index |
| General asset ingest | `asset_engine/pipeline/ingest.py`: `ingest_one_asset()` |
| Articulated asset generation | `asset_engine/clients/articulated_generation.py`, `asset_engine/utils/articulated_usdc_utils.py` |
| Generated task semantics | `embodichain/gen_sim/task_spec/` and `task_engine/task_spec.py`: compact TaskSpec generation and cache adapter |
| Robot profile selection | `task_engine/robot_profiles.py`: shared Robotiq hands, robot-specific arm assembly and mounting |
| Web app configuration | `gradio_ui/gradio_app.py`, `app_env.py` |
| Session-owned subprocesses | `gradio_ui/app_processes.py`: `SessionProcessRegistry` |

Generate: validate input → VLM understanding → Asset Engine geometry/articulation
generation → placement refinement → export. Segmentation and Asset Engine
clients are closed in `finally` blocks; VLM lifetime is separate.
Edit: import export → validate graph/typed edit plan → generate additions
→ change layout → overwrite export. Combined image/edit mode generates first.

Task Engine `prepare`/`run-all --robot-profile` selects `dual_franka`,
`dual_ur10` or `dual_tianji`; `tianji_marvin` remains a compatible selector alias
without changing existing serialized model IDs or frozen-reference receipts.
The latter two robots retain the canonical complete Robotiq hands. Marvin uses
its registered bare whole-body asset, not two copies of that robot.
Marvin's coincident solver roots are not shoulder locations: spatial hints use
the first controlled joint's fixed parent chain and origin. Its complete body
is placed at floor level from fixed-body collision geometry, outside the
measured table envelope; it must not inherit a bench-arm tabletop mount.
Valid mounting does not certify the task's workspace or continuous paths.
GenSim robot-mesh screens recognize the first moving link's own rigid mounting
ancestors across fixed solver frames. This is not a body-wide exemption: later
arm links, fixed siblings, the other arm and scene obstacles remain checked.
The Marvin/Robotiq deployment adds 0.01 kg m^2 armature to each arm joint for
simulation tracking stability with its low-inertia source links. Hand armature
stays zero; this is not a factory calibration or a shared Lab robot default.
Source-asset preflight precedes provider calls, while an explicit
physical reference owns its frozen model and arm anchors. Robot switching
requires a new bundle; `run --bundle` never replaces its embodiment. New-profile
TaskSpec envelopes seal robot assets as well as component declarations. Static
assembly/preflight is not collision, contact or complete-task qualification.
Official components resolve through `resolve_config_path`; concrete robot
presets import from `embodichain_robots`. Changes to endpoint names, motion
limits or validator declarations require bundle regeneration and new identity;
the bundle projection does not erase these differences to reuse older receipts.
Generated bundles explicitly seal `motion_contract=gen_sim.sampled_velocity/v1`
for fixed-time interpolation and preceding-waypoint IK seeds. The GenSim hooks
reuse the shared generator and numerical utilities, keep the existing URDF speed,
collision, release and stability gates, and do not claim the newer acceleration
budget. Explicit authored motion limits require `lab.motion_limits/v1`; bundles
without the contract keep that shared path. Native runs save `motion_contract.json`.
Generated MoveHeldObject, Pour and Place options select
`hand_source: grasp_command` to preserve drive authority under measured finger
deflection. Shared defaults remain `observed`; final open commands and physical
release gates are unchanged. These declarations participate in bundle identity
and require new full-task qualification, not reuse of old receipts.
UR10 ready/Park shoulder pan faces the table under the shared mount; its zero
pan pose faces away. `_task_program/solver_tolerances.py` preserves iterative
solver budgets and supplies explicit analytic-UR FK validation budgets without
requiring or mutating an iteration API on the shared solver.

Read [pipeline details](pipeline-details.md) for stage contracts, parser resume
behavior, Gradio artifact ownership and focused failure diagnosis.

## Durable scene boundary

The `scene_export/` directory contains `scene.json`, `scene_config.json`,
`scene_graph.json`, `mesh_assets/` and `articulated_assets/`.
Generation and editing also produce an optional `scene_usd/` delivery package
with a schema-v2 `scene.usda` carrying EmbodiChain entity metadata and a
relocatable `scene.usdz` single-file package. DexSim preview loads schema-v2
stages directly through `preview-scene --usd-file`; the manifest and native GLTF/USDC assets remain legacy
compatibility data. Build it only after the complete simulation scene is
prepared; validate UIDs and keep every packaged asset path inside the output
root.

glTF image-origin conversion is baked into the copied USD textures. Mesh UVs
remain authored as exported, and materials use ordinary USD texture nodes so
loading does not require development-only DexSim texture metadata or UV-transform
support. The packaged native glTF textures retain their original pixels.

- Scene object IDs and graph node IDs must be equal sets on import and export.
- Editable scene state is Y-up; portable runtime output is Z-up. Convert world
  pose at this boundary, and preserve `body_scale` separately from mesh geometry.
- Articulation export measures collision meshes relative to the native root
  rigid body, respecting USD up-axis, scale and deployment rotation, and sets
  root Z to leave 1 mm above the support. `proxy_init_pos` retains the independent
  GLB placement; import/edit must use it (legacy exports fall back to `init_pos`).
  An authored USD root translation is not a runtime rigid-body origin rebase.
- A generated scene includes a rigid table. Articulated assets need both a
  runtime USDC and a GLB proxy for editing.
- Overwrite validates/writes new scene state before deleting stale assets.
- The internal Scene Engine SimReady processor and general asset-ingest CLI
  are separate flows; change the selected owner rather than assuming one pipeline.

## Shared scene services

Geometric support detection, group translation/alignment and AABB separation
use private CPU strategies under `sim.scene`; tabletop and parent-surface
optimization adapt `sim.scene.layout`. These strategies reuse Torch geometry
kernels from `compute.geometry`, while Shapely, Trimesh and SLSQP stay on CPU. Existing adapters keep Y-up conversion and debug output;
`SceneLayoutConstructor` retains hierarchy traversal and edit-specific pose
propagation. `GravitySettler` owns its temporary simulator and fixed-step default;
optional convergence checks use shared scene validation. See
[scene services](../scene-services/scene-services.md) for frames, supported
geometry and physical-state boundaries.

GenSim owns scene understanding, asset generation, VLM orientation, its editable
scene graph and file formats. General services consume explicit UID/geometry
inputs rather than `SceneObject`, image masks or generation output directories.

Grouped randomization consumes one rigid-GLB table export and its explicit
support graph. The default SLSQP path retains the shared procedural solver;
the optional batched backend uses CPU/CUDA counter-addressed rejection sampling
from `compute.geometry._sampling`. Rejection has a different distribution and
acceptance rate from SLSQP repair. Batch boundaries do not change logical
proposal ordinals or the consumed attempt budget. CUDA selection is explicit.

The adapter owns group/member transforms, source/configuration snapshots and
canonical CPU float64 pose expansion for persisted identities. Version-3 policies
add explicit non-root XY/yaw targets and optional fixed group roots; empty targets
retain the original group path. `_intra.py` composes numeric postorder subtree
proposals with dynamic outer AABBs. UID-addressed streams retain the logical
prefix across batch sizes, and each failed layout consumes one whole attempt.
Hierarchical generator v2 samples outer roots from free center rectangles rather
than retrying collisions; version-1 hierarchical runs remain replayable.
Free-region temporaries use obstacle chunks and an adaptive batch ceiling;
resolved policies record the effective batch size without changing the prefix.

`_intra_geometry.py` adapts scaled assets and graph topology to the shared
parent-local support-grid strategy. Child bottom geometry guides plane selection;
ambiguous layers require evidence-checked overrides. Immutable content-addressed
caches include source/extractor/settings identities and leave the original
baseline unchanged. Cache v2 records the upper height deviation of the selected
layer so shallow bumps do not consume the support gap. Preserve earlier cache
height semantics when replaying old runs. Do mesh loading, plane extraction and polygon rasterization
only during preparation; do not call scene construction or SLSQP per candidate.

Batch candidates and loaded variants own lazy numeric sequences; consuming an
item constructs typed poses/configurations while saving validates numeric evidence.
Runtime collision settings are preserved. V3 layouts store root XY translations,
selected parent-relative full poses and cache references, then validate all source,
geometry, accounting and identities before exposing variants. Playback reconstructs
saved results without sampling or stepping physics. Existing v1 individual and
v2 packed group runs remain readable; complete resolved policies can be reused.

JSON remains the default storage; packed writes one atomic candidate container.
Both revalidate source/baseline and the exact cache snapshot before publishing
the completion manifest last. Concurrent cache writers use distinct staging files. Geometric
support/AABB acceptance does not establish physical or task success. See
[`scripts/tutorials/scene/README.md`](../../../scripts/tutorials/scene/README.md)
for policy settings, demo and end-to-end CPU/CUDA benchmarks and the
[public API supplement](../../../docs/source/api_reference/public_api.rst).

`randomization/adapter.py` projects rigid exports into the shared scene-layout
input. `generate_scene_from_image` and `edit_scene` accept an optional post-export
job; the CLI's combined generation/edit flow triggers it once after the final
export. New runs use source-independent v4 storage and retain the original
export for preview. `convert_layouts` preserves validated legacy poses and IDs
without sampling. File/model coordinate adaptation remains owned here.

## Runtime and extension boundaries

Keep parser capabilities idempotent so recorded stages support resume.
Provider configuration belongs to its loader/client, while coordinate and graph
contracts belong to importer/exporter. Task registration and physical/semantic
composition belong to [env-framework](../env-framework/env-framework.md).
TaskSpec owns the reusable normative task identity and evidence contract across
these boundaries; its schemas, predicate rules, and cache behavior are owned by
`embodichain/gen_sim/task_spec/` and detailed in the
[GenSim/TaskSpec design](../../../docs/architecture/gen-sim-taskspec-design.md).
Long-horizon specifications keep the terminal state in `goal` and represent
ordered intermediate achievements as acyclic `milestones`.
Interpretation checks explicit held-resource conflicts before grounding using
the existing bounded repair. Only `step_result` proves cross-step identity;
unknown scene aliases remain uncertain, not an identity or resource certificate.
The same two-attempt boundary rejects literal containment/repeated-destination
contradictions only when one current placement clause names its object and target.
Historical descriptions, prior-result objects and ambiguous repeated commands are
not locally reinterpreted; the guard never rewrites model output or task goals.
Completion evaluation requires trajectory evidence for those achievements and
their predecessor order; incomplete `always` windows cannot certify success.
Scene output is a scene-binding and measured-state input to that protocol; it
does not by itself certify a task witness.

Task Engine preparation derives the normative specification before binding,
then `SemanticTaskPlanner.plan_spec()` retains the original witness and emits
explicit expansion lineage. `semantic_graph_validation.py` checks every source
member, required resource, milestone dependency and surviving terminal clause.
The existing v38 graph/reader/executor and fingerprint rules remain unchanged;
new bundles add root-level `task_spec.json`, `semantic_graph_validation.json`
and offline `semantic_graph.html`. Transaction publication verifies companions
before relocating private paths, then rebuilds their full digests and root HTML;
readers never reseal stale artifacts. Missing legacy companions are not generated
implicitly. `validate-graph --bundle` is read-only, and `visualize --output *.html`
uses the same graph data rather than a separate execution representation.
`task_engine/semantic_graph_html.py` owns the offline report and its control-part
lanes. Optional bounded local thumbnails come from bundle-owned scene metadata;
source cutouts and generation-stage GLB renders are not final-geometry evidence.
Workflow and CLI completion write scoped monotonic wall time to `run_summary.json`
and refresh only the current run's selected viewer. This display-only sidecar
does not alter sealed companions, execution outcomes or TaskSpec qualification;
missing images and timing remain unavailable, without network or renderer calls.
Runtime captures supported final geometry before reset, separately from measured
milestone events. New-contract success requires a satisfied TaskSpec evaluation;
raw program success remains in `runtime_result.program_success`. Unsupported
observers, incomplete `all` selector universes and sparse temporal/invariant
windows remain unavailable, not certified. Historical reports are not rewritten.
Fresh runs also publish `task_group_results.json` beside the unchanged execution
report schema. It distinguishes failed entered calls, blocked descendants and
missing qualification; completed commands without a verified historical E
milestone or complete TaskSpec remain unverified. The summary is read-only and
does not change execution, state commits, or old receipts.
The isolated bundle runner uses one PyTorch intraop thread during a single CPU
episode or initial planning probe, restoring the caller's setting on exit.
Within that scope, GenSim Pick batches of at least 100 CPU PyTorch IK candidates
temporarily reuse the original thread budget, capped at eight, and restore the
single-thread setting afterward. Candidate poses, seeds and counts are unchanged.
GPU and multi-environment execution retain their existing thread policy. This
reduces small-tensor thread-pool overhead; it does not change physics, solver
tolerances, motion timing or recovery budgets.
Ordinary single-environment Place with preserved target orientation screens its
unchanged required keyframes with finite IK/FK before dense path planning.
An unqualified screen is not a proof of global unreachability; explicit height
caps, multi-waypoint goals and protected Cartesian/pour routes keep their prior
path. Owned adjacent Pick/Place calls share one 90-second cooperative wall ledger
across nested previews and recovery; measured rigid planning uses the same
threshold from `_task_program/_search_cost.py`. It rejects overdue planning at
instrumented boundaries, not by preempting a native operation, so elapsed time may exceed that
deadline until the next check. These search receipts neither prove acquisition
or release nor replace final TaskSpec qualification.
The normal coordinator derives call-local manipulation permissions from typed,
unconstrained E1 rigid-inside Pick/Place pairs through
`_task_program/policy_selection.py`. It selects measured preshape/inside geometry
only for sources geometrically enclosed by an articulation's measured bounds,
and later placements into the same receiver after such a transfer. Bounds request
planning qualification; they do not certify containment or contact. An earlier
operation on the same body invalidates this initial-geometry hint. Shared-receiver
selection follows dependency ancestors, never future suffixes. Ordinary tabletop
sources retain their existing preshape and transport. Heading-free rigid-inside
steps may select failed-real-held yaw recovery independently, without task IDs.
Reacquiring an E2 upright result retains its qualified revision-2 closing screen
instead of combining that screen with measured preshape. Explicit orientation,
axis/direction/layout, grasps, nonrigid receivers and other recipes are excluded.
Selection does not modify physical inputs; the existing single-environment native
Default IK qualification and complete physical gates remain required.
Direct bundle-builder callers retain explicit capability selection and legacy
behavior when declarations are omitted.
Explicit `world_yaw_free_places` names ordinary rigid-inside Place nodes whose
object heading may vary. It fingerprints with the task
constraints. Only failed actual-held calls try the existing bounded world-Z yaw
order; hypothetical Pick previews, explicit orientations and Cartesian routes
keep their original behavior. The qualified route retains 0.12 m release clearance,
0.05 m approach lift, the shared budget, landing-footprint screening and final
Place inspection. It does not claim continuous collision or physical success;
the ordinary runner and complete release/stability/TaskSpec gates remain required.
Explicit `measured_rigid_calls` opts named ordinary Pick or rigid-inside Place
occurrences into measured hand preshape and observed-geometry planning. It
fingerprints only selected bundles. Pick retains its
selected arm path and time grid while sizing the opening from native pads and
the complete target projection. Inside Place observes current rigid occupants
and the actual held footprint, retaining 5 mm floor/object planning reserves and
0.12 m release clearance; bounded higher approach/retract candidates must pass
native occupied-hand, operated-arm and cargo scene rest checks plus robot proxy
and full-robot velocity checks. These are discrete fixed-scene planning screens,
not continuous collision or physical-success certificates. Previews and legacy
calls retain their existing policy; the normal executor still owns acquisition,
retention, release, stability and TaskSpec acceptance. Convex-cover containment
retains every vertex/plane pair and its tolerance with bounded row-wise scratch.
Supported E1 placement allows up to 50 mm positive floor clearance while retaining
the 10 mm penetration limit. Full-footprint coverage, exact released-state receipts
and the other physical gates remain required; execution evidence records both limits.
Rigid-reference E1 `inside` observations qualify the full projected object
footprint against actual planar-floor triangles (including holes), resting height,
and exact measured terminal release receipts. Milestones read the recorder's
post-segment `end_steps` slots, independently of fresh final poses; later motion
cannot erase or falsely establish an earlier milestone. This uses the existing
runner and unchanged bundle declarations/fingerprints, not a second executor.
Curved/unmeasured floors and articulated reference parts remain unavailable;
the legacy planar landing-region certificate is not contact-force or arbitrary
3D-cavity qualification. Explicit `support_contract_revision=1` instead requires
native support samples within the same invocation's actual release phase and
observation boundary. The measured terminal RELEASE and separation, followed by
the complete post-action stability window, remain required. A continuously valid
sleeping patch retains its original sample time; no new impulse after terminal
acknowledgment is required. Command entry alone is not a release certificate.
An independent `pick_contact_revision=1` opts rigid Pick into a Default/B1 native
cooked-shape screen: complete named hand commands and actual rest offsets qualify
sampled nominal closure only until bilateral pad contact. It screens final TCP
rolls and emitted close commands, including after plan transforms; it neither
changes proposals/RNG nor replaces native acquisition/retention evidence. Both
markers default to absent for legacy equivalence and are independent of recipe
composition. Neither discrete screen proves continuous collision freedom.
`pick_contact_revision=2` additionally rejects nonpad native contact-halo entry
strictly before the sampled bilateral pad-rest boundary; potential halo overlap
at that boundary alone does not establish actual contact. Nonpad rest intrusion still
rejects at the boundary, and runtime bilateral acquisition/holding gates remain.
`pick_contact_revision=3` retains revision 2 and additionally screens the nominal
opening prefix against the exact rigid support of a bound ordinary on-Place.
Only source-qualified candidates enter this check. Native rest-envelope overlap
rejects; contact-halo proximity is diagnostic. This discrete geometric screen
does not replace measured release, separation, retreat or support evidence.
Explicit revision 4 combines revision 1's source rest-contact ordering with
revision 3's opening screen. The first nonpad contact halo before or at pad rest
remains diagnostic, not a collision certificate or standalone veto. Nonpad rest intrusion,
missing bilateral pad rest and all measured acquisition/release/support gates
still reject. Revisions 1/2/3 and the normal workflow's selection are unchanged.
Non-E1 and legacy bundles without TaskSpec retain their prior gates.

Explicit `upright_container_picks` maps selected E3 Pick node IDs to a maximum
pad-closing-axis tilt above the Z-up horizontal plane. It requires native Pick
closing geometry and is absent by default. Named cooked-pad geometry and nominal
FK define the axis; no universal TCP-X convention is assumed. The same mask runs
on candidates and emitted close FK, including after plan transforms. This grasp
quality rule is separate from native contact/rest safety and never replaces
acquisition or retention evidence. Undeclared Picks and shared GraspKit defaults
remain unchanged; regenerate consumers to opt in.

Explicit Pick contact revision 5 retains revision 4 and requires the native open
pads to be outside the target rest envelope. A call-owned proposal view may then
replace whole-object aperture fit with exact native open/close geometry, allowing
local rim or handle grasps without changing the shared generator. Revision 6 also
tries bounded grasp depths measured from the named pad span and pair rest offsets.
Original strategies remain first, accepted commands are cached unchanged, and
additional complete paths share the existing four-recovery budget. Extra depths
reuse the original geometric pools, branch keys and IK random states rather than
regenerating approach proposals or advancing unrelated random streams. Ordinary
Pick checks its already-required open-hand endpoint scene clearance before accepting a
depth; this introduces no new scene gate for Pour. Constraints and explicit grasps
retain their original depth. Handover end regions and owned Cartesian routes also
retain it unless the native8 source contract described below is explicitly enabled.
These opt-ins do not replace native acquisition, retention or final path checks.
Revision 7 also screens the fixed-arm nominal pre-contact closure against the scene
for ordinary native B1/env0 Picks, both during candidate choice and on emitted
plans. The read-only scene view permits intended source contact only in that
interval up to the first nominal bilateral rest sample; the open endpoint still includes the source and the native closing
screen owns every hand/source pair. Other objects, articulations, self/other-arm
checks, depth budgets and physical collisions remain unchanged. Post-contact
command FK is not a physical-penetration forecast: an acquired object can stop
actual closure before its command endpoint. Pour retains
revision 6 behavior. This option does not certify dynamic obstacle prediction.

Explicit Pick contact revision 8 retains revision 7's native geometry and runtime
gates, but its call-owned center-proposal view delegates the approximate source
proxy veto to the existing native cooked-hand/source qualification.
`pick_proposals.py` consumes GraspKit's cached `prepare_mesh()` contact pairs;
GenSim owns the resulting pose proposals, costs and truncation. Shared GraspKit
interfaces and defaults remain unchanged. These unqualified proposals cannot
bypass ground or end-region policies, and transformed candidates and emitted
closing commands must still pass the native screen. Non-native views and
revisions 1--7 retain their original proposal behavior. The screen records actual
proposal calls so configuration alone is not mistaken for an activated route.
Native center proposals retain their raw costs until the owning Pick applies
the current TCP transform and both roll variants. Native source/quality
qualification precedes the provider's configured candidate cap and shared IK;
unqualified proposals cannot consume every retained slot. Each existing depth
uses the same raw pool with its own transform. This does not add full-path
retries or change ground/end-region policies or the four-path recovery budget.
The unchanged inexpensive direction mask short-circuits native geometry queries
for already-incompatible columns; it adds no new acceptance condition.
Qualification follows the original stable cost order and stops when the
configured number of valid candidates is filled. This is the same qualified
top-k set as an exhaustive screen, without checking a provably higher-cost tail.
Ordinary B1 Picks without continuation previews or affordance branch sampling
qualify and solve candidates
in stable-cost batches of at most 50, stopping at the first IK-feasible batch
before invoking the existing full-path checks. The configured qualified-candidate
cap is still an upper bound, not a required population. Other callers retain the
exhaustive qualified top-k selector.
If the selected path or its full safety qualification fails, the original raw
pool and pre-IK random checkpoint restore that route's complete qualified-cap
population in place before the existing filtered recovery. This preserves
candidate coverage, not bitwise equivalence of random full-path planning.
Failed-path raw recovery qualifies individual
TCP rolls on demand in the existing cost/nearest-roll/route order; it does not
scan every raw route before applying the shared four-path budget. No native
source/scene check or execution gate is removed.
Revision 8 also checks the same native cooked hand against other rigid bodies'
actual pair rest offsets over the original precontact grid. Source contact
remains separately owned; the existing robot/articulation/proxy checks remain.
The scene snapshot is invocation-local and read-only. Emitted and transformed
closing plans repeat this qualification; it does not predict post-contact
dynamics or certify a physical grip. Revisions 1--7 retain their existing checks.

Per-call `pick_contact_revisions` also accepts revision 8 for generated E4
handover-source registered Picks. The lowerer's `ends` contract opts these calls
into the same open/precontact and emitted/transformed closing-scene checks.
Bounded depth search reuses the selected `supported_end` candidate pools and
shares the existing four-path budget, including owned Cartesian source routes;
it does not switch to ordinary center proposals or relax receiver selection.
Other registered calls, legacy E4 without this opt-in, Pour, and shared PickUp
defaults are unchanged. Geometry qualification still requires physical
acquisition, holding and subsequent HandOver verification during execution.
Regenerating an expanded E4 graph preserves its declared MoveHeldObject transfer
poses; upright landing refinement must not lower or rotate those transfer targets.

Explicit `post_handover_places` permissions bind only adjacent sequential
HandOver-to-relative-on Place calls with the same rigid object and exact receiver.
On real-held planning failure, the declared Place may reuse the existing
goal-conditioned waypoint branch with its observed `ends` grasp and preserved
release orientation. The normal Place attempt and successful plans remain unchanged.
For a singleton release goal with two Cartesian keyframes per translation, the
recovery distinguishes the generated free-transport midpoint from the original
preplace/contact/release targets. A fresh payload-aware cuRobo connection replaces
only that midpoint; continuous seeded IK retains the exact Cartesian descent and
reverse empty-hand retract. Explicit phase composition preserves passive and
release commands/effects, remaps phase-boundary monitors, and may extend only the
approach within the original action timeout. Final FK, velocity, robot mesh and
scene-envelope checks still qualify the emitted commands. Other waypoint branches
retain their fixed phase grids and fail closed when duration is insufficient.
The same explicit route also binds the preceding HandOver: an `ends` grip keeps
its inverse-grasp withdrawal before the final two waypoints move outward toward
the source arm's side by the existing retreat distance; a planned exit robot-mesh
gate participates in the existing bounded direction retry. Ordinary/default calls
remain unchanged. These planning receipts do not certify physical acquisition,
release or task success; complete native execution and TaskSpec remain required.

The packaged Task Engine CLI defaults select
`planning.runtime_profile: auto` and
`execution.acceptance_mode: experiment`, including when `--config` is omitted.
Explicit YAML overrides may select `default`/`strict` or
`composition_experiment`; omitted fields in override YAML and direct Python
configuration construction retain their legacy defaults. The CLI defaults and
both packaged experiment YAML files bound preparation to one worker, one scene
attempt and one action attempt. Experimental acceptance requires one environment;
vectorized legacy execution needs an explicit strict configuration.
Only the contact profile adds a material-only Robotiq pad
overlay (static/dynamic friction 2.85/2.70); composition retains original materials.
Automatic selection uses the validated recipe: any E3/Pour selects composition,
pure E1/E2/E6 recipes retain the original default grip policy, and other recipes
retain contact. It never tests task IDs, object UIDs or text-name heuristics.
The automatic ordinary policy still uses the authorized 100/100/30 mm supported
placement tolerances. Explicit profile overrides remain unchanged.
E3 auto-arm preference uses the existing source/receiver midpoint rule; explicit
arm requests and the owner of an already-held source keep precedence. This is
only a preference, not a reachability certificate; native continuation and
actual held-state planning still qualify the complete route.
Only a failed, single-environment retained E3 arc rejected by joint-velocity
qualification may reuse the existing continuous seeded IK resolver. It retains
every Cartesian target, the observed start, frame count and time grid, checks
the full emitted robot proxy, and keeps native retention gates. Owned E3 planning
shares the 90-second cooperative wall limit; successful and legacy Pour paths
remain unchanged. This does not certify arbitrary scene collision or fluid transfer.
The generated constraints bind the concrete profile; `runtime_profile_selection.json`
records the request and selection. Explicit profile overrides remain unchanged.
One bundle has one material policy, not dynamic per-call switching; mixed recipes
containing Pour retain composition and require their own physical qualification.
Both profiles use 100/100/30 mm COM/position/support tolerances
for supported on-placement, independently of final TaskSpec acceptance.
Generated handover contact revisions compose with existing occurrence permissions;
different revisions for the same call reject rather than replacing an earlier
selection. It derives horizontal end-grasp HandOver/Place
permissions from exact adjacent occurrences, not task IDs.
Authorized horizontal-transfer receiver Places retain ordinary 140-point free
transport even if the model groups them inside E4; they cannot inherit the
transfer group's Cartesian approach. HandOver and its source/staging stay
unchanged, and terminal descent/release/withdrawal checks remain required.
Related auto-arm Pour source/receiver reuse and terminal same-object reacquisition may add a
visible empty-arm clearance/restore pair. Its finite proximal-joint proposals
are ranked by FK endpoint separation, screened against the current scene and
rechecked on emitted (including transformed) MoveJoints trajectories. This is
a bounded heuristic, not a complete bimanual planner or stability guarantee.
Composition keeps original Pick closure and total sample counts, and retains
canonical gripper effort limits instead of recalibrating an earlier light-object
grip from which arm later Pour/control recipes use. Legacy/default and contact
profiles retain their existing effort calibration. Contact also
retains its qualified 24-point ordinary Pick closure with 12 extra total samples,
leaving approach/lift counts unchanged. Frozen physical inputs reject these opt-ins.
`planning.experimental_open_release` independently selects the release observer;
packaged defaults set false, while both explicit experiment configurations set true.
Old bundles and direct builder callers that omit the marker retain their original
profile-based release selection. Its native grip observer revision 2 permits an experimental detached
observation when the measured independent URDF driving joints are within the
profile's existing open tolerance and exactly one pad remains in contact.
Acquisition and holding still require native opposed bilateral contact;
nonfinite, dropped, bilateral or non-opposing two-pad observations cannot use
this release rule. Raw contacts and measured opening are recorded. This means
open-hand loss of bilateral grip, not contact-free separation; actual retract,
placement and settling still execute. Default revision 1 remains unchanged.
Frozen physical references require runtime_profile=default and reject incompatible
profiles before any model calls; coordinator and bundle checks remain in place.

Explicit experimental acceptance requires complete native program execution,
all original segment/post-policy/validator checks and cargo acceptance; only
the outer TaskSpec certificate is advisory. The canonical runner preserves
`task_spec_evaluation.json` and publishes `strict_execution_report.json` beside
the clearly marked experimental report. Both `run` and `run-all` use this same
production path, without importing artifact-directory workers. Every execution
also writes program_completion_summary.json, separating program completion from
strict success without re-executing or replacing the selected verdict. The CLI
default reports experimental program completion; strict TaskSpec certification
remains separate and can still be selected explicitly.

`generate_task_program_bundle(verify_pick_phases=True)` explicitly enables
registered Pick acquisition/lift verification through its fingerprinted lowerer
declaration. The default preserves legacy factory identity; closing geometry,
alignment, release and support contracts are independent. Phase ownership and
assurance are defined by [Task Program execution](../task-programs/execution.md).
Normal Task Engine preparation opts new bundles into this contract. E8 regeneration
preserves the input declaration; mixed per-Pick modes require explicit handling
and are rejected by the uniform generator rather than silently weakened.

Gradio uses explicit allowed roots and per-session process ownership. A
replacement run terminates the previous process for that session. Remote
access requires complete basic auth; repository/dotenv path restrictions belong
to `app_env.py`. Existing process environment values take precedence over loaded
environment files.

## Task Engine semantics and execution scope

Prepared scenes use intrinsic `XYZ` degree rotations. Bundle scene payloads
serialize explicit `init_local_pose` matrices so generic configuration decoders
cannot reinterpret those angles as extrinsic `xyz`; keep this boundary distinct
from the Task Program's `quaternion_xyzw` serialization. E6 runtime binding,
geometry bounds, rail direction and handle clearance use that same matrix;
never reinterpret a decoder's derived extrinsic angles as source `XYZ` angles.
Explicit proxy fitting updates an existing matrix together with its pose fields.
Legacy editable revisions bake runtime scale into copied GLBs and use unit
editable scale. Their scene config carries `world_alignment_applied`; preserve
it through edits so already-normalized poses are not rotated again.
Generated deployments retain the measured settled layout instead of teleporting
objects back to configured X/Y after settling.
Invocation-scoped Cartesian policy also covers single-EEF-target transports;
joint targets retain their joint-space behavior. E2 alignment preserves the
acquired orientation during the staging lift, then turns the object aloft.
E2's upright goal does not impose an initially fallen source pose: prior steps
may change the object's orientation. Grounding preserves the full object reference;
the existing runtime alignment and upright checks still execute without goal-based skipping.
E2 checks its final yaw-free alignment at release height before Place preserves
that heading. Alignment staging never descends below the current height before
the planner can choose the final orientation.
Ordinary GenSim Place retains its qualified initial Cartesian result. On initial
failure, an unowned-Cartesian, single-row preserve-orientation call may probe
shared waypoint IK/joint interpolation before density/local-IK recovery. Only
a fully qualified plan with at most 45 degrees of intermediate rotation skips
that repeated work; otherwise the pre-probe RNG is restored and the complete
original recovery runs. Safe early large-turn plans remain available if later
searches fail or produce worse waypoint plans; 45 degrees is a scheduling
criterion, not a safety threshold or rejection of the large-turn fallback.
Every required waypoint and the exact release pose remain unchanged. Acceptance
checks final emitted velocity/joint limits, discrete robot meshes, and the held
object's conservative scene envelopes through approach. Overlapping AABBs are
refined with convex-solid covers of the same complete source vertices, retaining
the 2 mm separation requirement. Containment, unavailable geometry and unresolved
distance reject; this is not native contact or continuous collision certification. Explicit Cartesian
routes, dynamic collision policies, drawer routes and E2 z caps are not overridden.
Missing bound rigid geometry or bound articulation obstacles disable this fallback.
Motion validation revision 13 separately owns positively bound ordinary on-Place
calls with `preserve_current_object_orientation=false`, B1 and `ik_interp/off`,
excluding explicit Cartesian, custom-grasp, pour, upright and direct routes.
It retains a qualified initial dense plan unchanged. Otherwise one goal-seeded
waypoint solve replaces the density/local-IK/via recovery chain for that call.
The shared builder still owns commands, tracking and effects; phase composition
retains every mandatory pose and the original timing. Emitted FK, joint/velocity,
robot-mesh, payload-envelope and release-resource checks reject unsafe results
without another retry. This is not continuous collision or physical certification.
Ordinary held transport retains shared IK/interpolation, the existing target
construction, state handoff, general joint/velocity limits and effect monitors.
It has no additional carried-mesh sweep, robot-mesh path qualification, bounded
held-path recovery or post-transform collision recheck. Its completed motion
does not certify intermediate table, payload-scene or other-arm clearance.
`held_command.py` retains compiled same-object/resource source declarations;
these declarations alone are not runtime trajectory qualification. E2 alignment,
Pick, Place and handover keep their own checks.
Explicit `pour_geometry_revision=1` uses `pour_services.py` and
`pour_continuation.py` to bind ordinary orientation/axis/direction-free E3 to its
exact Pick, staging, Pour and absolute return calls. Staging preserves current
observed orientation; `pour_geometry.py` supplies the candidate-dependent
five-role look-ahead and actual tilt/restore geometry. Existing held G remains
the command-planning reference, not a native pose witness. Direction-free tilt
uses the ordered pad axis; receiver-bearing drift is diagnostic, not permission
to reverse the qualified arc. The new Pour lowerer
declares existing acquisition/retention verification, with no new executor or
path fallback. Adapter v13 requires regenerated consumers after the current
Pick qualification changes; historical bundles and receipts remain unchanged.
The independent
recipe does not own already-held E3 continuations; normal preparation
keeps those compositions on their legacy declaration rather than opting them
into a five-call contract they do not have. The Default/B1
source-only sweep proposal is not hand/cavity/collision qualification, and
`pour_evidence.py` evaluates a separately named native pose proxy, never liquid
transfer. An unmodeled `poured` TaskSpec remains unavailable; geometric motion
must not silently become its certificate.
Explicit revision 2 requires `pour_proxy_position_tolerance_m` and a distinct
`pour_region_transport` factory. Its runtime-only native vertices/triangles
snapshot is shared by Pick preview and the actual stage lowerer. The same pure
`pour_region.py` rule insets the receiver's actual triangle projection by that
declared position budget and chooses its arm-nearest point. It changes staging
XY only, retains observed R/height, and rejects an empty or invalid region without
falling back to the center. The active arm origin comes from its compiled motion
endpoint's solver-root frame. Explicit staging points and constrained E3 remain
unowned; legacy and revision 1 declarations/fingerprints remain unchanged. This
is a render-mesh target proposal, not an opening, collision or fluid certificate.
Region invocation metadata must bind the same position budget used by the
pose-proxy observer; that observer does not certify the original `poured` goal.
Bare `table/on` targets retain legal source XY; only occupied landings select a
measured free tabletop region. Fixed-base prismatic travel envelopes qualify
endpoints only. Named regions, explicit poses and E2 in-place goals do not move.
The bare E1 table route uses a `supported_region` preset: its planning point is
a proposal, not a normative coordinate goal. Live stability and TaskSpec join
exact native support, full measured planar footprint, contact height and release
evidence. Point/upright/named targets retain their prior constraints. Episode
geometry caches remove only exactly duplicated payload points; poses, contact
and acceptance are freshly measured on the unchanged stability cadence.
New bundles may explicitly request `support_contract_revision=1` from
`generate_task_program_bundle()`. `support_evidence.py` captures rigid-pair load
patches during actual physics substeps; `support_contract.py` binds their original
sample time, episode and actors to the exact release phase. The pinned Default
contact-cache revision getter is capability-checked; missing provenance is
unavailable, never a newly stamped cached observation. `stability.py` uses the
canonical acknowledged environment clock, and `task_spec_runtime.py` independently
joins historical terminal support and current final support. Strong declarations
cannot fall back to geometry-only evidence; unsupported relations or moving links
remain unavailable. The marker changes the bundle fingerprint, not physical
parameters or legacy default behavior. It is not itself native qualification.
Up to three shared waypoint plans may be compared for the same acquired grasp
and unchanged targets. Each must pass every existing safety gate. FK records
maximum/intermediate rotation, not just final heading; a low-rotation plan is
preferred, while the best qualified large-rotation plan remains the fallback.
Extra path searches restore the caller RNG and do not reselect a successful Pick.
Successful paths and failed-search random streams remain unchanged. This is not
continuous collision/contact qualification or a remedy for unreachable waypoints.
Ordinary, factory-marked single-environment GenSim Pick qualifies every contact
sample against its declared approach direction and selected grasp pose. A curved
contact leg is rebuilt backward from the original grasp joints through shared
Cartesian IK. Free-space entry targets that exact pre-contact joint branch, using
checked joint interpolation first and a fresh-world cuRobo plan if necessary.
The selected grasp, close/lift suffix, passive joints and shared executor remain
unchanged; emitted limits, robot meshes and all approach scene samples are checked.
Failed or unavailable qualification stops dispatch, not a fallback to the unsafe
original. Extra searches restore caller RNG. Explicit, constrained, upright, pour,
drawer and multi-environment calls retain their prior routes and bundle fingerprints.
After ordinary Pick path qualification fails, up to four other legally
prefiltered grasps may be replanned with their own approach directions. Distinct
wrist orientations are tried before near-identical poses, within the same
four-plan budget. Each
complete plan keeps its attachment proposal coupled to its commands and must
pass contact FK, emitted-limit and mesh checks; a bound Place continuation must
also pass its private preview. Successful originals are not reselected.
Failed ordinary Place may try two elevated approach vias, retaining every
required release waypoint, its exact terminal pose, source dependencies and
timing. These candidates require emitted-limit, robot-mesh, held-object scene
envelope and release-FK qualification. Explicit Cartesian routes, height caps,
drawer routes and pour/ends grasps do not acquire this recovery. Extra searches
restore caller RNG; no TCP symmetry is invented to make a target reachable.
`_task_program/motion.py` owns one bounded seeded-path resolver shared by the
motion-generator adapter and E6's root-frame IK adapter. Previous-waypoint
seeding, local perturbations, joint limits and per-step velocity admission live
there; E6 retains its final FK, complete-plan velocity and collision checks.
This is greedy local continuation using existing solvers, not a new IK solver,
global branch search or physical-success certificate.
Deployments with an ordinary Pick bind grasp constraint observations to the
queried object's two native finger-pad contacts on the queried hand. Positive
impulses and opposing normals qualify acquisition; absent contacts qualify
release. One-sided, non-opposing, nonfinite or saturated observations remain
unverified. Shared pose-relation, consecutive-sample, lift and release monitors
are retained; a wide object does not need an arbitrary minimum closing angle.
This uses the canonical dual-Franka pad bindings and is not a new executor or
a certificate of complete-task success. Regenerate affected bundles.
Ordinary adjacent typed Pick/inside-Place calls with verified canonical monitors
may qualify the complete continuation before dispatching Pick. A qualified
original remains unchanged. Only a failed required-waypoint IK result permits
contact reselection; velocity-only rejection or unavailable evidence preserves
the original acquisition. Up to twelve existing legal grasp
contacts are considered, with full Pick/Place planning, emitted limits and mesh
checks; the semantic landing pose and resource bindings are never changed.
The preview projects the emitted Pick endpoint FK in private state and is not
measured acquisition evidence. Actual Place still observes and replans through
the canonical executor. This runtime-only binding adds no bundle fingerprint
fields and does not apply to upright, fixed, explicit, constrained or pour/ends
grasps. Collision evidence covers discrete robot self meshes and bound rigid
held-object envelopes, not whole-scene continuous or contact qualification.
Only yaw-free alignment calls enable alternative final headings; exact-pose
transport retains its orientation contract. All candidates use the same motion
and velocity checks.
Invocation-owned constrained center Pick also qualifies the complete emitted
approach, close and lift against robot proxies. A safe original plan is kept;
an unsafe successful plan may try bounded alternatives from the same filtered
grasp pool using the shared Pick planner. Explicit grasps, upright-rotating
Picks and downstream-lookahead Picks retain their separate contracts. Observed
start collisions and unavailable certificates fail closed. These checks do not
certify held-object scene contact or physical grasp success.
A free hand whose first open command alone exceeds the observed velocity
boundary may receive a bounded, real opening prefix within the existing hand
interpolation budget. The arm and passive joints hold their observed positions;
the original Pick suffix and attachment remain unchanged. The complete emitted
plan is qualified again, without treating failed attachment as held state or
changing the shared planner, command targets or velocity limits.
Yaw-free non-drawer GenSim held alignment also screens robot self/other-arm
proxies before accepting a heading. If the existing heading search exhausts,
one bounded local IK recovery may preserve the last heading's Cartesian path
and timing, including when every heading fails IK or velocity qualification,
or when raw IK succeeded but proxy screening rejected it.
An observed start collision does not permit that recovery. Extra searches
restore the first rejected legacy-stop random state, or the original exhausted
heading-search stop when no proxy checkpoint exists;
safe original paths retain their normal random stream. Emitted joints are
checked again after plan transforms. This is not a held-object scene-contact
certificate; attachment monitors and existing collision policies remain active.
Factory-owned, adjacent same-object/resource verified yaw-free Align/Place calls
also preview the canonical Place from a private planned endpoint. A qualified
original continuation retains the exact Align plan and random stream. Only a
failed joint continuation searches the original eight headings, with unchanged
targets, timing, limits and retained-command authority. Preview effects never
commit; actual Place still observes, grounds, replans and verifies release.
If that search fails after robot-proxy rejection, explicitly bound native-rigid
alignment occurrences may try two bounded staging detours. Their intermediate
poses preserve the acquired orientation, original staging height and final
goal; the passive arm does not move. Measured distal-mesh projections provide
only route hints. The shared planner rebuilds each complete plan, whose emitted
commands undergo the unchanged robot-proxy, joint-limit, velocity and retained
grasp checks, including the observed first command and post-transform boundary.
An unchanged, already-applied grasp target is command continuity rather than
new finger motion only when the adjacent verified attachment contract and fresh
native target position/velocity prove it; changed or unavailable targets still
fail the original boundary checks. This does not certify contact force.
Extra detours preserve the legacy-stop RNG state. Other calls and safe original
plans do not acquire this recovery implicitly.
Unrelated `DrawerRoute` declarations cannot clear explicitly verified ordinary
staging authority. Owned drawer transports still opt out through their actual
invocation-local direct-transport scope and retain their payload-aware route.
Mixed-route authority changes invalidate old integration fingerprints; regenerate
new bundles rather than resealing historical ones. This is an authorization
boundary, not full articulated-world or continuous-collision qualification.
Generated released-hand cleanup uses `task_engine/_task_program/clearance_recovery.py`.
An adjacent same-object, same-resource verified Place/clearance followed by one
or two distinct sequential Parks including the owning arm enables its bounded
route search. Both registered releases and curated Place use their actual
effect-monitor binding. Canonical shared-compiler policy grounds each configured
Park from the previous full-robot endpoint. Qualification checks every segment
and boundary in execution order: emitted commands, timing, joint limits, passive
holds, and fresh scene/self/other-arm collision meshes. An observed clearance
start collision stops route search immediately. The shared executor receives
only the accepted cleanup plan; each Park replans and revalidates observed state.
This gate currently supports one environment at index zero and fails closed
outside that scope or when collision evidence is unavailable. Sampled mesh
clearance is not continuous collision or physical-success certification.
Only those factory-bound pure Parks may recover an interior scene collision
using four finite joint vias derived by the existing IK solver. Original ready,
control grid, limits, passive holds and collision exclusions remain unchanged.
Cleanup previews and fresh actual Park planning use the same complete gates;
safe original plans and the caller's random stream remain unchanged.
E5 Robotiq grasp clearance is isolated in
`task_engine/_task_program/coordinated_grasp.py`: only the dual-grasp service
uses URDF collision surfaces and deployment open/grasp commands in TCP space.
Antipodal points seed a bounded finite-pad gap, centering and insertion search;
their separation is not assumed to be the actual contact gap. Watertight target
triangles preserve cavities that coarse convex unions can fill. Both inward pad
faces must reach contact, with the open-to-contact sweep clear of other hand
parts. Non-watertight targets fail explicitly instead of receiving guessed signs.
Near-contact inward-pad samples also require outward target normals within
60 degrees of the corresponding pad-facing direction. This excludes tangential
or back-facing proximity without treating it as force closure. Normal queries
reuse the target triangle BVH and run only on near-contact pad samples; rejected
endpoints do not enter the closing sweep. Two-sided occupancy probes orient
each queried normal using the signed-distance solid convention, not global
mesh volume: independently reversed components and nested cavities retain
their own inside/outside boundary. Ambiguous probes cannot certify contact.
Regenerate E5 bundles after changes
to this versioned geometry policy.
Single-arm generator methods and
their factories remain unchanged even in mixed E1/E5 programs; do not tune their
finger envelope or opening margin at program scope. This target-mesh screening
does not certify continuous approach clearance, contact dynamics, or attachment.
`coordinated_motion.py` composes the shared planning helpers for E5 continuation:
two verified grasps of the same object are retained through motion, and hands
open only at release. Missing or conflicting held state cannot fall back to
re-picking. Consecutive E5 relative targets accumulate for terminal checks;
release retreat exceeds the unchanged separation threshold with tracking margin.
`coordinated_evidence.py` observes registered E5 terminal holds separately from
motion execution. It binds both native ATTACH receipts, the authored hold policy
and recorded endpoint poses to the exact compiled occurrence; final holding also
requires fresh native contact for both hands without advancing physics.
`held_evidence.py` owns the shared receipt, pose and grip-query validation used by
E4 and E5. A terminal hold does not qualify in-flight acquisition guards,
continuation, released transport or cargo retention.
E5 `terminal_behavior=place` with `relation=on` retains the bound rigid support
instead of converting it into an initial-pose direction. Bundle generation
reuses E1 relative-placement height geometry; the coordinated lowerer targets
the observed support position while preserving the carried object's observed
orientation. Release clearance is excluded from the resting-position target,
which is checked relative to the support again after cleanup. This uses the
existing E5 position/stability acceptance, not a new contact certificate; mesh
envelope limitations remain. Motion budgets are declared per generated segment
in `constraints.json.motion_samples`, then bound to compiled invocation IDs by
`invocation_policy.py`. The base execution policy is no longer raised by scanning
the whole program. E2/E4/E6, drawer-placement and on-support E5 recipes retain
their staged-motion budgets without changing unrelated calls. The GenSim engine
selects immutable request policies and observes the runner's initial plan;
planning, execution, recovery and state effects remain in the shared engine.
Generated PytorchSolver configurations retain GenSim's 5 mm position tolerance
explicitly, without changing the shared solver's 0.5 mm default. Existing
explicit solver tolerances take precedence; rotation tolerances are unchanged.
Larger caller budgets, control dt and
velocity limits are preserved. Missing/stale bindings require bundle regeneration.
Cartesian sampling is declared alongside budgets in `cartesian_calls`, bound
to compiled invocation IDs, and enabled only within an exception-safe planning
scope. Appending a HandOver, clearance or drawer recipe no longer switches the
whole program's generator. Hardware, initialization, remaining option selectors
and random-state scopes still require separate qualification.
Normal bundle execution writes `planning_probe.json` from the actual first
invocation via `_task_program/planning_probe.py`, without separately planning it
again. Full-robot post-resampling velocity validation still gates that plan
before dispatch; diagnostics distinguish execution evidence from the independent
`--plan-probe-only` mode. The run-local observer matches the exact invocation ID,
restores its scope on exit, and retains no executable plan. Genuine runtime
replans continue to use fresh observations; this is not a cross-run plan cache.
For non-drawer Place, the current `GenSimPlace` planning scope selects the same
Cartesian sampling policy in both checked and approach motion generators. Its
planning and recovery do not depend on unrelated E2/HandOver calls elsewhere
in the program. Outside this scope, each generator's existing policy remains
unchanged; drawer and cuRobo routes retain their own policies. Ordinary Place
paths that previously used joint interpolation may therefore change; regenerate
bundles after the motion-policy revision. For a single environment only,
exhausted Cartesian IK planning may retry through the shared motion generator
with nearby joint seeds and a joint-limit-margin preference. Successful
Cartesian plans do not enter local IK recovery;
the fallback preserves Cartesian samples, control timing and random state,
and rechecks full-robot velocity after Place resampling. GenSim Place retains
v38's scene-dependency guard through approach only; expected contact motion
does not replan release/retract. After that boundary external disturbances
are also ignored by this guard, so final effect and stability checks remain
required. Shared Place retains its full-action guard. This is a bounded
kinematic recovery, not collision or physical-placement qualification.
E6 recovery is bound to exact registered Slide/withdraw invocation IDs, not
all calls of those underlying skills. Bound single-environment `ik_interp`
Slide retains its 85% preshape unless every finite candidate fails aperture.
Only then, measured whole-hand FK selects the smallest sufficient opening in
the 84--80% closure band and rebuilds once with matching Slide options and
call-local clearance. The 4 mm pad reserve, full-hand/table screening and all
execution gates remain; width-fitting/table/IK failures do not widen the hand.
Initial-plan evidence records the selected plan, not the intermediate rejection.
Successful defaults, explicit nondefault preshapes and non-E6 calls are unchanged;
regenerate E6 bundles for the preshape fingerprint revision.
Failed single-environment CPU
`PytorchSolver` / `ik_interp` pull-and-release plans can screen up to four
terminal configurations, solve the contact path backwards, then connect a
free-space joint approach. Slide recovery is capped at 1.5 times the request
and 390 command frames; larger caller budgets are not shortened. Failed E6
withdrawal can try local IK without changing its targets or frame budget.
For an adjacent Slide/withdraw pair with identical part/state/resources,
failed-Slide recovery also qualifies each proposed release posture against
the declared withdrawal sample budget. It shares the runtime's unchanged
withdrawal geometry, screens continuous IK, velocity and robot meshes, and
discards the preview path. This is not a predicted-scene certificate: actual
withdrawal is freshly planned and checked against current scene meshes before
dispatch, including when its nominal plan succeeds, only when the lookahead
actually rejected the formerly admissible terminal and selected another one.
Unchanged successful branches and unpaired calls retain their existing checks;
they do not acquire a blanket new scene gate. Qualification is read from the
paired public ExecutionSession's installed plan attempts, not from speculative
planning side effects. Rejected revisions cannot erase it; a later nominal
replan in that session does not erase earlier installed reselection evidence.
The E6 recovery fingerprint revision owns this change.
Failed five-stage push/close plans may instead qualify up to four grasp-seeded
branches using the original per-stage targets and frame counts. Backward reach
joins forward push, release and return through continuous IK; only the empty-hand
approach uses joint interpolation. Shared Slide rebuilds every command. Final
full-robot velocity and robot-mesh checks, fresh approach-scene screening and
the existing paired withdrawal preview remain required. Scene screening does
not certify future contact/return geometry or physical closing. Successful
push plans remain unchanged. A rejected widened preshape retains its own
diagnostic failure alongside the original nominal rejection, without changing
which failed plan is returned.
The shared skills still rebuild arm/hand commands and the shared executor
still owns execution. Both routes retain tolerances, restore RNG/planning
scope, validate final full-robot velocity, and require fresh discrete mesh
clearance for the changed free-space motion. This is not continuous collision
or zero non-target-contact certification. Successful Slide plans, other semantic
calls, collision-planner routes and shared Lab defaults remain unchanged.
Recovery needs the GenSim `yourdfpy`/`python-fcl` dependencies; unavailable or
unsupported geometry fails closed. Regenerate E6 bundles for the recovery
fingerprint revision; non-E6 fingerprints are unchanged.
E2 release preserves the observed aligned heading instead of adding another
fixed yaw during descent; an upright instruction does not require that rotation.
Generated E1/E2 deployments, including light E5 compositions, cap only grippers
used by single-arm picks at `max_effort: 0.5` when every picked rigid body and
E5 carrier has an explicitly declared mass of at most 10 g.
The full closing range is retained; smaller existing effort limits are not
increased. Unknown/heavier loads, E5-only programs and other recipe families retain their declared
controls. Qualification is per grasp resource: an E6/E8/E9 on the other hand
does not disable the cap; a hand shared with those operations keeps its authored
controls. Payload qualification is likewise hand-local. Imported mimic behavior
and shared robot defaults are unchanged.
For an existing, unedited scene, `planning.physical_reference` in the ordinary
Task Engine config explicitly selects a reference bundle's physical components
before initial binding. `workflow.py` pins one `FrozenPhysicalInput` through
SceneAdapter, final inspection and coordinator bundle generation; inspection
measures that selected model under its physical digest, not the original scene.
The selection owns episode limits and rejects additional asset/mass adaptation,
source-transform overrides and reference/output overlap. Existing physical
manifests are verified before import. Original-source visual grounding and
reference-image review are not valid evidence for the selected model and are
disabled in this mode. `selected_physical_input.json` records source/model
differences; selection never certifies original-scene equivalence or historical
success. Without an explicit reference, legacy graph-dependent calibration is
unchanged; this opt-in contract is not default-wide composition isolation.
Ordinary E1 placement on another object retains its position tolerance and uses
`supported_placement`: scaled local mesh vertices are cached per environment,
then transformed by observed poses for vertical gap and projected support-envelope
checks. It does not impose initial roll or either object's initial up direction.
Both objects must still remain stable for the declared window. `on` additionally
requires native upward contact between the exact object pair supporting the COM,
plus measured release; it does not require whole-silhouette containment. Support
sensors are registered in the environment's physics-substep registry. A sleeping
contact witness survives only unchanged body poses, is invalidated by overflow
or movement, and is cleared for reset environment rows. Final support is measured
again; milestones retain their exact observed boundary receipt. This is not
arbitrary concave/tilted-support or three-dimensional cavity qualification.
Relative side placement binds an articulation body or an explicitly grounded
prismatic `reference_part` to one native `SceneLinkRef`. Geometry includes only
the selected link's collision meshes, in its scaled native frame, never a proxy
or the whole moving articulation envelope. Pick lookahead, Place and stability
read that same live link and world-axis displacement; landing height comes from
the measured tabletop. A preceding Slide therefore does not invalidate a
qualified reference. TaskSpec side acceptance measures the selected link's full
world-axis envelope, tabletop support and exact release evidence. Historical
milestones require the matching post-policy link pose and recorded object pose;
final acceptance observes the current link rather than reusing that receipt.
Regenerate relative-placement bundles to declare qualified link routes. This is not a
swept collision certificate. Articulated `on/above/inside` support still requires
its separate qualified link geometry; spatial references do not authorize
joint manipulation.
Explicit upright, oriented hold and stack constraints retain v38's 10-degree
alignment tolerance. Upright handover-to-stack retains the table restaging,
regrasp, preserve-yaw alignment and released-hand cleanup route rather than
directly stacking from the receiving grasp.
Source pose matrices take precedence over Euler fields.
After UID grounding, independent E1 placement, E2 upright, and E5
lift-and-return `count`/`all` sets lower to ordered single-object steps;
line layout, E5 terminal hold, other task routes, and a downstream
`step_result` consumer still reject implicit expansion.
E6 `count`/`all` sets retain their qualified synthetic part UIDs through
binding, then expand to ordered Slide/withdraw/Park recipes with a distinct
part binding per step. Cabinet-root sets do not stand in for drawer parts.
For opening sets, measured world-space handle heights order parts bottom-to-top
within each cabinet. The existing 1 cm spatial tolerance groups same-height
parts, retaining their original lateral order, including horizontal rows and
multi-column layouts. Missing geometry leaves that cabinet's order unchanged.
Explicit separate steps and closing sets are not reordered. This is a default
sequence preference, not a collision-free planning guarantee.
E6 uses each qualified handle's position for automatic arm
preference rather than the shared cabinet center; explicit arms take precedence.
This code-owned geometry stays outside the model prompt and runtime scene config.
The final program segment rechecks the last requested position of every E6
joint, so later manipulation cannot silently invalidate an earlier drawer.
Repeated open/close commands on one joint retain only its last terminal target.
Regenerate existing E6 bundles to include these terminal checks.
Part inventory exposes a configured initial open/closed endpoint only when
the runtime default zero reset matches an explicitly authored, validated
`gen_sim:closedPosition` endpoint or its opposite endpoint. Missing endpoint
semantics, explicit `init_qpos` vectors (unqualified joint ordering), and
non-endpoint resets remain unknown. This is configuration evidence, not a
settled-state/contact observation; no asset or reset value is changed.
For an unbound existing Gym scene, Task Engine automatically renders the
current GLB assets into UID-labeled overviews, segmentation masks and crops,
then retries grounding with the same LLM transport configured for text binding
in `gen_sim/.env` or the process environment;
`--reference-image` is not this binding input. A deterministic tenth of already
bound scenes receives an audit-only visual spot check. Text and visual binding
share semantic identity rules: descriptive categories are not exact-match labels;
synonyms, translations and supported broad names may identify the same entity,
but explicit modifiers, exact UIDs and counts cannot be relaxed into substitutes.
Model responses include matching support, conflicts and nearby candidate UIDs;
reported conflicts cannot authorize a resolved binding. This evidence is not
new scene state or physical capability. Legacy four-field response replay stays
supported. Genuine missing/ambiguous results are not retried into guesses.
Run-local binding audits retain prompts, schemas, responses and model identity,
including early failures; candidates and source fingerprints survive failed
selection. Visual evidence also records scene revision and image hashes; the
VLM may select only visible inventory UIDs, while native semantic compatibility
remains authoritative. `task_engine/binding_evaluation.py` provides frozen
positive/rejection controls and per-reference/paraphrase metrics; provider calls
require `--run-model`. These text-only metrics do not certify visual or physical
success. Articulation
proxy GLBs do not represent live joint state, so E6-E9 binding does not use
this visual path. Without usable Task Engine LLM configuration, text-only
binding and its existing conflict behavior remain unchanged.
E4 may defer unnamed transfer/receiver arms as `auto`; the planner binds the
source from the held state or nearest scene side and the other arm as receiver,
while preserving explicitly named arms. GenSim declares `pick_purposes` per
generated Pick segment, bound to the compiled runtime invocation identity;
missing or stale declarations reject loading. Built-in Pick identity and
compiler-owned downstream reachability remain unchanged. Ordinary and pour pickup
try top-down before arm-relative diagonal approaches; an infeasible top-down
strategy retains the diagonal fallback. E4 source lowerers try the arm-near end top-down and
diagonally before considering the far end, leaving the other end for reception.
The wrapper marks only an owned goal
snapshot; upright and stack constraints take precedence over ordinary policy.
`GenSimPickUp` screens each
candidate strategy through the shared full pickup planner and retains the exact
successful trajectory per row; endpoint IK alone is not acceptance. Explicit
grasp poses and fixed grasps retain their contracts. A failed constrained,
single-environment Cartesian Pick without downstream look-ahead may try one
symmetric roll in the calibrated grasp frame. The original opening, region and
release-clearance filters gate that alternative; the shared Pick planner rebuilds
its trajectory and proposed attachment together. Full-robot velocity and arm
joint limits gate the final resampled path. Successful original plans and later
calls' random streams remain unchanged; exhaustion retains the original failure.
Its direction prefilter removes only candidate columns whose two roll variants
are rejected in every environment, then delegates to shared `PickUp` and restores
the original candidate indices. Upright adjustment and unconstrained approaches
delegate unchanged. Keep the full environment axis; do not copy the IK loop or
change shared defaults to optimize this GenSim selection.
`GenSimHandOver` chooses the receiving end from the actual source attachment.
Vertical receiving grasps try diagonal then top-down; horizontal ones reverse
that order. Candidate, IK or path failure triggers one alternate-direction
attempt through the complete shared HandOver planner before execution. Successful
rows retain their first trajectories and attachment candidates. Exhaustion
fails normally; this planning retry adds no source regrasp or cross-call recovery.
Generated E4 deployments qualify the receiver hand's emitted joint-command
grid in the supported single-environment Default scope against invocation-owned,
enabled native non-target collision shapes through
`handover_safety.py`. Materialized geometry owns its effective scale; do not
reapply an asset YAML scale that USD preservation may have replaced. Safe
trajectories, effects and timing are retained; full-plan failure may use only the
existing alternate receiving direction. Deterministic (`affordance_sampling=None`) receiving
selection first filters that same candidate pool with the original Cartesian
approach grid and bound-open native hand geometry, without extra candidates,
IK calls or random draws. Candidate costs and ordering remain unchanged; the
final emitted-command check is still required. If that original pool is exhausted
only by scene geometry, `handover_exchange.py` may select an internal exchange
position from bounded, native-geometry-derived displacements in the root middle
plane. It preserves the original relative grasps and candidate costs, uses the
configured IK and joint limits, restores random state, and does not change the
semantic task target or physics. The shared existing-hold receive hook returns
the exchange pose with its grasp sample, so source transport and the destination
attachment use the same pose. Safe original pools do not enter this selection.
Nominal candidate screening is
not a certificate for an IK solution or physical acquisition, and stochastic
branch sampling retains its existing selection plus the final command check.
This discrete hand/environment check does not
certify target contact, robot self-collision, payload clearance, future scene
motion or between-sample clearance. Its E4-only revision is fingerprinted;
regenerate changed E4 bundles instead of resealing historical artifacts.
Transfer, release, retreat and state effects remain owned by shared HandOver.
Only derived adaptive handover ends may request the grasp service's call-local
`supported_end` policy. An already viable mesh-end pool retains its poses,
costs and random draws unchanged. If that geometry pool is empty before IK,
the same prepared pairs and approach variants may define an end from the
angle/NMS/static-qualified support, before ranking truncation. Explicit
top/bottom or hard task regions remain `mesh_extent`; IK, path and physical
acquisition failures never activate this reinterpretation. The service owns
the geometry stages, not a second GenSim sampler or deployment-wide setting.
Its revision is fingerprinted only for generated handover deployments.
Generated E4 integrations include the curated Pick monitor needed by the existing bounded
workflow recovery, even when source acquisition uses a registered alias.
This does not change recovery budgets or qualify initial grasp stability;
regenerate bundles to acquire the declaration rather than resealing old files.
Generated profiles select `source_hold_mode=grasp_command` to retain the bound closing
drive during existing-hold transfers. GenSim composition applies that default
when the field is absent and preserves explicit modes. Missing retreat budgets
also retain v38's default; shared composition keeps its own defaults. The mode's
contract belongs to
[Atomic Action execution](../atomic-actions/execution.md).
E5 paired grasping and E6 handle policies remain separate. These GenSim-local
skill replacements use the shared goals and options. Unmarked calls
delegate unchanged; drawer deployments install the same ordinary wrappers.
Constrained E2 Pick lowerers
snapshot their object/target rule selector into the goal; only that Pick uses
an immutable filtered-generator view. Unscoped sampling, including HandOver,
checks opening clearance without applying another stage's region/release rules.
Drawer transport and release remain drawer-owned. Relative placement geometry and
stability use preceding alignment operations, never future E2 state. E2 after
an explicit on/above placement retains that support and uses its measured top
for staging, release and grasp clearance. Its constrained regrasp uses a
top-down approach so uprighting yields a side grasp instead of an end-directed
rim approach. Standalone E2 retains its original table target and grasp policy.
The GenSim-owned `gen_sim.place_upright` call reuses Place while
separating the later upright release geometry from the earlier ordinary Place
route. Repeated occurrences retain stage selectors; conflicting geometry for
the same stage is rejected rather than overwriting its route. An implicit
singleton retains the v38 declaration shape, while an authored selector is
preserved even for a singleton.
Terminal preserve-pose handovers verify receiver attachment and the
declared world-axis alignment. A preceding upright stage makes that axis
world Z instead of reusing the initial fallen orientation.
`handover_evidence.py` connects exact native TRANSFER receipts to E4 TaskSpec
events, separately from terminal receiver holding and orientation observations.
Authored hold policies retain their existing duration and attachment/axis checks;
an undeclared hold window is not added. Historical milestones use their own
recorded boundary, while final holding also checks fresh receiver-relative pose.
Missing receipts or unsupported later execution remain unavailable, not inferred
from command completion. This changes observation, not HandOver execution.
E4 `place` outcomes with rigid `on`/`inside` targets also collect exact receiver
release and placement evidence, separately from the earlier native TRANSFER
milestone. Observer revision 7 records those two boundaries in execution order;
final geometry cannot fill a missing transfer or historical placement. Explicit
support revision 1 applies the same native support contract to these E4 placements
without weakening legacy E1 or terminal E4-hold behavior.
`task_engine/_task_program/identity.py` owns v38 bundle identity for both
generation and verification. It projects only equivalent default HandOver and
empty relative-route stage fields at validated declaration slots; explicit new
behavior, parameters, revisions, program and constraints remain fingerprinted.
The shared registration retains its full runtime fingerprint and drift checks.
Compatible old files are neither rewritten nor regenerated. Actual contract
changes still fail closed; do not bypass their fingerprint checks or extend
the shared Pick/AxisAlign configuration schema.
Multiple children
placed inside the same measured container reserve each prior landing footprint
and margin; insufficient space or an unmeasured shared floor fails preflight
instead of assigning overlapping destinations. Each landing uses only declared
orientations preceding its own Place; a reservation retains its placement-time
orientation rather than a later alignment target. This is a planning snapshot,
not a live occupancy tracker or physical placement acceptance.
Generated non-drawer held-object transports declare a measured attachment postcondition.
GenSim installs PickUp, HandOver, MoveHeldObject, Place and Pour wrappers for all
deployments after shared engine validation and configure shared Lab action
options through the skill profile. The transport wrapper stages above the target and
returns the input attachment as its expected effect, so the existing composite
monitor verifies retention before observed-transform reconciliation. This is
terminal verification, not in-flight slip prevention. Regenerate older bundles
to include the wrapper registration and revised lowerer fingerprint.
E3 now uses the vessel's geometric upright axis only as an upright reference,
not as its rotation axis. The selected grasp is checked at runtime against the
actual pad line before the tilt arc is generated;
The GenSim Pour wrapper reads the two native finger-pad link poses (not TCP-X)
and derives the projected pad-line as its rotation axis. Its visible object
tilt is the cross product with upright, hence perpendicular to the actual jaw
line. It samples the full tilt-and-return arc through the shared motion
generator.
No spout annotation is required for this tilt-and-restore contract. Receiving
vessel identity is bound per Pour invocation from that recipe's explicit staging
reference, not from an object-wide map. Its position is late-bound while the source's upright heading is preserved
independently of receiver yaw. Rim height, vessel radius and the declared finger
envelope bound the clearance; E3 transports stage above the target before
descending. New bundles bind 0.12 m retreat/approach height only to each E3
post-Pour return Place invocation; ordinary Place keeps 0.05 m. Legacy bundles
without that declaration retain their original profile options. Release
acceptance is unchanged. This is geometric/kinematic screening, not global collision planning
or proof of fluid transfer; ambiguous upright geometry still needs qualification.
For level containers with a dominant planar floor, inside placement selects an
arm-side point from the actual floor triangles, eroded by the projected child
footprint and a margin. Holes are retained. Unsupported floor geometry retains
the existing target policy; an oversized footprint on a measured floor fails
explicitly. This geometric preference still requires downstream IK and physical
release/containment qualification.
Opt-in grasp fitting reserves the declared/default robot and object contact
envelopes before selecting a uniform scale. The adaptation report records the
effective per-jaw clearance; the 0.25 minimum scale and source-preservation
boundary remain unchanged.

`task_engine/ontology.py` owns scene-independent task meaning and capability
requirements; `task_engine/interpretation.py` owns strict intent validation and
model guidance. E6 is a prismatic part opened or closed (`target_state=open` or
`closed`, `slideable`); E7 is a revolute door opened (`target_state=open`,
`openable`). Closing a drawer remains E6 regardless of the selected arm.
Closing a hinged door is not represented by either contract.

`task_engine/agent.py` derives SceneRequest requirements from that ontology;
`task_engine/orchestration/scene_adapter.py` checks declared capabilities without
aliasing legacy `pullable`/`pushable` labels. Empty capability metadata remains
unknown, not proof of native joint type. Native joint qualification is deferred
until execution integration.

Persisted candidates must match the current intent and exactly derived scene
request. Regenerate legacy E7 closing candidates and E6/E7 candidates with old
capability declarations; do not silently relabel them. The serialized field
layout is unchanged. Execution admits serial compositions of E1-E6 and calibrated E8/E9;
E7 remains rejected before graph generation and bundle publication. E6 uses the explicit registered
Slide/withdraw/Park recipe in `_task_program/articulation_binding.py` and
`articulation_slide.py`. Its first supported binding is one fixed-base,
single-prismatic, self-contained metre-authored USD with uniform scale and an
unambiguous handle mesh, in one simulation environment. Asset hashes, joint
ownership, and declared limits are checked again at runtime. Public Slide owns
planning; public Task Program and Gym own execution. Joint-target retention is
checked after every recipe call; this is not in-flight contact qualification.

E8 is owned by `_task_program/twist_binding.py` and `twist_runtime.py`: a source-qualified
revolute knob lowers to prepare/bounded Twist chunks/Park through the public `Twist` skill and
ordinary Gym runtime. Each chunk is capped at 5 degrees and the next semantic segment reads
the measured qpos again. Nonzero targets reserve 32 feedback calls; nominal 5-degree coverage
plus the bounded candidate recovery must fit that bound. Verified coarse contact travel permits
unused calls to become one-frame holds; Park still checks release and clearance. The printed setting uses
pointer/label geometry and an asset hash; ordinal settings and joint-limit guesses are rejected.
`twist_semantics.py` resolves asset-owned joint/grip/pointer/label roles through the
[twist source semantics contract](twist-source-semantics.md), not a source-SHA allowlist or
mesh-name heuristics. Source and normalized semantics identities remain separate integrity
checks. Missing semantics are unavailable, while stale or conflicting declarations fail
closed. This supports unseen annotated assets, not visual recognition of arbitrary unlabeled
knobs; hash integrity alone does not supply printed-label semantics.
`twist_adaptation.py` screens source-qualified geometry against the active gripper: compatible
dimensions remain unchanged; mismatches propose a uniform assembly scale with preferred 10 mm
grip depth. The preference is clamped to the uniformly scaled depth/opening feasible interval;
an empty interval still rejects rather than bypassing physical dimensional limits.
The canonical minimum depth is empirically qualified, not inferred from opening-axis
finger thickness. Dimensional candidates still require actual physics acceptance. Audited XY
relayout moves only unreferenced rigid `on(table)` children,
excluding supports/parents. Four axis-separation candidates reuse Scene Engine's 20 mm target
margin and check table bounds and other AABBs; this is not its full layout optimizer.
The formal generator proposes 1 mm total extra lift after preserving the resized support bottom.
`twist_support.py` checks full swept geometry; `twist_attempts.py` rebuilds from the original
PreparedScene at 2/3/4 mm only after support-related startup rejection. Geometric penetration also
gets bounded preflight alternatives. Source USD, mass, drives and contact envelopes remain unchanged;
after verified startup/mass readback, typed table-screen rejection or exhausted zero-contact
candidates may trigger the finite grip-depth search owned by `twist_attempts.py`. Every candidate
is rebuilt from the original PreparedScene; scale and lift never accumulate across attempts.
Other failures, resource stops and unqualified observations do not authorize dimension retries.
E8 retains source scalar mass while converting COM/inertia to the scaled actor
frame before Default materialization, not through runtime native writes. `twist_mass.py` verifies
the source-bound native mass, local COM and full body-frame inertia at sensor configuration,
including the SDK's bounded near-diagonal numerical conversion. When the Lab loader exposes
`body_scale_mass_policy`, the compatibility path may select its USD `fixed_mass` conversion.
Otherwise bundle generation automatically uses GenSim's `twist_mass_source.py`; the planning
field `twist_mass_source` can also explicitly select this path. It generates a self-contained
deployment USD that only scales authored COM and principal inertia, retaining a pristine
source copy, sealed roles and a separately pinned mass-lineage manifest. The deployment removes
the Lab mass-policy field, so the pristine asset is not edited and conversion is not repeated
by Lab. Default-only qualification compares the complete authored layer and rejects external
asset-valued dependencies. Native readback still uses the pristine source and existing
tolerances; the deployment conversion is not guaranteed bitwise equivalent to the compatibility
path. Mass ownership does not change feedback or execution ownership. A raw or repeated scale fails
closed rather than applying another runtime correction. The adaptation audit
separates actual gap from conservative contact-envelope reference. Direct helper calls retain
their conservative default. Canonical startup evidence covers sensor-observed Prepare substeps,
not direct World updates during reset/settling; its `complete` flag means the observation window
was finalized, not that every reset physics step was captured. Invalid/overflowed
observations, motion or table/robot contact fail closed. Final E8 coarse acceptance requires at
least 15 degrees of debounced absolute travel during target contact, release and stability;
Park rechecks all conditions. The 0.5-degree anchor deadband excludes small oscillations,
and contact loss resets the anchor so uncontacted motion cannot earn credit.
Target-setting error and signed directional travel remain diagnostics, not success gates.
Parent contact is recorded as a warning, while released contact and conservative
40 mm world collision-input AABB separation after Park remain required. `twist_geometry.py`
owns source-hash-bound USD collider inputs and URDF finger colliders. Mesh transforms are
baked into each owning link frame before deployment scale is applied exactly once;
observed native rigid poses are not scaled again. Candidate contact proximity uses only
the calibrated grip collider, not the full render-link union. Parent screening and Park
use owned collider inputs, including non-selected articulation bodies. Default render
vertices are not collision inputs. Authored convex-cooking policies remain diagnostic;
these proxies do not certify backend cooked shapes or continuous collision clearance.
Candidates use final tip bites
B/3, 2B/3 and B, where B=min(grasp_depth, grip_depth/2). Their TCP shifts are outward-only,
so no extra inward offset can exceed the requested bite. Depth takes precedence over closed-hand,
arena-frame geometry scores. The shallowest feasible layer selects the original first roll.
Zero-contact recovery retains the same-roll depth order and its dimensional-retry receipt.
Two consecutive contact chunks without new debounced travel may instead choose a shallow
qualified candidate on another parallel-gripper axis, only after measured release and
control-tick joint stability. Axes are equivalent modulo pi, and an explicit fixed roll
never permits other directions. First grasp plus both recovery kinds share three attempts
total; no candidate is reserved twice. Each new candidate is replanned and screened from
fresh observations. Progress and completed goals retain their candidate. Exhaustion is an
explicit failure, not more sampling or relaxed acceptance. Failed IK cannot
silently fall back to an undeclared base pose. The lowerer measures matching jaw aperture and uses invocation-owned typed grasp
overrides; public `Twist` and non-E8 routes remain unchanged.
Reaching the coarse contact threshold makes remaining Twist calls stationary holds; it does
not skip final release/clearance checks or imply precise arrival at the printed setting.
An opted-in feedback controller still records those successful guarded stationary calls;
they require no correction but cannot be mistaken for unowned/missing plans.
Completed holds and waits retain the issued joint target instead of re-anchoring to measured
gravity sag; dynamic-goal and velocity checks remain active.
The E8 factory borrows its existing acceptance sensor for an instance-owned
`twist_geometry_guard.py`. Before ranking, successful candidates must clear the actual
native table bounds plus measured contact offsets across all command waypoints in
approach/reach/retract, covering the active arm and hand source collision inputs.
Unsupported snapshots fail closed; source meshes and paths remain bound to their hashes.
This discrete source-input screen is not continuous clearance or actual servo-tracking proof.
Finalized plans may monitor the fixed root parent instead of the rotating child only when
source shell topology, native hinge/current qpos, public grasp invariance and every extra
rotor collider's swept-envelope separation qualify. Complete convex projection can prove
separation where a conservative AABB cannot; unresolved qualification retains the original
child dependency. Commands, trajectories, recovery/tracking policies and the reach cutoff
are unchanged, and merged dependency cutoffs cannot shorten an existing monitor window.
The public Twist and non-E8 registrations do not construct this guard.
The qualified zero-contact retry may explicitly enable route-owned `twist_feedback.py`;
ordinary compatible routes leave the controller absent. `GenSimActionEngine.start` selects
GenSim's `TwistFeedbackSession` only for the opted-in single-environment, effectless,
empty-hand Twist call. It extends the public `ExecutionSession.tick` and `revise_current`
boundary while retaining the original Runner, SemanticCallExecutor, Gym bridge and transports.
The public phase view supplies scheduling identity only, not a held-object assertion.
Native observations remain measured; continuation seeds require matching native command-target
readback, and replacements retain the original logical/wall deadlines. Correlated effect,
gate or held-guard results fail closed rather than entering a revised plan with stale identity.
`twist_continuation.py` plans only a bounded local suffix, screened through the same geometry
guard before installation. Its joint angle comes from the same-timestamp live SceneSnapshot,
not TaskState's verified symbolic joint accessor or a lowering-time geometry measurement.
Opted-in initial approach/reach retain endpoint alignment holds before checking native feedback;
the holds consume the original deadline and undergo the final velocity/geometry checks.
Closure uses one paired aperture/tip-offset calculation; no native
pose or physical-property setter supplies success. Adaptive closure and convex-cooking
experiments remain unintegrated; this opt-in does not certify dual-pad grasp or printed-setting
convergence. Grasp-axis recovery discards only the old grasp bias and recent correction
samples; pending revisions prohibit switching. Episode counters, per-call deadlines and
audit history remain intact rather than using the episode reset method.
The adapter remains on the v38 contract; only E8/E9 lowerer declarations
change identity. Non-control routes retain their behavior. E8/E9 support multiple serial
occurrences, including repeat operations on one object and interleaved E1-E6 steps.
Each occurrence owns its call alias, sensor, post-policy presets, candidate/feedback state
and sampling epoch. Startup observes its own stable, in-limit initial angle, not a forced zero.
Control post-policies stamp the exact route/source identity and sample window at the
terminal segment. TaskSpec goals and milestones consume those measured receipts through
the canonical runner. E8 coarse contact travel is the selected formal success criterion;
the requested setting and actual target error remain diagnostics, not false joint-near evidence.
Its normative predicate is a semantic event, with a route-owned strict-setting criterion
interface. Repeated events never borrow another occurrence's evidence.
`twist_evidence.json` records measured qpos/contact evidence. `no_twist` and `centroid` are
negative/diagnostic variants only; source assets and shared robot defaults remain unchanged.
E8 alone has a bounded episode-step cap. `orchestration/execution_resources.py` serializes ordinary
E8 subprocesses and monitors available RAM; resource stops retain receipts and cannot be accepted
as task success. `twist_recording.py` streams the original camera frames with unchanged encoding
parameters, committing only verified complete movies. Other task defaults/recorders remain unchanged.

E9 is owned by `_task_program/press_binding.py` and `press_runtime.py`: an
explicitly bound, fixed-root prismatic button lowers to prepare/Press/Park
through the ordinary Task Program and Gym runtime. Generated deployments audit
scaled limits, pre-resize calibrated moving-link mass and an explicit release bias;
source USD and shared physics defaults remain unchanged. No synthetic latch is
inferred from a STOP label; the sensor never commands the button. Grasp TCP is offset to
the closed fingers' leading geometry. Mandatory post-policies verify released
stability, at least 0.05 mm continuous contact-backed travel and 40 mm axial
finger withdrawal. Park requires that episode's verified withdrawal, released
button/housing contact, and 40 mm world mesh-AABB separation between every
finger and target articulation link, not the same axial half-space. This is a
conservative geometric bound, not exact collision-mesh distance. Rebound does not erase the event; full mechanical
travel, device activation and self-latching are not claimed. New standalone and
mixed E8/E9 programs use the same authored verified execution policy. Empty
effect plans project no state claim and need no effect monitor; physical truth
still belongs to the mandatory contact/release/withdrawal post-policies, which
cannot be removed. Adding an unrelated E no longer selects a projected policy.
One environment and unambiguous source-qualified controls are supported. Multiple serial
press events, mixed recipes and repeat objects retain independent contact/release receipts.
The old standalone press probe was removed; qualification uses
the normal Task Engine route. Regenerate E9 bundles after lowerer revisions.
E9 preserves the declared/native button, housing and finger contact envelopes;
it does not override contact or rest offsets. Normal E9 generation sets only
the selected button assembly to unit body scale, preserving XY, rotation and
the original support bottom. When E8 and E9 share an articulation, E8 alone owns
its scale/support adaptation and both controls rebind to that geometry. E9 accepts
the derived mass source only with E8's pinned, fully revalidated source lineage,
original/runtime hashes and deployment scale. It then calibrates drives against
the fixed source mass without a second mass/inertia overlay. Independent E9
controls retain their original scaled-mass calibration. Other objects and source USD remain unchanged.
Table-footprint or object-AABB conflicts reject the deployment. The audit in
press_adaptation.json records original and deployed geometry; mass and release
drive retain their pre-resize calibration. This explicit deployment policy is
not gripper-driven size estimation or proof of original-scene success.
E9 physical configuration identity hashes resource contents rather than their
absolute locations, so nested transaction publication and final-bundle copies
preserve identity without allowing physical-parameter or asset-content drift.
An explicit E9 route may declare 0--10 mm extra commanded press travel.
`planning.press_extra_distance` exposes this as a request-owned override,
fingerprinted in the generated E9 lowerer routes. Explicit zero keeps only the
native button stroke; omission preserves the legacy 10 mm extra travel. The CLI
records and forwards zero without changing other requests, source limits, drives,
contact/release observations or withdrawal acceptance. This changes only public Press's
command distance, not button limits, target state, physics or acceptance. The
extra distance is included in the route fingerprint; the bound is not a
collision-clearance or hardware-safety certificate.

`task_engine/scene/articulation_geometry.py` measures Z-up, metre-authored USD
collision meshes in the native base-link frame, not the default prim's world
frame that runtime reset replaces. Final inspection now measures articulated
geometry. E6 rejects an unmeasured tabletop, more than 2 mm initial table
penetration, or a buried handle before bundle publication and at runtime binding.
An explicit proxy-fit repair returns a new configuration using uniform scale and
a 1 mm placement clearance, levelling only bases within one degree while preserving
yaw; normal loading never silently applies that repair.
Publish repaired exports separately with asset provenance. GenSim assembly and
E6 binding check fresh native joint limits and reapply the same values only when the
public pre-scale cache is stale; it does not enlarge the physical travel range.
Generated E6 bundles apply the calibrated 1 kg mass only to discovered
prismatic moving links. Fixed cabinet/root links retain source-authored mass,
while contact offsets and materials remain asset-owned; this preserves the
Slide grasp response across the legacy loader and Spawn without restoring the
old global articulation-physics override.

E6-open / E1-inside / E6-close composition is owned by
`_task_program/drawer_binding.py`, `drawer_geometry.py`, `drawer_runtime.py`,
and `drawer_curobo.py`.
When drawer routes exist, the adapter creates `DrawerPlacementEngine` but keeps
ordinary wrapper installation unchanged. Explicit drawer planning scopes delegate
transport and Place to the original primitives. The `gen_sim.drawer_transport`
semantic call reuses MoveHeldObject with the effectless drawer contract; ordinary
`simulation.move_held_object` calls retain attachment verification even in the
same program. Drawer motion and later Place/E6 acceptance are not a new transport
retention certificate. Adapter contract v8 requires regenerated bundles with
invocation policies and distinct drawer routing.
Inside targets share E6's asset-qualified part/link identity, never a duplicate
rigid root. Geometry supplies a floor/center hint, not an interior certificate.
The GenSim adapter refreshes the placement target from actual link/object poses
on every Place plan/replan while preserving the declared scene manifest.
No cavity-fit or separate FCL rejection gate is installed. A Pick with an explicit
downstream drawer target defers that reachability screen; a later drawer task on
the same object does not disable ordinary Pick lookahead. Generated bundles insert an
explicit `gen_sim.drawer_transport` before Place: cuRobo plans that free-space
transport, while E6, pickup and final lowering/release retain their existing
planner. The task-owned transport service snapshots live rigid objects and all
articulation links for each plan/replan. Articulated collision components become
separate link-local boxes, preserving openings; the held object is removed only
from the planner world and represented by a conservative sphere cover on a
separate tool-attached collision link. This never attaches objects in physics.
The planner and its payload attachment are released after each planning scope.
This scoped snapshot does not implement concurrent moving-obstacle prediction
or add self-collision checking to the shared cuRobo backend.
Place is planned from the acquired grasp. Composite bundles use the extended
motion sample budget rather than raising joint velocity limits. Ordinary command
planning, effect verification and E6 joint-state acceptance remain.
Execution completion does not prove an object is contained.
Evaluate actual placement/closing from the recorded physical outcome.

## Focused validation

| Change | Tests |
|---|---|
| Portable scene/pose/overwrite contract | `tests/gen_sim/scene_engine/test_scene_core_and_export.py` |
| Grouped layout generation, provenance and playback | `tests/gen_sim/scene_engine/test_randomization_*.py` |
| Edit plans and graph | `tests/gen_sim/scene_engine/test_scene_edit.py`, `test_scene_edit_plan.py`, `test_scene_graph.py` |
| Ingest formats and metadata | `tests/gen_sim/asset_engine/` |
| UI roots/auth/session workflow | `tests/gen_sim/gradio_ui/` |
| Task intent and scene capability contracts | `tests/gen_sim/task_engine/test_agent.py`, `test_interpretation.py`, `orchestration/test_scene_adapter.py` |

Select provider-backed or simulator conversion tests only when that runtime
boundary changes; source/unit checks do not establish remote service quality.
