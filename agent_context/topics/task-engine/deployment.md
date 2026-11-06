# Task Engine deployment and evidence contracts

`embodichain.gen_sim.task_engine` owns task understanding, candidate declarations,
normative TaskSpec generation, semantic graph contracts and workflow state. Its
instruction/candidate import path does not initialize Lab or a simulator.

## Detailed source map

| Request | Owner under `gen_sim/task_engine/` |
|---|---|
| Instruction intent and caller boundary | `interpretation.py` |
| Supported task intentions and relations | `ontology.py` |
| Candidate, draft, scene request and success declarations | `contracts.py`, `agent.py` |
| Candidate-to-normative-spec conversion | `task_spec.py` |
| Grounded semantic recipes, occurrences and source lineage | `semantic_planner.py`, `_semantic_recipes.py` |
| Canonical Semantic Call graph and declaration coverage | `semantic_graph.py`, `semantic_graph_validation.py` |
| Role bindings and scene/report declarations | `orchestration/contracts.py` |
| Task Engine defaults, profile selection and independent outcome reporting | `config.py`, `defaults.yaml`, `experiment_policy.py` |
| Workflow state and request ownership | `state_machine.py`, `workflow_contracts.py` |
| Existing-scene workflow, candidate/attempt scheduling and child lifecycle | `workflow.py`, `scene_backend.py`, `orchestration/execution_resources.py` |
| CLI workflow/execute/validate/display commands and run allocation | `cli.py`, `run_directory.py`; unified command registration in `embodichain/cli/main.py` |
| Stable run publication and interruption audits | `orchestration/workflow_artifacts.py`; sealed bundles retain original paths and bytes |
| Frozen component import and drift checks | `physical_input.py`; Data enumerates files and Scene validates physical declarations |
| Source scene formats and detached planner/runtime views | `orchestration/source_scene.py`; shared Scene `source_physics.py` owns physical defaults/attributes/collider assembly, and physical math/control qualification stay in Scene |
| Resolve recipe action options and effect-monitor declarations | `action_profile_declarations.py`; explicit resources and alias-owned option inputs |
| Resolve semantic scene entities and affordances | `scene_binding_declarations.py`; source placement/mount math remains shared Scene-owned |
| Resolve stage-qualified relative placement targets | `placement_declarations.py`; shared Scene/Compute own geometry, semantic call order owns reservations |
| Resolve named stability conditions | `stability_declarations.py`; public Task Program configs with explicit support-sensor and resource bindings |
| Execute one initialized sealed bundle episode | `execution.py`; public demo executor plus runtime evidence session, caller-owned reset/teardown |
| Original execution-report schema and read-only HTML graph display | `reporting.py`, `semantic_graph_html.py`; original bundled UI assets, identity-matched recorded data and recomputable coverage |
| Resolve shared control operations and contact sensors | `control_declarations.py`; canonical model roles from `embodiment_declarations.py`, no sensor implementation subclasses |
| Resolve complete prismatic operation declarations | `prismatic_declarations.py`; shared source inspection and standard joint policy, no planning implementation |
| Assemble semantic segments and terminal checks | `semantic_program.py`; consumes explicit control-operation bindings and named stability conditions |
| Bind owner-resolved checks to semantic segments | `program_declarations.py`; no geometry or runtime strategy implementation |
| Original bundle generation entry | `task_program_bundle.py`; pure preparation and source graph refinement through shared geometry owners, followed by immutable publication |
| Bind owner declarations and publish generated bundles | `bundle_assembly.py`; compile occurrence identity, complete preflight, then atomic publication |
| Preflight, seal, publish and load configured bundles | `bundle_deployment.py`, `bundle_identity.py`, `bundle_io.py` |
| Load a bundle for the current CLI/workflow package | `bundle_loading.py`; independent current implementation and actual external file hashing |

TaskSpec protocol identity, predicate/temporal evaluation and caching remain in
[Task specification](../task-spec/task-spec.md). Task Engine derives declarations
but does not define that protocol or silently re-sign a changed specification.
Semantic graph calls use the public Task Program language decoder, and graph
payloads exclude low-level goals, controllers, joint trajectories and planner
parameters. Model calls are supplied through the instruction-caller interface;
fixed task declarations can be validated without invoking a model.

Atomic Skills, motion search, grasp geometry, native observations and execution
services belong to their shared Lab/GraspKit owners. Task Engine must not define
skill, planner, engine or contact-sensor implementation subclasses. Shared
compute, simulation, Task Program and GraspKit code must not import Task Engine.
Source bundle generation explicitly declares terminal release/attachment effect
contracts. Shared integration factories own those boundaries and their full
terminal verification; generic Task Program phase/retention defaults stay with
the shared owners. This is a sealed deployment declaration, not a runtime rewrite
based on task or object identities.
Runtime policies and deployment composition use the public
[Task Program](../task-programs/task-programs.md) contracts.

Use the existing candidate/spec-generation cases and `test_layer_boundaries.py`
under `tests/gen_sim/task_engine/`, plus `tests/gen_sim/task_spec/`, for this
semantic front end. Provider calls and physical simulation are not needed for
these checks. Physical program effects and strict TaskSpec acceptance remain
separate from a workflow stage being marked complete.

The shared-runtime bundle envelope is version 5 (`shared/v3`) and rejects predecessor adapter
contracts before runtime composition. `prepare_configured_bundle()` consumes
canonical declaration files, preflights public owner contracts, verifies the
source graph's exact program projection, and seals identity without writing.
`write_bundle_atomic()` publishes the prepared snapshot to a fresh directory.
`load_configured_bundle()` verifies bytes/current inputs first, then recomposes
and compares integration, registration, program and TaskSpec identities. The
physical owner consumes component snapshots rather than reopening component
files. No path imports `_task_program` or converts its constraints file.
When the original API supplies no TaskSpec, `bundle_spec.py` emits the closed
`task_spec.absent/v1` marker in the still-required hashed `task_spec.json` file.
The loaded specification is None; this is executable program-only scope with
strict normative acceptance unavailable, never an invented empty successful spec.
Present TaskSpec formats and semantic identity validation remain unchanged.

`prepare_generated_task_program_bundle()` returns a pure snapshot for original
scene/task inputs. `orchestration/coordinator.py` retains bounded semantic
candidate selection and includes final audits before atomic publication;
`scene/` owns semantic scene contracts while Lab Scene and Compute own geometric
measurements. Published bundles are immutable, including during workflow retries
and HTML rendering. Runtime reports and displays belong to a fresh run directory.
Physical qualification remains a separate contract.

`load_current_task_program_bundle()` reads the versioned envelope first, hashes
actual external files before and after preflight. It separately enumerates the
runtime asset paths and transitive local dependencies from sealed physical and
service declarations: deleting a file name from a self-consistent identity cannot
hide a required mesh, robot source or referenced asset. Robot presets resolve
through the public config-only Gym owner; URDF assembly destinations are outputs.
Configured adapter asset providers supply model/side dependencies, and Data owns
mesh metadata plus composed USD asset and layer references. Dependency-set changes,
including a newly present source sidecar, reject before runtime assembly. It requires the installed
package's implementation revisions, including the optional `nvidia-curobo`
distribution version read without importing the SDK (`unavailable` when absent).
It never adopts envelope values as the
independent expected inputs. Explicit qualification-owned revision mappings remain
available only through the configured loader; old runtime contracts reject before
runtime imports and require regeneration.

Current-runtime qualification accepts the generic `RobotCfg`'s fully materialized
sources. A concrete model configuration additionally needs its own
`DeclaredAssetFileProvider`; source-only enumeration cannot attest hidden model
inputs. Standard native-Marvin keeps its concrete selector and audited owner
provider. This does not change ordinary Lab configuration or the explicit
qualification-owned loader's caller-input contract.

Every sealed v4 bundle includes `source_input_provenance.json` with the closed
`gen_sim.source-input-provenance/v1` schema. Groups contain owner-to-path/SHA
mappings only: generation records physical inputs, hand services and additional
inputs; low-level application assembly uses an explicit application group by
default. Overlap requires identical hashes, and the union must exactly equal the
identity's external inputs. Group names never select runtime behavior. This
source inventory supplements independent runtime dependency enumeration and
cannot replace it. Supplemental artifacts cannot override the reserved manifest.

The current loader binds its independently checked inputs to the loaded value.
`assert_current_inputs()` rejects later input/dependency/implementation drift;
configuration and runtime construction check it before and after their boundary,
without reopening sealed component files. The public episode entry also checks
before selecting segments, after preparation and before returning final evidence.
The worker checks after reset and after a planning probe before recording its
result. Both direct bundle publication and the sealed orchestration transaction
check an explicit source-owner validator before atomic no-replace publication;
both reject sealed ancestors before creating staging directories and again before
publication, including resolved symbolic-link aliases. Unsealed audit replacement
keeps its separate archive behavior but cannot relocate a sealed descendant.
Generation checks every physical/hand/additional source file against its original
digest and independently verifies runtime dependencies again after preflight and
before publication, never resealing drift. Predecessor v3/shared-v1 bundles require regeneration
from frozen declarations before any Lab import; TaskSpec/program versions stay
unchanged. A new source identity cannot reuse an old execution certificate.

Workflow input tasks run concurrently, but the source single scene attempt and
bounded action-attempt order remain unchanged. Seeds advance only for physical
action attempts; E8 resource interruptions retry with the same seed and never
select an interrupted result. The exact successful resource directory is chosen
by the parent attempt record, not by a report-supplied path. Final acceptance
delegates to `verify_bundle_execution_report()` with that exact directory and the
current loaded bundle, then applies required semantic-step coverage.
The worker exit code must also agree with the verified report (0 for all accepted
rows, 2 otherwise). Abnormal termination or incomplete finalization cannot make
the parent workflow succeed; original program/effect/TaskSpec facts stay stored.

Linux child cancellation stops only the execution's owned session, including
workers remaining after the parent exits, and closes its captured output stream.
Ordinary execution keeps its five-second termination grace; E8 resource
supervision retains the original signal order and timing budgets.

Run roots stay at a stable path while their individually sealed bundles publish.
Selecting `final/bundle` atomically copies the original verified snapshot, without
rewriting declarations or identities. Cancellation after preparation retains the
sealed attempt and an interruption audit. Existing runs containing sealed bundles
cannot be overwritten. Reports, trajectory/video outputs and HTML remain outside
sealed declarations, including when a destination is reached through a link.

A loaded bundle can `configure_environment()` from its sealed source snapshot.
This uses the public Gym parser with physical component snapshots and inline
Task Program declarations; it does not reopen mutable component files or create
a simulator. Runtime callers create the environment, reset, select segments,
and borrow `env.active_task_program_bridge` for episode-bound evidence before
consuming the first segment. There is no need to patch bridge factories.

Frozen physical inputs select existing scene, embodiment and execution-policy
components before semantic planning. Their files, nested snapshot and dependency
inventory are checked without resealing drift. `physical_input.py` does not load
old trajectories or adapt old runtime contracts. `part_catalog.py` labels shared
qualified part descriptors; physical geometry/pose calculations remain in Scene.

The source recipe generator declares native grip evidence explicitly for every
source joint-evidence profile. It preserves the ordered grasp-affordance object
scope (background is excluded), model-owned pads and source query capacities.
Optional driver-open release retains the original profile's open vector and
tolerance; model-owned driver roles select indices. The old embodiment software
evidence declaration is moved into integration, never duplicated. Historical
snapshots missing hand geometry consume the canonical embodiment calibration as
an additional hashed external input; simulation/control values remain frozen.
Coordinated recipes select the public paired-grasp service and cargo capture by
explicit ordered hand targets, preserving source candidate budgets. Static
preflight/generation does not establish physical success or evidence qualification.

Frozen sampler annotations and other externally frozen runtime inputs can be
supplied as `generate_task_program_bundle(external_input_digests=...)`. Absolute
file paths and SHA-256 values are checked before sealing and again before atomic
publication, cannot override physical/calibration ownership, and become part of
the bundle identity. Default calls retain their existing input set.


Pour recipe declarations preserve three source modes: omitted geometry revision
uses static Scene/Compute staging plus occurrence-bound `jaw_pour`; explicit
revisions 1/2 select transport/retained factories and the five-role projection
service. The source witness and complete independent recipe remain required for
those opt-ins. Ordinary E3 transport selects a 6 cm lift and retained attachment,
without alignment recovery or Cartesian mode. Successful static preflight is
not evidence that an action has executed or that fluid transfer succeeded.

`alignment_declarations.py` emits constrained Pick, held-alignment and released
clearance routes. Shared action options and named occurrence policies own the
actual algorithms. Staging predecessors come from exact graph dependencies with
matching object/resource, never runtime scanning or instance-name inference.
Constrained Pick aliases receive explicit source objects during normal assembly.
Scene/Compute own authored-axis approach and release-plane calculations; the
declaration resolver tracks only preceding semantic alignment and selected
upright-support groups. Ordinary E2 public calls retain their existing input
defaults; other constrained routes still reject absent geometry declarations.
Stability creates sensors only for conditions that require physical support;
axis-only upright conditions do not gain a new support-contact success rule.
Explicit `auto` selection uses the original semantic profile resolver. A profile
whose service declarations are not wired yet fails before publication rather
than inheriting default behavior. Full continuation and physical qualification
remain separate from successful frozen-declaration preflight.

`drawer_declarations.py` binds an inside affordance to the already inspected E6
part and semantic link identity. Scene owns authored link transforms and region
inspection; Compute owns placement geometry. Vertical preservation remains a
per-occurrence declaration even when two placements share one container route.
Closing a drawer may bind multiple payload rows to the same close occurrence;
shared runtime validates every payload. These declarations do not replace missing
original composite task inputs for fixed-input physical qualification.

The exact first serial occurrence receives final velocity qualification in
`mask_only` mode and disables whole-call retries for fully failed complete plans.
This is the source initial-plan contract; later occurrences keep existing
retiming/retry choices. It is not selected by E-type or task identity, and does
not bypass bounded recovery inside the complete first request.

`continuation_declarations.py` selects the original bounded Pick/retained-align
release capabilities by exact adjacent semantic occurrences and effective policy.
The shared integration owns future grounding, prediction and evidence boundaries.
Drawer acquisitions instead defer their entire downstream target tuple to the
explicit drawer service. Direct drawer transport uses the motion_gen owner and
cannot inherit generic Cartesian/lift/alignment modes; inside Place and closing
Slide receive exact payload/physical-part bindings, including every close payload.

Native Pick permissions retain the original global revision and explicit
source-node override forms. Their generated closing bindings use exact compiled
occurrences; ordinary on-release openings bind the final Place identity and read
its already-grounded target at runtime. Source alias phase checks are explicitly
selected with `verify_pick_phases` and do not alter ordinary built-in defaults.

The evidence session consumes the registration's optional cargo observation
selection through Lab's passive collector. E5 declarations select carriers and
intentionally manipulated exclusions and require trajectory recording; physical
capture and complete-prefix checks stay in Lab/Compute. Reports retain the
separate `cargo_observation` facts and acquisition errors without rewriting
program success or TaskSpec judgments. Missing channels remain unavailable.

Generated Pick declarations select ordinary approach qualification only for the
ordinary grasp path. Pour and end-source sampling retain their dedicated shared
contracts; native revisions preserve source opening behavior. Whole-request
90-second budgets follow structural verified Pick/release pairs and retained Pour
ownership, independently of prospective-recovery eligibility. Other sampled
calls retain their original unbounded wall budget.

Normal `PreparedScene` physical preparation is routed through
`source_physical_input.py`. It composes explicit Asset/Scene normalization and
optional fit, model-owned deployment calibration, source execution policy and
Lab-owned reset/cadence declarations into a fresh checked physical snapshot.
`asset_preparation.py` selects acquired semantic identities and projects the
returned physical mappings; geometric fitting and normalization live in Scene
Engine/Compute. Frozen physical snapshots bypass the preparation stage. Additional
control-specific source scale/mass preparation must precede snapshot creation.

The public bundle generator accepts ordinary `PreparedScene` through the same
physical preparation owner. Derived meshes/models are staged in the separate
`.task_engine_assets` sibling cache before pure declarations and atomic output
publication. Existing `FrozenPhysicalInput` follows its original bypass and
rejects physical adaptation. Non-frozen Press/Twist still require their separate
control scale/mass owner before publication; the basic source path cannot silently
skip those operations.


`experiment_declarations.py` retains source profile permissions as pure semantic
and physical declarations. Auto selects default/contact/composition by the
existing recipe contract, contact preserves slow closing and its twelve-sample
increment, and composition leaves unrelated defaults unchanged. Generated native
permissions merge without overriding explicit revisions. Exact idle clear/restore
pairs use public MoveJoints services; post-handover end-grip release is selected
only for the adjacent receiver and original source grip provenance. Frozen physics
remains unchanged; the non-frozen physical owner applies its separate overlays.

Explicit inside world-yaw permission is resolved by
`continuation_declarations.resolve_inside_yaw_declarations` against the original
ordinary verified Pick/release pair, rigid subject/parent and the 12 cm container
release contract. The generated preset selects existing `inside_world_yaw`
recovery and declares the exact late native landing subject/container. The shared
Place service retains heading order, footprint reservations and runtime rejection;
inside placement alone does not grant yaw freedom.

rejects physical adaptation. Non-frozen Press/Twist delegate explicit scale/support/drive and derived-mass preparation
to the shared source-control owners before snapshot creation. Those preparations
retain their source receipts and independently hashed input files.

Finite-count E1 placement/orientation, E2 orientation and E6 part projection verify complete covered
subject/group membership and retain each member's actual recorded boundary.
Placement also retains each member's release occurrence and the shared actual
final boundary. The shared TaskSpec composite contracts own aggregation,
cardinality checks and sparse historical event semantics; see
[Task specification](../task-spec/task-spec.md).

`runtime_bindings.BundleEvidenceProjector` projects sealed coverage, compiled
occurrences and named policies into the actual episode evidence session. It
never acquires physics or weakens proof requirements; explicit missing/unsupported
proofs remain unavailable. Other multi-member predicate families require a
complete shared proof contract before this projector can qualify that scope.

Current-run reports use `reporting.build_bundle_execution_report` and
`verify_bundle_execution_report` to preserve distinct program, physical effect,
strict TaskSpec and cargo outcomes. The evidence session creates a fresh run UUID
and binds the configured recorded-contact collector before execution. A workflow
checks the sealed bundle/program identity and its exact attempt output directory;
it never upgrades a historical report to a new runtime certificate.

The session's read-only `binding_declarations` copies every effective verified
binding through its canonical owner serializer, including unreached recorded
events, original hashes and live acquisition identities. The public episode
return keeps this snapshot alongside its existing measurements.
`to_raw_evidence()` exports `task_engine.raw_evidence/v1`, linked to bundle/run,
episode/row scope and the original execution metadata. The worker writes
`raw_evidence.json` before publishing the physical report and retains the existing
`recorded_measurements.json`. Missing measurements and projection diagnostics
remain missing/unavailable; export does not observe, re-project, resign or
evaluate. Offline re-evaluation consumes these declarations and packets with the
sealed original TaskSpec/program, rather than reconstructing live identities.


Explicit `measured_rigid_calls` are resolved into shared Pick/inside options and
exact Lab occurrence bindings. The source 90-second whole-request budget remains
attached to each selected call. Candidate generation, native observation,
preshape fitting, occupied-floor projection and recovery are implemented by their
shared Atomic/Scene/Compute owners; bundle generation remains declaration-only.

`resolve_released_hand_declarations` retains the original adjacent release,
clearance and one-or-two Park source binding. Every role has an exact occurrence
identity; Park order and distinct resources are preserved and must include the
cleanup resource. Invalid release/object/resource chains reject during bundle
preflight. The shared Task Program and Atomic owners perform future grounding,
qualification and bounded cleanup recovery.

The generator emits those cleanup bindings under the closed public
`planning_services.released_hand_continuations` field and selects the typed
`released_clearance` / `released_park` recovery presets for exactly their bound
occurrences. Release verification and suffix grounding remain shared preflight
and runtime contracts; no task-number strategy override is introduced.

`native_pick_exploration` keeps the original explicit non-frozen dual-UR10
scope. The generator validates the public boolean/robot selection, then selects
`PickUpOptions.native_scene_trial` for ordinary acquisition. Shared Atomic owns
command checks and records the scene precheck as uncertified; effect tracking and
TaskSpec acceptance remain independent. Frozen physical inputs reject adaptation.


Normal generation calls `PreparedScene.assert_source_assets()` before drawer or
other source-geometry refinement; physical preparation repeats the same check
before baking. A changed file requires a freshly prepared scene;
the generator does not overwrite the earlier asset identity with a new digest.
Inputs with unchanged declarations/files retain their serialization and hashes.

Routed ``simulation.move_held_object`` declarations keep their direct source path;
late-bound scene targets and single absolute poses do not select alignment's
batched-target lift. ``gen_sim.align_held`` owns that explicit lift permission.


`source_graph_declarations.py` preserves the original pre-bundle semantic target
refinement order over unrefined inputs: upright targets, carrier lift, untouched
upright Pick, exact source aliases and first-use handover staging. Pour return
heights have their separate later projection. Geometry/frame/radius observation
lives in public Scene adapters and float32 pose/height arithmetic in Compute;
these transformations never construct skills or grant physical acceptance.



Normal preparation verifies any existing `PreparedScene.asset_hashes` before
reading/baking source geometry. A changed file requires a freshly prepared scene;
the generator does not overwrite the earlier asset identity with a new digest.
Inputs with unchanged declarations/files retain their serialization and hashes.

`drawer_graph_declarations.py` preserves composite drawer preparation before
physical assembly: canonical part IDs, inserted/idempotent transport calls and
source alternate closing resource declarations. It reuses shared prismatic
source discovery and the existing semantic recipe owner; ambiguous parts or
conflicting transport declarations reject before program compilation.

The public prepare pipeline runs drawer part/transport canonicalization before
physical source assembly, source target refinement before experiment permission
selection, and Pour return/close-resource projection before coverage and program
compilation. This also applies the original idempotent refinements to frozen
inputs. Source held-world routes include upright and handover staging alongside
Pour routes, using ordinary public configured transport factories.

Control programs retain the original articulation-root wait entity and reused
`turned`/`pressed` named post preset on the last action and cleanup. The source
declaration owner records effective condition selections separately; after pure
compilation each becomes an explicit program-fingerprint/post-occurrence binding
in Lab runtime services. Physical child-link identity and terminal clearance
remain control-owner declarations. Runtime evidence resolves these same public
bindings before selecting terminal measurement scope. Program/TaskSpec formats,
hold grids and success criteria do not change.

Runtime manipulation evidence uses `evidence_binding.TaskEngineManipulationBinding`
to seal a Lab observation declaration and a distinct TaskSpec acceptance binding.
Projection reads the complete frozen owner condition or explicit caller criteria,
never a runtime monitor's resolved parameters. Their identities and predicate,
program, episode and row scopes must agree. `runtime_evidence` passes only the Lab
declaration to collectors, archives both declarations and original measurements,
then `evaluate_task_engine_evidence` normalizes facts before TaskSpec evaluation.
The raw archive contract is `task_engine.raw_evidence/v2`; predecessor runtime
bundles must be regenerated from frozen declarations. Existing historical results
retain their original identities and are not qualifications of changed code.
