# Task Program Scene Expansion and Expert Trajectory Generation Design

Status: Design proposal, to be implemented in stages. Date: 2026-10-10 (Asia/Shanghai).

Implementation and review tracking: [Issue #748](https://github.com/DexForce/EmbodiChain/issues/748).

This design connects robot workspace analysis, scene expansion that preserves task semantics, and Task Program execution into a shared expert trajectory generation workflow for handwritten Task Programs and GenSim. It retains a dedicated `embodichain.lab.sim.scene_expansion` module for spatial candidate generation. Task Program integrations connect task constraints to execution, while Gym continues to own the physical lifecycle and recording.

Code references: workspace baseline `57334705733d9d3208795f77e6df2442e896a001`; reviewed revision of [PR #729](https://github.com/DexForce/EmbodiChain/pull/729): `b60d90ee1f972ad12c8e81c715faf70e4ee8c6f7`. The PR was still OPEN when checked. The following sections distinguish existing foundations from proposed capabilities; they do not imply that this design is implemented or that the PR has merged.

## 1 Goals and Scope

The goal is to obtain expert trajectories with measured evidence of success by varying scene layouts, adding distractors, and selecting robot deployment poses while preserving task meaning. Expansion has two levels: first generate new scene instances, then optionally augment motion within a validated scene.

Two entry points are supported:

- Handwritten Task Programs: authors provide the program, environment configuration, role bindings, permitted scene variations, and acceptance criteria.
- GenSim: TaskSpec and Task Engine provide task semantics and program generation, while Scene Engine provides the base scene and materialization capabilities.

Shared constraints:

- Preserve task goals, role requirements, required ordering, and success criteria. Free-space motion paths and implementation details not constrained by the task may change.
- Distractors may block the original path; planning and execution validation must run again in the new scene.
- Handwritten environments keep the robot fixed by default. Robot deployment search is added in a later stage; GenSim may jointly search deployment poses and layouts.
- The initial version uses B=1, rigid-body Pick/Place tasks, a fixed robot, and a fixed pool of distractor assets.
- This feature does not integrate handwritten expert functions, fixed action/qpos sequences, or real2sim trajectories. Existing expert APIs and expansion capabilities for other sources remain compatible.
- The initial version does not vary robot models, object sizes, physical parameters, or the physics backend, and does not introduce mobile-base planning during execution.
- Scene expansion configuration remains separate from `program.yaml`; no executable DSL syntax is added.

## 2 Existing Foundations and Proposed Additions

| Existing capability | Current responsibility | Proposed addition |
| --- | --- | --- |
| `Robot.sample_reachable_pose()` and workspace caches | Sample cached qpos and perform FK using the current robot base | Use samples to screen object manipulation poses, layouts, and deployment candidates |
| Workspace spatial randomization functor | Set object poses from sampled end-effector positions | Check actual grasp/placement affordances, support relations, and all required task phases |
| Combined Expansion | Reference families, candidate composition, and scheduling contracts; current scene randomization is limited to cube initial states with fixed drop targets | Accept general scene variant sources while retaining compatibility with existing profiles |
| Task Program integrations and Gym bridge | Compilation, live binding, Atomic Skill execution, and demo integration | Refresh bindings after preparing a scene variant and connect shared execution and acceptance interfaces |
| Task Engine in PR #729 | Scene binding, feasibility assessment, semantic planning, bundles, and execution reports | Rebuild geometry-dependent artifacts for each new layout and connect the shared expansion workflow |
| Workspace feasibility in the PR | Declare runtime checks for phases such as pickup, handover, and placement | Consume actual workspace/IK check results while preserving the distinction between verified and unverified outcomes |

Kinematic reachability, planning success, program completion, physical task success, and data persistence are separate outcomes. Neither workspace caches nor trajectory motion-limit checks replace full task acceptance.

## 3 Module Ownership and Dependency Boundaries

| Module | Responsibilities | Responsibilities outside its ownership |
| --- | --- | --- |
| **New `lab.sim.scene_expansion`** | Scene candidate contracts, layout sampling, distractor placement, deployment candidates, and spatial constraint screening | Parsing TaskSpec/Task Program, model calls, reset/step, full task acceptance, and writing datasets |
| `lab.sim.motion.workspace` | Kinematic caches, end-effector sampling, and manipulability metrics | Scene materialization, collision-world maintenance, and task success decisions |
| `lab.task_program.integrations` | Translate task requirements into expansion inputs; connect scene preparation, binding refresh, program preparation, and acceptance | A separate compiler/runtime or physical execution loop |
| `lab.task_program.semantics` and the existing runtime | Roles, affordances, call/effect contracts, and execution | Scene generation pipelines and asset production |
| Atomic Skills and MotionGenerator | Live action planning, IK, collision planning, and execution policies | Changing task goals or relaxing success criteria |
| Gym host | Environment construction, reset, settling/restoring initial state, step, observations, and demo recording | Task Program schemas, compilers, or layout generation algorithms |
| GenSim Scene Engine | Asset materialization, scene editing, scene revisions, and import/export | The shared action executor and full task acceptance |
| GenSim Task Engine | TaskSpec adaptation, semantic binding validation, and graph/program/bundle regeneration | A separate simulation, recording, or expansion session lifecycle |
| Existing motion expansion and data pipeline | Trajectory candidates, budgets, coverage, receipts, and persistence | Implicit scene changes inside trajectory operators |

### Why Retain a Dedicated scene_expansion Module

Layouts, support relations, distractors, and deployment candidates can be expressed through entity geometry and spatial constraints. Their responsibilities differ from trajectory transformations within a fixed scene. A dedicated module allows both entry points to share algorithms without making handwritten tasks depend on GenSim models, asset generation services, or the full scene editing pipeline.

`scene_expansion` accepts explicit geometry, poses, constraints, and workspace query interfaces. It does not import TaskSpec, Task Program, or Gym. Task roles are translated into entity references and spatial requirements; task semantics must not be inferred from entity names. Integration adapters supply simulation state and robot queries.

Do not create another authoritative Scene, SceneGraph, or runtime registry. Candidates describe changes to an existing scene, while entity identity and execution bindings continue to use existing configurations, Scene Engine, and `SceneRegistry`. Any shared numerical algorithms extracted into `compute` must accept numerical data only and must not depend on `lab`.

## 4 Minimum Public Contracts

The following are proposed interface responsibilities. Field names will be finalized in P0; existing contracts remain with their current owners.

| Contract | Minimum content | Owner |
| --- | --- | --- |
| `SceneExpansionCfg` | Movable entities, sampling regions, distractor pool/count, deployment bounds, seed, and candidate/retry budgets | `scene_expansion`, using `@configclass` |
| `SceneVariant` | Parent scene identity, entity pose changes, added asset references, robot deployment, and candidate seed/provenance | `scene_expansion` |
| Task constraint adapter | Fixed roles, permitted changes, initial-state relations, manipulation pose requirements, and validator references | Task Program / GenSim integrations |
| Workspace query adapter | Sampling, frame transforms, and candidate information for a specified robot and control part | Simulation integrations and existing workspace facilities |
| Scene preparation result | Final entity set, settled initial state, and binding/asset/deployment identities | Host, with upstream materialization input from GenSim |
| Check and execution results | Stage, status, reason, evidence references, actual trajectory, and persistence outcome | Existing validation, execution, and recording contracts, with metadata extensions as needed |

Maintain three identities: task identity, scene instance identity, and execution attempt identity. GenSim references the TaskSpec semantic hash. Handwritten tasks reference explicit semantic declarations and validator versions; a program source hash alone is not a task semantic identity.

Candidate ordinals and seeds support proposal reproduction; settled initial-state snapshots support execution restoration. Initial-state and attempt identities must not depend on temporary physical slot/env_id assignments. Record object/robot state, asset content identities, role bindings, program and integration versions, control period, planning policy, and validator version. A seed alone does not promise bitwise reproduction of physical trajectories across backends or versions.

## 5 Workspace and Scene Expansion Integration Points

### Candidate Sampling

With a fixed robot, sample object poses within permitted support surfaces and spatial regions, then derive actual manipulation targets from object-local affordances. For example:

`T_arena_tcp = T_arena_object × T_object_grasp`

Reachable TCP samples may also be used to derive candidate object poses, but the results must still satisfy support, orientation, and other spatial constraints. Changing a sample's height or orientation invalidates any assumption that its original reachability still applies.

Workspace data guides proposals, supplies IK seeds, and ranks candidates. A pose missing from a sparse cache remains pending verification and is evaluated using IK for the target pose. Consumers must respect the validity mask for invalid samples. Corrupt caches or configuration errors must fail explicitly rather than being ignored.

All transforms must distinguish the local arena, robot/control-part base, and object-local frames. Scene Engine's Y-up editing state is converted to Z-up at the existing export boundary; expansion must not convert already normalized deployment poses again.

### Task Phase Screening

Check grasp, pre-grasp, lift, placement, and retreat targets, as well as motion between phases. Individual arm reachability does not establish a simultaneous handover: compatible poses and collisions between the robots must also be checked. Later articulated tasks require checks across the complete manipulation travel.

Recheck the actual initial state after settling. Whenever held-object or articulation state changes during execution, each plan/replan reads fresh state. An initial workspace check is not a certificate for the entire episode.

### Robot Deployment

P4 performs bounded sampling of base XY/yaw within declared mounting regions and derives height from mounting rules. The relative mounting arrangement of the two arms is preserved by default. Rank candidates by coverage and manipulation quality across all task phases, then run IK, motion planning, and physical validation. Explicit arm assignments and resource constraints take precedence and must not be rewritten to improve acceptance rates.

This selects deployments between episodes; it does not control a mobile base during execution.

### Collision and Binding Refresh

| Change | Required refresh |
| --- | --- |
| Existing object poses | Live state, pose-dependent targets, and planning snapshots; movable entities use collision declarations that support updates |
| Added distractors | Physical entities, relevant registries/providers, and the complete collision world |
| Base deployment changes | FK/IK transforms, robot-relative obstacle poses, and affected sensor state |
| Integration declaration changes | Fingerprints, compiled artifacts that depend on the old declarations, and runtime assembly |

The initial version reconstructs the environment when the entity set changes; it does not require dynamic entity insertion or deletion during execution. Local collision checks in specific GenSim routes do not establish equivalent support for every route. A selected route that lacks the required checks or replanning capability must report that limitation explicitly.

## 6 Preserving Semantics and Constraining Distractors

Fix roles and required relations first, sample layouts second, and add distractors last. Required initial conditions include support, containment, and open/closed articulation states. Goals and invariants during execution follow explicit task requirements.

- Distractors must exist in both the physical scene and the planning collision world. Check support, overlap, stability, and required manipulation space.
- Distractors may block the original path. Reject a candidate if no new path satisfying the task constraints can be found.
- A task referring to "all red blocks" cannot retain its old target set after another red block is added. A moved "leftmost cup" must still satisfy that description. Reject such semantic changes rather than changing task meaning to accept the variant.
- Make the instruction's reference frame explicit. Changing robot deployment must not silently reinterpret relations such as "left" or "in front."
- Do not repair failures by replacing bound objects, narrowing the target set, or lowering thresholds. GenSim model suggestions must pass the original constraint checks.
- If the initial state already satisfies a goal that should be achieved through execution, reject it according to the task's initial-state requirements to avoid invalid expert demonstrations.

## 7 Complete Expert Trajectory Generation Workflow

```text
Handwritten program + scene + constraints    GenSim TaskSpec + scene + program generation
                         \                  /
                          Scene expansion request
                                     |
                scene_expansion candidates <- workspace queries
                                     |
                 Semantic/geometric prechecks and materialization
                                     |
                     Host instantiation, reset, and settling
                                     |
               Actual initial-state checks, binding/artifact refresh
                                     |
             Shared Task Program -> Atomic Skills -> Motion / IK
                                     |
                  Gym execution, observations, trajectory recording
                                     |
                         Measured task acceptance
                           /                  \
          Failure classification          Persist successful trajectory
             and bounded retry                       |
                                          Optional within-scene expansion
                                                     |
                                      Restore, execute, independently validate
```

1. **Prepare inputs**: fix task identity, base scene, permitted changes, validators, and budgets.
2. **Generate candidates**: sample layouts, distractors, and optional deployments, then run inexpensive prechecks.
3. **Prepare the actual scene**: materialize assets, construct/restore the environment, and let it settle. Physics configuration remains owned by the corresponding environment component; expansion profiles do not override physics/backend selection.
4. **Confirm initial state and program**: recheck actual state, refresh scene bindings, collision worlds, and geometry-dependent program artifacts. If this process changes the deployment again, settle, inspect, and capture the initial state again.
5. **Execute and record**: begin recording from the confirmed initial state. The shared Task Program runtime produces actions through Atomic Skills and motion planning; the Gym demo executor calls `env.step()`. The bridge does not advance physics directly, and the compiler does not read live simulation state.
6. **Validate and persist**: preserve separate results for runtime completion, measured task outcome, and dataset acceptance/commit. Send accepted successful trajectories to the existing sink under the shared acceptance rules.
7. **Optionally expand trajectories**: use a successful execution as the scene's reference. Each variant restores the same initial state, executes again, and passes independent validation; it cannot inherit the reference's success result.

### Handwritten Task Program Integration

The author supplies the program/integration, scene expansion configuration, initial-state constraints, and measured acceptance criteria. Object-local affordances and explicit relative targets follow object changes; absolute targets remain fixed.

Program structure may be reused when only poses change and declarations remain unchanged, with a fresh runtime bridge. Update the relevant compiled artifacts when declarations, fingerprints, or geometric constants change. Apply scene changes during reset/preparation; final observations and the first recorded frame must correspond to the actual settled initial state.

### GenSim Integration

Task Engine fixes the TaskSpec and target roles, and Scene Engine produces independent candidate revisions. After materialization, settling, and final inspection, prepare the layout-dependent semantic graph, program, constraints, and bundle.

A temporary deployment may be assembled for simulation preparation, but the actual execution attempt must use artifacts consistent with the final scene and bindings. Recompute layout-dependent choices such as arm selection, handover points, placement targets, and articulation operation order. Record changes introduced by existing geometric/physical adaptation and recheck the final deployment; execution evidence applies only to that deployment.

Preserve the task-route support boundaries of PR #729. Workspace integration does not automatically qualify unsupported E7/E8 or other routes for execution.

## 8 Acceptance, Retries, and Data Coverage

Accepting an expert episode requires a valid initial state, valid program execution, measured terminal state, required intermediate events, and constraints maintained during execution. Missing required observations produce an unverified result. Projected effects cannot replace required physical evidence. Passing discrete collision screening does not establish continuous collision freedom.

Use consistent failure stages: semantic mismatch, geometry/settling failure, IK failure, planning failure, execution failure, acceptance failure, insufficient evidence, and persistence failure. Do not hide configuration, schema, or interface errors as ordinary candidate rejections.

The host coordination layer bounds candidate generation and execution attempts. Retain existing action recovery policies, but count nested recovery toward the total budget. New independent execution attempts start from the saved initial state. Scene Engine, Task Engine, and the execution layer must not independently add unbounded retries or relax requirements.

Reuse existing session, budget, coverage, receipt, and persistence mechanisms instead of creating a competing generation state machine. Scene expansion supplies new scene cases, initial states, and reference families to the existing workflow. The host still owns physical restoration and writes. Record scene-level and trajectory-level attempts separately, and increment persisted coverage only after a persistence receipt confirms the write.

Track the following separately:

- Scene coverage: workspace region, target distance, distractor count/obstruction level, and robot deployment pose.
- Within-scene trajectory coverage: valid motion variants from the same initial state.
- Generation efficiency: rejection rates by stage, planning/execution cost, measured success rate, and confirmed persisted episode count.

Success evidence establishes only that this exact scene and execution passed the declared checks. It does not establish success for nearby positions, other seeds, or the original unadapted scene.

## 9 Three Developer Responsibilities and Collaboration Interfaces

Responsibilities are assigned by role. Map them to GitHub assignees during implementation; no colleague account is assumed.

| Owner | Primary responsibility | Main deliverables |
| --- | --- | --- |
| Shared framework lead (proposal author) | Module architecture, public contracts, workspace and execution integration | `scene_expansion` interfaces; workspace/deployment evaluation adapters; binding/collision refresh; Gym execution/recording integration |
| Task Engine developer | Task semantics, program generation, and task acceptance | Role/constraint adapters; handwritten baseline task and validators; GenSim graph/program/bundle regeneration |
| Scene Engine developer | Shared scene algorithms and the GenSim scene backend | Layout/support/distractor algorithms in `scene_expansion`; asset materialization, revisions, export, and mounting-region candidates |

The shared framework lead owns the `scene_expansion` module; the Scene Engine developer owns its scene algorithms. Shared code does not import the full GenSim pipeline.

In P0, the shared framework lead authors the interface PR. Both colleagues confirm four boundaries: task constraint inputs, scene candidates, scene preparation results, and check/execution results. Agree on frames, entity identities, initial state, budgets, and invalidation rules before implementing separate module files in parallel.

| Cross-module decision | Accountable owner | Collaborator inputs |
| --- | --- | --- |
| Reachability and deployment feasibility | Shared framework lead | Scene Engine supplies candidates; Task Engine supplies manipulation phase requirements |
| Preservation of original task semantics | Task Engine developer | Actual scene entities, spatial relations, and runtime observations |
| Support/layout geometric validity | Scene Engine developer | Shared framework lead supplies physical settling results; geometric and physical conclusions are recorded separately |
| Environment reconstruction and collision refresh | Shared framework lead | Scene Engine supplies added assets and poses |
| Program/bundle invalidation and regeneration | Task Engine developer | Scene and integration identity changes |
| Retry scheduling and persistence | Shared framework lead | Both engines return structured results without rewriting goals |

Merge the interface PR first, then develop the shared runtime, scene algorithms, and task semantics in parallel in separate files. Task Engine provides handwritten task acceptance from P0. Scene Engine provides candidates independent of the full generation pipeline from P1. Neither developer needs to wait until P3 to begin.

## 10 Staged Implementation and Acceptance

| Stage | Shared framework lead | Task Engine developer | Scene Engine developer | Stage acceptance |
| --- | --- | --- | --- | --- |
| P0 Contracts and baseline | Public interfaces, host integration, result recording | Handwritten Pick/Place, role constraints, measured acceptance | Fixed scene and deterministic candidate examples | Behavior is unchanged with expansion disabled; program completion cannot accept a physical failure |
| P1 Layout expansion | Workspace, pose IK, settling/binding refresh | Relative/absolute targets and initial-state semantics | Position/yaw sampling on support surfaces, boundaries, and spacing | Replan and complete tasks across layouts; identify reachable object centers with unreachable manipulation poses |
| P2 Distractors | Environment reconstruction, collision updates, bounded replanning | Referential ambiguity, target-set changes, failure classification | Fixed asset pool and distractor placement | Route around obstacles on the original path; fail within budget for complete blockage; reject semantic changes |
| P3 GenSim integration | Shared execution, acceptance, and recording for both sources | TaskSpec and graph/program/bundle regeneration | Materialization, revisions, final-inspection inputs, and export | Both entry points work; stale bindings, planning worlds, and evidence are not reused incorrectly |
| P4 Deployment search | FK/IK for base candidates and whole-task scoring | Arm selection, phase, and handover constraints | Mounting regions, support conditions, deployment candidates | Correct frames and collisions; preserve relative mounting and explicit resource constraints |
| P5 Coverage and throughput | Trajectory expansion, coverage, compatible batching | Dual-arm/articulated tasks and acceptance | Complex support relations, scene caching, more distractors | Independent trajectory acceptance; correct collision worlds per row; separate coverage and cost reporting |

P0–P3 keep B=1, the robot base, and the physics backend fixed. The first joint milestone is P2: a handwritten Task Program produces trajectories with measured success in new layouts guided by workspace data, including distractors that may block the original path. The second milestone is P3: GenSim uses the same execution and data acceptance path.

Later batching combines only compatible entity sets, initial-state representations, control periods, backends, and recording schemas. Rows with different robot-relative layouts require matching collision worlds. Physical settling must not unintentionally advance other rows that are not participating in reset.

## 11 Validation Plan

| Layer | Required cases |
| --- | --- |
| Pure contracts and candidates | Seed/ordinal reproduction, budget exhaustion, stable entity identity, explicit invalid-configuration failures, invalid workspace sample handling |
| Spatial and semantic checks | Support boundaries, overlap, frame conversion, rechecking modified height/orientation, changes to all/count/leftmost references |
| Reachability and planning | Reachable object center but unreachable grasp; individually reachable arms that collide jointly; replanning around a blocked original path; complete blockage |
| Lifecycle | Settling outside the allowed region; first frame matches actual initial state; initial-state restoration; binding/collision updates after entity or base changes |
| Task acceptance | Program completion with failed goals; incorrect intermediate-event order; missing evidence; later actions invalidating earlier outcomes that must persist |
| Both entry points | Consistent result contracts for equivalent handwritten and GenSim goals; each GenSim variant's bundle matches its final scene |
| Persistence | Failures do not enter the success set; failed commits do not increment confirmed coverage; trajectories and evidence reference the correct scene and attempt |

Add focused production-contract tests at each stage, followed by small physical simulation runs across multiple seeds. Report candidate acceptance rates, measured success rates, and costs separately. Do not add tests whose primary subject is a tutorial; examples exercise the underlying production interfaces.

During implementation, follow the project's `/add-test`, `/pre-commit-check`, and public API documentation workflows, and review affected project context. The current change is a target design only; do not register unimplemented modules as existing runtime capabilities.

## 12 Dependencies and Related Discussions

- [PR #729](https://github.com/DexForce/EmbodiChain/pull/729): foundation for GenSim Task Engine integration. P0–P2 for handwritten Task Programs can proceed first; GenSim integration adapts to the interfaces that ultimately land.
- [Issue #670](https://github.com/DexForce/EmbodiChain/issues/670): existing general expert generation and expansion lifecycle. This proposal adds scene sources that preserve task semantics without replacing session/execution/persistence boundaries.
- [Issue #655](https://github.com/DexForce/EmbodiChain/issues/655): related discussion of task goals, physical evaluation, and environment variation. This proposal focuses on Task Program expert data generated through scene expansion.
- [Issue #709](https://github.com/DexForce/EmbodiChain/issues/709): real2sim trajectory integration, a separate source outside this proposal's scope.

This proposal does not require moving the existing TaskSpec into a public package first or reorganizing existing motion expansion. Share capabilities through adapters, preserve compatibility, and review new implementation work independently in the stages above.
