# Action Engine v1 Architecture

Action Engine owns semantic planning, deterministic compilation, and live
execution. Its `ProgramExecutor` consumes the route-free Execution Program
directly; no legacy graph adapter or second production runtime is involved.

## Data Flow

1. `planning` asks an LLM for a route-free, multi-step semantic Task Agent.
2. `domain` validates scene references, operators, dependencies, and actor
   constraints.
3. `compiler` deterministically lowers semantic skills into one immutable,
   coordinate-free Execution Program.
4. `ProgramExecutor` schedules the Execution Program directly, while
   `ActionGrounder` resolves live geometry, arms, IK, and atomic-action targets.
5. Runtime observations and failures are recorded without mutating the
   persisted program.

This split deliberately gives the LLM ownership of semantic composition while
keeping geometry, arm allocation, collision checking, IK, and trajectory
execution deterministic and inspectable.

## Artifact Ownership

The Task Agent is the only LLM-authored artifact. It contains scene object
identifiers, semantic skills, symbolic goals, actor constraints, and explicit
step dependencies. It cannot contain poses, trajectories, joint arrays, or an
atomic-action sequence.

The Execution Program is the authoritative Seed Graph. It owns the complete
atomic-action DAG, stable node and edge IDs, symbolic target bindings,
resources, named motion policies, postconditions, and optional arm-allocation
groups. Runtime consumes `seed_task_graph.json` directly; there is no second
executable graph format.

Task Agent v1 supports top-level `allocation_groups`. A group names independent
semantic steps that must use distinct arms. The constraint remains symbolic so
runtime can resolve arms against each environment's live state.

`--regenerate` validates the persisted v1 Task Agent and recompiles its Seed
Graph in memory. The regenerated hash must match the hash shared by the gym and
agent configs.

The agent config also snapshots the effective Action Engine runtime policy and
its independent hash. Runtime policy is execution configuration, not semantic
task intent, so it is deliberately excluded from the coordinate-free Execution
Program and its hash.

## Capability Boundary

The capability registry is the shared contract between planning and
compilation. Each planner-visible skill provides an LLM-facing description,
input validation, deterministic semantic expansion, atomic-action lowering,
and runtime postconditions.

The first-stage planner surface exposes exactly five closed-loop semantic
skills:

- `arrange_line`
- `build_stack`
- `place_relative`
- `orient_object`
- `coordinated_transport`

Older `hold_hover`, `press`, and `coordinated_place` definitions remain marked
`planner_visible=False` as internal coverage. They are absent from the LLM
prompt and planner vocabulary, so they are not part of the Action Engine v1
task contract. A future skill becomes planner-usable only after both its
semantic lowering and production runtime support exist; adding prompt
vocabulary alone is not enough.

## Live Execution

The ready-edge scheduler respects explicit dependencies and resource
constraints. It resolves automatic actors independently per vectorized
environment, preserves required-arm constraints, and parallelizes only work
whose dependency and resource contracts allow it.

Target poses, arrangement slots, support heights, orientation policies, IK
results, collision-free paths, and trajectories are computed from live state.
Line slots use current table bounds and object footprints. Shared containers
receive non-overlapping placement slots. Free arrangements may be rematched
only after complete-path planning proves the nominal assignment infeasible.
Contact-sensitive `orient_object` skills keep distinct-arm allocation but run
their pickup-to-release lifecycles serially, so one object is not held aloft
while the other arm completes a separate skill.

`orient_object` persists the support object, XY anchor policy, and upright
local axis instead of asking runtime to infer the task from a generic
"upright" label. Runtime resolves an `auto` local axis from mesh geometry,
never from UID words such as "bottle" or "can", and uses direct-final
transport so an unrequested staging pose is not a mandatory IK constraint.
Final semantic success
uses the simulator's live object pose, including uprightness and XY proximity
to the reset-time anchor. Contact-backed grasp confirmation remains a runtime
limitation and must not be inferred from IK feasibility alone. Upright release
also still needs surface-aware grasp screening: the chosen grasp must admit a
closed-gripper descent to a stable support contact before the hand opens.

## Robot And Scene Boundary

Generation accepts a Prompt2Scene `gym_export` directory or its
`gym_config.json`. CLI aliases `franka`, `ur5`, and `ur10` resolve to their
dual-arm profiles; explicit `dual_franka`, `dual_ur5`, and `dual_ur10` names are
also accepted. Named policies are resolved per canonical robot profile, while
robot-specific assets and limits remain data-driven rather than being encoded
in planner routes. Generation defaults to `dual_ur10` and preserves the
exported scene deterministically. Pose and table-height randomization require
the explicit `--randomize-scene` flag.

Package-owned defaults centralize generation policy and execution-sensitive
runtime policy without moving coordinates into the Seed Graph. Generation
policy covers scene preservation, physics, randomization, environment wiring,
dataset settings, and runner limits and is materialized into
`fast_gym_config.json`. Runtime policy covers scheduling, grounding clearances,
soft arm allocation, grasp sampling, named motion policies, and predicate
fallbacks. It supports canonical robot-profile overrides and is snapshotted
with an independent hash in `agent_config.json`.
Runtime consumes the snapshot rather than silently picking up later package
changes. Existing configs without a snapshot still resolve package defaults;
the earlier arm-selection-only v1 snapshot is migrated onto the complete
profile policy after its original hash is verified.

Persisted legacy Action Agent artifacts are not accepted as Action Engine
inputs. Callers must use `--overwrite` to regenerate the four canonical v1 artifacts:
`task_agent.json`, `seed_task_graph.json`, `agent_config.json`, and
`fast_gym_config.json`.

## Review Artifacts

Generation renders `seed_task_graph.png` from the same validated in-memory
Execution Program serialized as `seed_task_graph.json`. Runtime writes semantic
checkpoints and a final task graph with observed actions, resolved arms,
targets, statuses, and failures. Graphs and records are review outputs and are
never execution inputs. Runtime metadata records the effective policy snapshot
and hash so grounding, grasp, motion, allocation, and verification decisions
remain reproducible after package defaults change.

## Invariants

- LLM output remains semantic and coordinate-free.
- The registry is the single source of planner-visible skill vocabulary.
- Compilation is deterministic and never reads simulator state.
- Production execution uses the Action Engine-owned runtime directly.
- Production code has no dependency on `action_agent_pipeline`.
- Runtime arm selection, grounding, and final postconditions are based on live
  per-environment state.
- A required arm is never silently replaced.
- Failed or inactive vectorized environments preserve their last valid state.
- Rendering and recording never define or mutate executable behavior.
- Video review is the first-stage semantic acceptance gate; automated checks
  validate contracts and execution integrity rather than claiming task success.
