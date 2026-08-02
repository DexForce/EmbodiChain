# Action Engine v1 Architecture

Action Engine is a compositional successor to `action_agent_pipeline`. During
the parity phase, its default `pipeline` backend lowers the route-free
Execution Program through `runtime/pipeline_backend.py` and executes it with
the mature Action Agent graph runtime. The `independent` backend remains
available only as an explicit characterization target. This keeps the new
planner/compiler boundary without silently maintaining two production runtime
implementations.

## Data Flow

1. `planning` asks an LLM for a route-free, multi-step semantic Task Agent.
2. `domain` validates scene references, operators, dependencies, and actor
   constraints.
3. `compiler` deterministically lowers semantic skills into one immutable,
   coordinate-free Execution Program.
4. The default runtime adapter validates a semantics-preserving Seed Graph v5
   view, then the mature pipeline schedules, grounds, executes, and checks it.
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
`planner_visible=False` only as internal characterization coverage while the
independent runtime is stabilized. They are absent from the LLM prompt and
planner vocabulary, so they are not part of the Action Engine v1 task contract.
A future skill becomes planner-usable only after both its semantic lowering and
production runtime support exist; adding prompt vocabulary alone is not enough.

The parity adapter currently requires all steps in one Execution Program to
belong to the same mature runtime route family. Unsupported cross-route
composition, payload transport, and non-upright `orient_object` requests fail
before execution rather than falling back silently. Removing those restrictions
requires extracting the mature controllers from their historical route-level
graph contract, not duplicating them in the independent executor.

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
"upright" label. The adapter resolves an `auto` local axis from runtime mesh
geometry, never from UID words such as "bottle" or "can", and maps upright
in-place execution onto the mature direct-final controller. This avoids making
an unrequested staging pose a mandatory IK constraint. Final semantic success
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
in planner routes. Parity generation defaults to `dual_ur10`, matching the
mature pipeline, and preserves the exported scene deterministically. Pose and
table-height randomization require the explicit `--randomize-scene` flag.

Persisted legacy Action Agent artifacts are not accepted as Action Engine
inputs. The parity conversion is an in-memory runtime view of a validated
Action Engine program. Callers must use `--overwrite` to regenerate the four
canonical v1 artifacts:
`task_agent.json`, `seed_task_graph.json`, `agent_config.json`, and
`fast_gym_config.json`.

## Review Artifacts

Generation renders `seed_task_graph.png` from the same validated in-memory
Execution Program serialized as `seed_task_graph.json`. Runtime writes semantic
checkpoints and a final task graph with observed actions, resolved arms,
targets, statuses, and failures. Graphs and records are review outputs and are
never execution inputs.

## Invariants

- LLM output remains semantic and coordinate-free.
- The registry is the single source of planner-visible skill vocabulary.
- Compilation is deterministic and never reads simulator state.
- Production execution defaults to the mature pipeline backend; selecting the
  independent backend must be explicit.
- Runtime arm selection, grounding, and final postconditions are based on live
  per-environment state.
- A required arm is never silently replaced.
- Failed or inactive vectorized environments preserve their last valid state.
- Rendering and recording never define or mutate executable behavior.
- Video review is the first-stage semantic acceptance gate; automated checks
  validate contracts and execution integrity rather than claiming task success.
