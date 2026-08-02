# Action Engine v1 Architecture

Action Engine is an independent, compositional successor to
`action_agent_pipeline`. Production code does not import the legacy package;
it reuses only public EmbodiChain robot, simulator, solver, and atomic-action
APIs.

## Data Flow

1. `planning` asks an LLM for a route-free, multi-step semantic Task Agent.
2. `domain` validates scene references, operators, dependencies, and actor
   constraints.
3. `compiler` deterministically lowers semantic skills into one immutable,
   coordinate-free Execution Program.
4. `runtime` schedules the dependency DAG, grounds targets from live simulator
   observations, executes public atomic actions, and checks postconditions.
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
A future skill becomes planner-usable by registering one complete capability
and supplying its usage description rather than adding a router branch.

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

## Robot And Scene Boundary

Generation accepts a Prompt2Scene `gym_export` directory or its
`gym_config.json`. CLI aliases `franka`, `ur5`, and `ur10` resolve to their
dual-arm profiles; explicit `dual_franka`, `dual_ur5`, and `dual_ur10` names are
also accepted. Robot-specific assets and limits remain data-driven rather than
being encoded in planner routes.

Legacy Action Agent artifacts are intentionally not converted. Callers must
use `--overwrite` to regenerate the four canonical v1 artifacts:
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
- Runtime arm selection and grounding are based on live per-environment state.
- A required arm is never silently replaced.
- Failed or inactive vectorized environments preserve their last valid state.
- Rendering and recording never define or mutate executable behavior.
- Video review is the first-stage semantic acceptance gate; automated checks
  validate contracts and execution integrity rather than claiming task success.
