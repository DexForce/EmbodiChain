# Reviewer Perspectives

Use this reference to select expert roles and supplemental review lenses after
the changed files and their contract neighbors are identified. It complements
`review-matrix.md`: the
matrix describes subsystem contracts, while this file describes the questions
to ask from a user's or operator's point of view.

## Perspective selection

| Change signal | Role ID / lens | Evidence to inspect |
| --- | --- | --- |
| Ownership, dependency direction, cross-subsystem contracts, new extension points | `architecture` | Producers and consumers, layering, registries, ownership, and contract migration |
| Hot paths, tensor batching, allocations, CPU/GPU transfers, synchronization, parallel execution | `hpc` | Call frequency, shapes, complexity, transfer and allocation sites, and contention |
| Simulation/render lifecycle, backend state, scheduling, reset/close or resources | `engine` | Lifecycle owners, backend interfaces, update order, cleanup, and recovery |
| Frames, units, kinematics, dynamics, planning, collision, joint limits, control timing | `robotics` | Transform conventions, numerical assumptions, motion constraints, and physical feasibility |
| Datasets, policies, training/evaluation, rewards, metrics or reproducibility | `ai` | Data provenance, splits, objectives, rollout semantics, and evaluation claims |
| Task Programs, semantic lowering, tools/actions, planning/execution/verification loops | `agentic` | Goal/state contracts, action outcomes, recovery, cancellation, and observability |
| Public Python modules, `__all__`, config classes, YAML/JSON schemas, CLI, task IDs, package data, examples, or migration behavior | `api` | Public facades, strict loaders, callers, registration, docs checker, installed-package paths, and round-trip tests |
| User-controlled files or programs, path resolution, dynamic imports, subprocesses, native extensions, plugins, secrets, or workflows | `security` | Loader and validation boundaries, path joins, execution APIs, dependency loading, workflow permissions, and failure paths |
| Reset/close/cancel, retries, checkpoint/resume, asynchronous workers, distributed execution, or long-running services | `reliability` supplemental lens | Lifecycle owners, cleanup, idempotence, restart state, bounded queues, observability, and worker-failure handling |
| Tests, fixtures, mocks, validation logic, or regression claims | `verification` mandatory pass | Test oracle, old-versus-new behavior, negative and boundary cases, partial batches, cleanup, determinism, and installation-level coverage |

## Public API / developer experience

Review the change as if consuming the installed package without repository
knowledge. Confirm that:

- public imports, `__all__`, signatures, defaults, typing, and lazy facades
  agree with the documented API;
- `configclass` objects, required fields, nested defaults, and serialized
  configuration preserve their contract through construction and round trips;
- CLI output, exit status, task discovery, relative paths, optional
  dependencies, and package data work after wheel installation;
- errors identify the invalid input at the owning boundary and do not leave a
  partially initialized or misleading state;
- breaking changes are intentional, documented, and accompanied by migration
  or clear failure behavior.

Do not report naming or prose preferences unless they make a supported workflow
unreachable, ambiguous, or incompatible.

## Security and trust boundary

Trace data from untrusted configuration, Task Programs, file paths, workflow
inputs, or optional extensions to the first side effect. Confirm strict
decoding, safe path containment, least-privilege execution, secrets
isolation, and fail-closed behavior. Treat unsafe deserialization, unchecked
dynamic execution, path escape, dependency confusion, credential exposure, and
workflow permission expansion as findings only when the changed code makes the
scenario reachable and the impact concrete.

## Supplemental lenses

`reliability` and `verification` have stable role IDs and responsibilities in
`SKILL.md` under Reviewer identity. Use `reliability` when lifecycle or restart
coverage needs a supplemental pass; `verification` remains required for test
or validation changes. Separate agents are optional. Follow the
[expert protocol](expert-protocol.md) to record either a dedicated supplemental
role or a lens within an existing role's scope consistently.
