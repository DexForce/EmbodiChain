# Reviewer Perspectives

Use this reference to select review lenses after the changed files and their
contract neighbors are identified. It complements `review-matrix.md`: the
matrix describes subsystem contracts, while this file describes the questions
to ask from a user's or operator's point of view.

## Perspective selection

| Change signal | Select this perspective | Evidence to inspect |
| --- | --- | --- |
| Public Python modules, `__all__`, config classes, YAML/JSON schemas, CLI, task IDs, package data, examples, or migration behavior | Public API / developer experience | Public facades, strict loaders, callers, registration, docs checker, installed-package paths, and round-trip tests |
| User-controlled files or programs, path resolution, dynamic imports, subprocesses, native extensions, plugins, secrets, or workflows | Security and trust boundary | Loader and validation boundaries, path joins, execution APIs, dependency loading, workflow permissions, and failure paths |
| Reset/close/cancel, retries, checkpoint/resume, asynchronous workers, distributed execution, or long-running services | Reliability / operations (conditional lens) | Lifecycle owners, cleanup, idempotence, restart state, bounded queues, observability, and worker-failure handling |
| Tests, fixtures, mocks, validation logic, or regression claims | Verification / test quality (mandatory pass, not a separate identity) | Test oracle, old-versus-new behavior, negative and boundary cases, partial batches, cleanup, determinism, and installation-level coverage |

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

## Conditional perspectives

Reliability and verification are review lenses rather than mandatory reviewer
identities. Use reliability for lifecycle and restart behavior that is not
fully explained by the engine or agentic passes. Use verification for every
test or validation change, checking that tests observe the contract and would
fail on the relevant old behavior rather than merely mirroring the
implementation.
