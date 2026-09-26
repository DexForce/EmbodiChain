# Benchmark framework

The shared technical-report core lives in `scripts/benchmark/core/`;
offline aggregation, comparison and report/figure outputs live in
`scripts/benchmark/reporting/`.
The camera pilot is the first consumer. Other benchmark domains retain their
existing owners and can adopt these helpers incrementally.

## Entry points and ownership

| Request | Start here |
|---|---|
| Public command | `embodichain/cli/main.py` → `scripts/benchmark/__main__.py` |
| Experiment definition and lifecycle states | `scripts/benchmark/core/contracts.py` |
| Matrix expansion and budget conservation | `scripts/benchmark/core/planning.py` |
| Run identity and worker-result validation | `scripts/benchmark/core/records.py` |
| Repeats, subprocesses, timeout/interruption and status ledger | `scripts/benchmark/core/execution.py` |
| Stage, continuous and attempt measurement | `scripts/benchmark/core/measurement.py` |
| Standard artifacts, JSONL idempotency and config hashes | `scripts/benchmark/core/artifacts.py` |
| Software/source/device provenance | `scripts/benchmark/core/provenance.py` |
| Grouped metrics, missing values, denominators and uncertainty | `scripts/benchmark/reporting/aggregation.py` |
| Comparison eligibility and paired ratios | `scripts/benchmark/reporting/comparison.py` |
| Legacy conversion and report tables/figures | `scripts/benchmark/reporting/compat.py`, `tables.py`, `figures.py` |
| Offline generic report | `scripts/benchmark/reporting/report.py` |
| Camera workload and runtime selection | `scripts/benchmark/rendering/workload.py`, `run_benchmark.py` |
| Camera process and platform APIs | `scripts/benchmark/rendering/worker.py`, `backends/` |
| Camera-specific report layout | `scripts/benchmark/rendering/report.py` |
| G-03 case, attempt, receipt and fixture runner | `scripts/benchmark/expert_generation/` |

## Boundaries

Core and shared reporting import only the standard library. They must not
import domain workers, NumPy, Torch or simulators. Domain configuration and
worker modules may use their own installed dependencies. The camera CLI's
report-only path stays ahead of simulation-dependent configuration imports.

The launcher freezes `RunSpec` values and runs platforms serially in separate
processes. Core writes the complete plan and not-run records before starting.
Timeout and interruption terminate/reap the isolated worker group. Invalid
worker JSON is retained, not silently interpreted as success. JSON writes are
validated and atomically replaced; existing experiments cannot be overwritten.

`ExperimentDefinition` freezes the parameter matrix, quality protocol,
comparison invariants and `Budget`. `BudgetLedger.reserve_run()` atomically
consumes one run and one attempt, so retries cannot evade the fixed budget.
The standard directory contains definition/config/assets, raw JSONL, metrics,
quality and artifact-index files. `ArtifactStore` rejects conflicting duplicate
record identities while allowing an identical retry submission to be replayed.

Domain callbacks own completion synchronization and value validation. Core
excludes warm-up, measures callback latency, and separately retains the full
loop window. A host-return timer alone is not proof of GPU completion.

Reports keep execution status independent of quality/task/data outcomes.
Aggregation separates workload fingerprints and excludes failed measurements
without dropping failures from scheduled counts. Missing values remain null.
The domain declares comparison invariants and supplies quality qualification;
core never substitutes image non-emptiness for quality certification.

Aggregate metric records include a definition version, SI unit, population,
aggregation rule, denominator, uncertainty summary, missing reasons and source
run IDs. The generic report/figure helpers are standard-library only; SVG/CSV
outputs are generated from already frozen run IDs and do not initialize a GPU.

Physical stepping, reset, task predicates and production commit receipts
remain in existing domain/host owners. Shared benchmark execution does not
provide a second generation coordinator or dataset sink.

The G-03 benchmark runner consumes those owner-provided outcomes through an
executor callback. Its accepted-yield rule requires completed execution,
measured validation, task success and a confirmed persistence receipt. It keeps
attempt failures, duplicate commits and budget-exhausted not-run rows visible;
the fixture under `scripts/benchmark/expert_generation/` is deliberately
simulator-free and does not replace #670's coordinator or `GenerationSession`.

## Camera-specific cautions

The pilot measures render-to-blocking-host RGB, not pure GPU rendering or RL
throughput. Its two adapters share procedural geometry and camera parameters;
lighting/quality equivalence is not established. Isaac Lab's render generation
must advance before reading the camera's ProxyArray buffer when physics does
not step. EmbodiChain uses its public camera-group render and camera look-at
APIs. See `scripts/benchmark/rendering/README.md` for exact supported runtime
and reproduction commands.

## Validation

Use `tests/benchmark/core/`, `tests/benchmark/reporting/`, the camera pilot tests
under `tests/benchmark/rendering/`, and `tests/test_main.py` for CLI regressions.
The scalar-core tests prove reuse without images or simulators; import tests
use Python without site-packages. Real GPU camera runs qualify the installed
backends separately. Tutorial scripts are references, not production test
subjects.
