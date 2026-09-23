# Benchmark framework

The shared technical-report core lives in `scripts/benchmark/core/`;
offline aggregation and comparison rules live in `scripts/benchmark/reporting/`.
The camera pilot is the first consumer. Other benchmark domains retain their
existing owners and can adopt these helpers incrementally.

## Entry points and ownership

| Request | Start here |
|---|---|
| Public command | `embodichain/cli/main.py` → `scripts/benchmark/__main__.py` |
| Run identity and worker-result validation | `scripts/benchmark/core/records.py` |
| Repeats, subprocesses, timeout/interruption and status ledger | `scripts/benchmark/core/execution.py` |
| Synchronous operation/window timing | `scripts/benchmark/core/measurement.py` |
| JSON artifacts, effective config and workload hashes | `scripts/benchmark/core/artifacts.py` |
| Software/source/device provenance | `scripts/benchmark/core/provenance.py` |
| Grouped metrics, missing values and units | `scripts/benchmark/reporting/aggregation.py` |
| Comparison eligibility | `scripts/benchmark/reporting/comparison.py` |
| Camera workload and runtime selection | `scripts/benchmark/rendering/workload.py`, `run_benchmark.py` |
| Camera process and platform APIs | `scripts/benchmark/rendering/worker.py`, `backends/` |
| Camera-specific report layout | `scripts/benchmark/rendering/report.py` |

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

Domain callbacks own completion synchronization and value validation. Core
excludes warm-up, measures callback latency, and separately retains the full
loop window. A host-return timer alone is not proof of GPU completion.

Reports keep execution status independent of quality/task/data outcomes.
Aggregation separates workload fingerprints and excludes failed measurements
without dropping failures from scheduled counts. Missing values remain null.
The domain declares comparison invariants and supplies quality qualification;
core never substitutes image non-emptiness for quality certification.

Physical stepping, reset, task predicates and production commit receipts
remain in existing domain/host owners. Shared benchmark execution does not
provide a second generation coordinator or dataset sink.

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
