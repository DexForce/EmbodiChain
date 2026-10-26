# Unified Warp IK acceptance — 2026-10-26

Status: **ACCEPTANCE_PASS_WITH_KNOWN_ENVIRONMENT_FAILURES**. Default checkpoint: `5bb718515226772f94c07511e1c50ef9cab0d814`; final receipts explicitly qualify validation, including the preserved full-fast environment failures.

All four robot families use normal `backend="warp"` on CPU/CUDA. Optional pinned official SSIK is an accuracy oracle only. Success requires original-URDF/PK FK at returned float32 precision, ≤1e-5 m/rad, valid limits; no automatic fallback.
## Time & Memory

Actual-Robot rows show the largest observed per-arm/variant median and p99 within each family; they do not construct a synthetic timing distribution. Each constituent uses 10,000 completed changing-target calls. Bulk uses5 warmups + 30 completed calls. Memory columns are MiB; — means unmeasured in that timing scope.

| Family | Device / scope | cost_time_ms | p99_ms | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb | Evidence / gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Aloha Mini | cpu / actual Robot | 1.692845 | 2.212617 | — | — | — | P1 carried; numerical source bound; old receipt exact source identity |
| Aloha Mini | cuda / actual Robot | 0.799350 | 1.201516 | — | — | — | P1 carried; numerical source bound; old receipt exact source identity |
| Marvin (bare + gripper) | cpu / actual Robot | 1.845102 | 2.388336 | — | — | — | P1 carried; numerical source bound; old receipt exact source identity |
| Marvin (bare + gripper) | cuda / actual Robot | 2.401724 | 2.972328 | — | — | — | P1 carried; numerical source bound; old receipt exact source identity |
| ARX5 | cpu / actual Robot | 0.917244 | 1.066118 | — | — | — | 4748 carried, not retimed at d73 |
| ARX5 | cuda / actual Robot | 1.264906 | 2.875361 | — | — | — | 4748 carried, not retimed at d73 |
| Rizon4S | cpu / actual Robot | 2.069650 | 2.623597 | — | — | — | QUALIFIED |
| Rizon4S | cuda / actual Robot | 3.847243 | 4.505248 | — | — | — | QUALIFIED |
| ARX5 | CUDA resident N=1000 | 8.362725 | 8.701131 | — | — | — | original 445.000ms / 53.212x; 30 calls |
| ARX5 | CUDA resident N=10000 | 83.739096 | 86.325272 | — | — | — | original 5029.118ms / 60.057x; 30 calls |
| Rizon4S | CUDA resident N=1000 | 168.771495 | 172.371570 | — | — | — | original 430.093ms / 2.548x; 30 calls |
| Rizon4S | CUDA resident N=10000 | 1638.354010 | 1641.732907 | — | — | — | original 4996.131ms / 3.049x; 30 calls |
| Aloha Mini | fresh driver scratch | — | — | — | — | — | 34.250000MiB /128MiB; P1 fresh-PID carried; driver/local stack included |
| Marvin (bare + gripper) | fresh driver scratch | — | — | — | — | — | 80.000000MiB /128MiB; P1 fresh-PID carried; driver/local stack included |
| ARX5 | fresh driver scratch | — | — | — | — | — | 34.000000MiB /128MiB; 4748 fresh-PID carried, caller outputs conservatively included |
| Rizon4S | fresh driver scratch (own-PID NVML) | — | — | — | 96.000000 | 708.000000 | 95.723434MiB /128MiB; Current 7bbc fresh-PID; larger driver growth less final output; not Torch-only |

Actual paired CPU/CUDA p99: Aloha 3.870/2.145ms; Marvin bare 3.194/5.403ms; gripper 4.086/5.433ms (all≤8ms). Full P1 standalone/control and bulk matrices remain bound to the prior receipt, not duplicated here.

ARX new d73 resident throughput is 53.212x/60.057x at 1k/10k; its CPU/CUDA controls and driver-memory certificates are carried from 4748 with unchanged numerical code/TCP bindings. Rizon uses fresh 7bbc formal certificates; its original-Pytorch denominators are older isolated same-cohort measurements with independently verified source/input identity.

Rizon fresh unbounded/control/actual-Robot p99: CPU 3.925/1.747/2.624ms; CUDA 4.672/4.016/4.505ms (each 10k). ARX carried standalone-control p99: CPU 0.292/CUDA2.544ms at4748. These certificates have their own completion/quality/isolation records.

## Success & Other Metrics

| Family | Devices | Wide (10k/model) | Continuous (20k/model) | Unreachable false positives | Mandatory gates | Qualification |
| --- | --- | --- | --- | --- | --- | --- |
| Aloha Mini | CPU + CUDA | 100.000% | 100.000% | 0 / 500 each model | PASS | Carried P1; both arms, CPU/CUDA |
| Marvin (bare + gripper) | CPU + CUDA | 100.000% | L98.345% / R97.185% | 0 / 500 each model | PASS | Carried P1; both variants, CPU/CUDA; radial-shell certificate |
| ARX5 | CPU + CUDA | 100.000% | 95.225% | 0 / 500 each model | PASS | Current d73 wide/bulk; P2 full trajectory + independent step-feasibility certificate carried |
| Rizon4S | CPU + CUDA | 99.990% | 100.000% | 0 / 500 each model | PASS | Current 7bbc CPU/CUDA; wide8659 known finite-pool miss |

Aloha full paths: 100%. Marvin left 98.345%/right 97.185% retain radial-shell certificates and both gripper variants. ARX 95.225% retains the actual-model initial-seed/step infeasibility certificate. Rizon wide 99.99% retains classified finite-pool miss8659; its 20k paths are100%. All marked successes satisfy the original 1e-5 checks.

Official SSIK small precision alignment: fresh Rizon accepted reference 68/108, Warp 87/108; both zero false successes. Seven unchanged models are carried with numerical/AST source bindings. These mixed positive/negative small cohorts and differing finite pools are not a coverage ranking or a performance certificate.

## Leaderboard

Rank is only within the same family and validation scope; CPU/CUDA Warp tie in validated coverage. Historical baseline denominators and precision-only oracle entries are unranked.

| Within-family rank | Family / scope | Implementation | Wide validated | 20k paths | Qualification |
| --- | --- | --- | --- | --- | --- |
| 1 (tie) | Aloha Mini | Warp CPU + CUDA | 100.000% | 100.000% | Eligible |
| 1 (tie) | Marvin (bare + gripper) | Warp CPU + CUDA | 100.000% | L98.345%/R97.185% | Eligible |
| 1 (tie) | ARX5 | Warp CPU + CUDA | 100.000% | 95.225% | Eligible |
| 1 (tie) | Rizon4S | Warp CPU + CUDA | 99.990% | 100.000% | Eligible |
| — | ARX5 / Rizon4S | Original Pytorch CUDA | ARX strict0%; Rizon see receipt | — | Timing denominators; original 5e-4 policy retained |
| — | All eight arm models | Official SSIK CPU | Small precision-only cohort | — | Oracle only; no runtime/performance claim |

Original Pytorch retains30 samples/500 iterations/5e-4 internal tolerances. ARX strict0% refers only to external1e-5 acceptance, not a claim that the original solver violates its looser contract.

ARX is an explicit 9-slot bounded numerical hybrid; Rizon uses 25 locked samples/600 algebraic slots plus bounded scalar pencil-root correction. Neither claims a complete general 6R/7R solution manifold. P99 values are measured acceptance results, not hard-real-time guarantees.

Prior report SHA256: `c959555f1937e6b7c336f6df4ed7af0b314ab440360c7f3b2183722c16890f02`. Prior immutable raw index: `edf7d3f47c7490c64ccb4c946cd6dd577275f22bccd1c0d3c93ef17c590f6884`. Current raw archive binding: `4d10f8c75f9c035efbffab6503128de28b4bc3ccc6517ea2c3b12f3b75641763`. Complete vectors/cohorts/source/process/isolation evidence resolve through the two indexes and bound receipts; the public report is excluded from its own raw checkpoint.

Final validation bindings: wheel receipt supplied; original full CPU-fast10010passed/12failed/31skipped preserved;186focused and33 root-refinement checks, API3212/3212, context42, Black26.10.1 and Sphinx-W receipts are carried in the JSON.

The full CPU fast run is **10010 passed /12 failed /31 skipped**, not an all-pass result. Current global Newton1.7 rejects scalar gravity; the same11 tests pass with compatible private Newton1.6, without claiming a stored prior global version. The existing upstream installed-benchmark test removes all editable finders, including external editable dexsim-engine; our first temporary wheel probe had the same filter. External dependency paths were repaired only in private consumer environments, leaving global packages and production tests unchanged. Independent private-environment attribution/probe results are recorded separately; none may be summed into a fabricated10022-pass full run.

Machine-readable evidence: [ssik_unified_warp_acceptance_20261026.json](ssik_unified_warp_acceptance_20261026.json).

Production IK focused checks:186 passed; root-refinement compute/public checks:33 passed; independent full datasets and mandatory formal controls/bulk/memory passed. API3212/3212, context42 and Sphinx-W passed as reported by root; final receipt binding remains explicit. This scope does not claim unrelated renderer/physics tests passed in the changed global environment.
