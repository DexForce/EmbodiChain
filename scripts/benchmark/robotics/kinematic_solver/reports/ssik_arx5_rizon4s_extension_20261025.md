# ssik_arx5_rizon4s_extension Benchmark Report

Generated at: 2026-10-25T22:03:24

This run uses the latest-main ARX5 TCP: Rz(90 degrees) with an 85 mm tool offset, verified against the unchanged prepared source-chain artifact. Earlier identity-TCP measurements are excluded.

ARX5 and Flexiv Rizon4S are available through explicitly selected CPU reference artifacts. Both retain Pytorch defaults; their actual geometry is unsupported by the current Warp backend. This report certifies no large-sample, continuous-control or GPU performance gate.

Hardware/runtime: AMD Ryzen 9 9950X 16-Core Processor; NVIDIA GeForce RTX 5090; Python3.11.14, Torch2.7.1+cu128, CUDA runtime12.8, Warp1.17.0, PK0.10.0, NumPy2.2.6, SciPy1.17.1. The recorded run uses1 Torch intra-op and32 inter-op threads, with OMP_NUM_THREADS/MKL_NUM_THREADS/OPENBLAS_NUM_THREADS unset. No other benchmark or pytest process was active during the final tiny timing run.

[Compact metrics and full provenance](ssik_arx5_rizon4s_extension_20261025.json). Large arrays/raw timing samples remain in the ignored [raw evidence directory](../../../../../outputs/benchmarks/ssik_arx5_rizon4s_extension_20261025/). The [original acceptance report](ssik_acceptance_20261025.md) remains unchanged.

## Time & Memory

| model | impl | component | scope | cost_time_ms | sample_size | p95_ms | p99_ms | max_ms | repeats | cpu_rss_delta_mb | torch_peak_increment_mb | warp_pool_delta_mb | warp_pool_peak_mb | solver_buffer_mb | attributed_scratch_mb | scratch_gate | queries_per_second | sustained_queries_per_second |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arx5 | ssik_reference_cpu | ssik_ik | initialization | 27.96347299590707 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| arx5 | ssik_reference_cpu | ssik_ik | first_call_including_jit | 3.4380729775875807 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| arx5 | ssik_reference_cpu | ssik_ik | resident | 2.231088001281023 | 1 | 2.2770204115659 | 2.281103292480111 | 2.282124012708664 | 3 | 0.0 | 0.0 | unavailable | unavailable | unavailable | unavailable | UNVALIDATED: allocation ownership not established for this backend | 448.2118138889324 | 466.26416097739843 |
| flexiv_rizon4s | ssik_reference_cpu | ssik_ik | initialization | 795.7582348026335 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| flexiv_rizon4s | ssik_reference_cpu | ssik_ik | first_call_including_jit | 21.239561960101128 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| flexiv_rizon4s | ssik_reference_cpu | ssik_ik | resident | 19.443938974291086 | 1 | 19.509220588952303 | 19.51502339914441 | 19.516474101692438 | 3 | 0.0 | 0.0 | unavailable | unavailable | unavailable | unavailable | UNVALIDATED: allocation ownership not established for this backend | 51.429908380303345 | 51.3344923742001 |

## Success & Other Metrics

| model | impl | component | scenario | sample_size | success_rate | reported_success_rate | false_successes | translation_err_mm | rotation_err_deg | max_position_m | max_rotation_rad | max_accepted_joint_step_rad |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arx5 | ssik_reference_cpu | ssik_ik | wide | 32 | 1.0 | 1.0 | 0 | 5.898555884345644e-05 | 1.804623503596113e-05 | 1.4971030451804604e-07 | 8.249021974729435e-07 |  |
| arx5 | ssik_reference_cpu | ssik_ik | boundary | 12 | 1.0 | 1.0 | 0 | 8.66473717718062e-05 | 2.0639035166403084e-05 | 1.9823810533992064e-07 | 6.527922659049395e-07 |  |
| arx5 | ssik_reference_cpu | ssik_ik | singular | 8 | 1.0 | 1.0 | 0 | 0.0001691497309946796 | 2.9058033678834947e-05 | 3.8545048858174283e-07 | 1.1548181906750423e-06 |  |
| arx5 | ssik_reference_cpu | ssik_ik | unreachable | 8 | 0.0 | 0.0 | 0 | 0.0 | 0.0 | None | None |  |
| arx5 | ssik_reference_cpu | ssik_ik | trajectory | 48 | 0.6875 | 0.6875 | 0 | 1.3396563684980608e-05 | 4.278319172931917e-06 | 4.164105256225239e-08 | 1.3789322434928787e-07 | 0.006581366062164307 |
| flexiv_rizon4s | ssik_reference_cpu | ssik_ik | wide | 32 | 1.0 | 1.0 | 0 | 8.210301972492912e-05 | 1.8506826888472325e-05 | 1.930107001073597e-07 | 5.268154377400659e-07 |  |
| flexiv_rizon4s | ssik_reference_cpu | ssik_ik | boundary | 12 | 1.0 | 1.0 | 0 | 6.947820631127414e-05 | 2.006942458829897e-05 | 1.4224334780188003e-07 | 4.854960074934813e-07 |  |
| flexiv_rizon4s | ssik_reference_cpu | ssik_ik | singular | 8 | 1.0 | 1.0 | 0 | 0.00010238851079147651 | 2.170168801532577e-05 | 1.4098532193500562e-07 | 4.943993864611233e-07 |  |
| flexiv_rizon4s | ssik_reference_cpu | ssik_ik | unreachable | 8 | 0.0 | 0.0 | 0 | 0.0 | 0.0 | None | None |  |
| flexiv_rizon4s | ssik_reference_cpu | ssik_ik | trajectory | 48 | 0.3333333333333333 | 0.3333333333333333 | 0 | 0.0001229963994458149 | 1.6943685258524346e-05 | 1.338531889175174e-07 | 3.5192120001087654e-07 | 0.022531449794769287 |

## Leaderboard

| rank | algorithm | overall_success_rate | avg_translation_err_mm | avg_rotation_err_deg |
| --- | --- | --- | --- | --- |
| 1 | ssik_reference_cpu:arx5 | 85.00% | 0.000056 | 0.000014 |
| 2 | ssik_reference_cpu:flexiv_rizon4s | 68.00% | 0.000092 | 0.000019 |

## Notes

- Latest-main synchronization: ARX5 uses the exact origin/main preset at 8cb2b53537e8c83b42a79c166e2777336ca0a2d0, including its Rz(90 degrees) orientation and +85 mm TCP offset. The actual source-chain fingerprints still match both prepared artifacts; only the ARX5 TCP-bearing workload/model fingerprint changes. Previous identity-TCP rows and their report/provenance are historical and excluded, preserved in the sibling raw directory with suffix_pre_main_sync.
- Scope: explicit CPU reference diagnostics for the actual ARX5 X5A and Flexiv Rizon4S presets. Both robot defaults remain Pytorch. The standard SSIK model default selection remains the original six Aloha/Marvin chains; these two models require explicit --ssik-models selection.
- Neither actual chain satisfies the strict supported Warp geometry. Model extraction classifies both as unsupported and skips Warp without fallback. This is not a CUDA implementation or a Warp acceptance result.
- Both models require explicitly supplied offline source artifacts and adjacent manifests. ARX5 uses the actual rounded-axis model and its generated general-6R RR artifact. Rizon4S explicitly reuses the pinned prebuilt joint-lock model after prefix verification and a fixed end-frame correction. Its wrapper evaluates only the 16 verified cached RR slices, disables upstream HP secondary rescue, and allows off-grid seeds to reorder that cached grid only. No symbolic derivation occurs at IK runtime.
- Fixed seed 2531; per model 32 uniform wide targets, 12 near-boundary targets, 8 verified singular/near-singular targets, 8 guaranteed unreachable targets and 3 paths of 16 frames (smooth joint/line/circle). These small sets provide diagnostic evidence only. Cartesian path reachability is not guaranteed globally.
- Both models accept all 52 wide/boundary/singular targets and reject all 8 unreachable controls, with zero false successes. Under the 0.05 rad previous-posture step bound and full-path fixed-initial-seed candidate semantics, ARX5 completes 33/48 frames and Rizon4S 16/48. These continuity results do not qualify either default for migration.
- The Rizon4S generic 7R artifact uses upstream default 16 locked-joint samples (joint 3/zero-based index 2); SSIKSolverCfg.num_samples=128 configures SRS swivel sampling and does not change this prebuilt joint-lock search. Finite samples do not represent the continuous manifold.
- Every accepted output is checked independently with the actual original float64 PK chain and its TCP at 1e-5 m/1e-5 rad. The raw source limit oracle allows 1e-7 rad for float32 representation; production acceptance enforces effective float32 limits strictly. Reported errors describe accepted outputs only.
- Resident N1 times use five excluded warmups and only three completed calls per model. Their p95/p99 values are order statistics of three samples, not control-latency certificates. The medians shown in the timing table cannot establish formal p99 or throughput gates. Upstream candidate trace logging is suppressed only during the benchmark and the previous logger level is restored.
- Compilation/initialization and first-call scopes are separate. The CPU general-reference scratch allocation ownership is unvalidated; Torch-only counters are not presented as total native NumPy/SciPy memory. GPU, 1000/10000-batch speedups,10,000-call control tails and full-size correctness workloads remain UNMEASURED.
- Pytorch comparison baselines were intentionally skipped in this small extension run; no coverage or performance superiority over the defaults is claimed. Leaderboard entries rank only these identical smoke scopes by independently validated attempted-target success, excluding unreachable controls.
- The 15 new reference-model guard tests passed, including forbidden cold RR/HP builders and off-grid seeds. The public CLI and helper tests passed 13 cases (36 unrelated cases deselected), including typed artifact forwarding, preservation of the six-model defaults, geometric Warp skips, artifact parse validation, missing-artifact skips and logger restoration. Five changed Python files are Black 26.10.1 clean; git diff --check passed.
- Earlier Rizon smoke rows before the cached-grid-only guard are superseded and excluded from this report; their raw evidence is retained in the sibling directory with suffix_superseded_pre_guard. The earlier first-phase acceptance report is frozen and unchanged. Its Aloha/Marvin conclusions do not transfer to these general/near-geometric models. The new extension source/model/artifact fingerprints are recorded separately.
- Geometry arx5: unsupported; actual-chain fingerprint 6fb7bc72811c6cc813cdaeac7e6dc71990bffb334401d76f0862010cf4dbdaef.
- Skipped arx5/warp/cpu: actual-chain geometry is unsupported; Warp requires three-parallel 6R or strict SRS 7R. No numerical fallback or acceptance result is substituted.
- Geometry flexiv_rizon4s: unsupported; actual-chain fingerprint a526344e32701b7e5c4eec409d11329227407a6f1cc0f70b4bbad67cc5cedb79.
- Skipped flexiv_rizon4s/warp/cpu: actual-chain geometry is unsupported; Warp requires three-parallel 6R or strict SRS 7R. No numerical fallback or acceptance result is substituted.
- Gate arx5/cpu: UNVALIDATED (smoke workload; acceptance requires full workload and repetitions).
- Gate flexiv_rizon4s/cpu: UNVALIDATED (smoke workload; acceptance requires full workload and repetitions).

## Reproduction

Run from the repository root, using the pinned SSIK revision `4109147f37a7ca3e2516c83853aca583612783f3`:

```bash
export PYTHONPATH="$PWD:$PWD/extensions"
PYTHON=/home/dex/miniconda3/envs/open/bin/python
# Install the pinned [ssik-reference] extra, or append its pinned checkout src
# directory to PYTHONPATH (the measurement used /tmp/ssik-analysis-20261025/src).
"$PYTHON" -m embodichain.cli.benchmark robotics-kinematic-solver --solvers ssik --ssik-smoke --ssik-models arx5 flexiv_rizon4s --ssik-devices cpu --ssik-skip-baseline --ssik-reference-artifacts arx5="$PWD/extensions/embodichain_robots/_ssik_artifacts/arx5.py" flexiv_rizon4s="$PWD/extensions/embodichain_robots/_ssik_artifacts/flexiv_rizon4s.py" --ssik-batch-sizes 1 --ssik-artifact-dir /tmp/ssik-arx5-rizon4s-extension-main-sync
```
