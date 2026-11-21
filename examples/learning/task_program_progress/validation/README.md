# Native Task Program reward acceptance

The control targets were collected from the calibrated planning Task Program and replayed through ordinary `JointPositionAction` terms in a configuration-owned Task Program reward environment. No policy model or optimizer was constructed. The `execution: policy` setting declares external action ownership.

| Case | Control steps | Weighted stage events (row 0) | Outcome sum (row 0) | Maximum scalar error |
|---|---:|---|---:|---:|
| success | 893 | [0.5, 0.5] | 10 | 8.9406967e-08 |
| hold | 64 | [0.0, 0.0] | 0 | 1.70985e-12 |
| truncated | 600 | [0.5, 0.0] | 0 | 8.5532665e-08 |
| failure | 1 | [0.0, 0.0] | -10 | 0 |

All four cases passed across 1,559 native env.step calls and 3,118 environment reward samples, including one post-reset transition. Repeated-read error was zero; maximum discounted-shaping identity error was 1.4968932e-08.

Each milestone pays 0.5 once; accepted task success pays 10 once. Failure pays -10. A time limit retains the successor potential and pays no success/failure bonus. True termination applies the absorbing-potential correction, so the final total reward includes both event bonuses and a negative shaping term.

The second environment holds its initial joint targets. It receives no milestone or success events; the failure fixture deliberately fails both rows. Resetting only row 0 preserves row 1's next reward, with absolute error 9.8225428e-13.

The scalar oracle independently evaluates reward composition using measured skill potentials. It is not an independent geometry or skill-potential-formula oracle. All actual, expected and repeated values are checked for finiteness.

Physical hold/stability windows now inspect every physics substep and restart after interrupted or invalid evidence. Transient final-substep contact, missing samples, and object or independently actuated hand writes after sampling have production regressions. Invalid rows restart their windows even when a diagnostic host continues without automatic reset; unaffected rows retain their history. Equivalent call/stage aliases suppress the same automatic milestone. Missing standalone composition and nonfinite metric targets reject during configuration.

The related Python 3.12 regression run passed **384 tests** with two capability-dependent skips and one deselected test, including 12 API-documentation checker tests. Python 3.11 passed **308 focused tests**, with two skips and one deselected test. Coverage includes the real scheduled-stereo replay path (restored disparity 8 instead of stale 2, preserved randomized extrinsics and valid reward initialization), unmatched direct-Python acceptance selectors, checkpoint defaults, reward/profile arithmetic and project context. API documentation coverage is **5199/5199**; the entire checkout passed Black 26.10.1. Native qualification passed supported/floating/empty-grip fixtures and rejected both negative fixtures when invalid measurements were injected, retaining per-row diagnostic flags before exiting nonzero. Eight trace-validity and 21 qualification/preview probes passed. The earlier insufficient stable-window probe and report-identity checks remain part of the qualification contract. These checks require no policy training.

[Annotated native reward preview](task_program_rewards.mp4) displays the actual reward ledger. The last frame is held for approximately one second. [Machine-readable results](results.json) retain controller/configuration identities and numerical results, plus the preview digest, raw-video digest, reward-trace digest and 447 captured frames. Preview generation verifies the raw footage, trace hash and frame indices; a run without video does not reuse older footage. Full per-step artifacts are generated locally rather than committed.

Physics-substep observation adds acquisition work. On the shared AMD EPYC 9K65 host, Python 3.12.16 / Torch 2.11.0 / DexSim 0.6.0rc2, two CPU-physics arenas, four substeps per 0.04 s control step and one Torch thread, the final success replay plus recording took 201.82 s (single run, excluding environment construction). This is an end-to-end fixture timing, with rendering and scalar auditing; it is not a throughput benchmark or a GPU performance claim. Observer storage retains bounded per-row state and the last sample, rather than the trajectory. Large-batch/GPU throughput remains unqualified.

Reproduce with the commands in the [example guide](../README.md#task-program-reward-acceptance-without-training). CPU physics and the project simulator dependencies are required; use a native display or Xvfb for the preview.
