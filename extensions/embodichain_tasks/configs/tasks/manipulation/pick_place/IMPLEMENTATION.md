# ODS PickPlace implementation and experiment evidence

Date: 2026-10-17, Asia/Shanghai. Engineering interfaces and executable comparisons
are implemented. **The student has not passed the one-episode repeatability gate.**
No 32/160-episode learning benchmark, reserved 100-state final evaluation or ODS
learning-speedup claim is reported. The [machine-readable result](IMPLEMENTATION_RESULTS.json)
retains counts and local evidence paths. Earlier runs remain in EXPERIMENTS.md.

## Implemented behavior

- Measured 51-feature geometry with MLP-256 and a GRU-128 fallback; seven pose/
  gripper requests, controller-space loss, frozen statistics and independent
  student inference. Zero quaternion history padding is made finite before
  conversion; masked prefixes receive no padded supervision.
- Reusable `RelativeEefPoseAction`, arena rotation-vector composition, current-state
  targets, and explicit IK diagnostics. Optional DLS uses the actual solver root,
  validated joint order, a device-safe TCP Jacobian shift, 0.05 damping, a 0.1
  maximum joint increment and live joint limits. Its success flag means finite
  local control, not achieved pose convergence.
- ODS supports explicitly ordered raw policy-request recording independently of
  the legacy expert joint schema. Frame identities and acceptance timestamps are
  published under the sample-copy lock. Invalid language/provenance IDs remain
  -1 in history padding. Worker failure still terminates sampling immediately.
- One/four-reset qualification stages, immutable checkpoints, baseline-inclusive
  selection, resumed optimizer/random state, fresh ODS source on resume, trace
  capture, independent Viser evaluation and TensorBoard.
- Offline-full, offline-32, frozen and online delivery; schema/content-validated
  numeric corpora; causal recurrent offline sampling and exact MLP batch replay.
  Serial refresh is explicitly unmatched. Exact recurrent manifest replay is
  rejected rather than silently approximated.
- Efficiency/timeline reports, censored endpoints, physical GPU allocation
  accounting, unique supervised frames, sampled PSS and comparison figures.
  Development T80 is reserved for full development panels; tiny reset diagnostics
  do not certify it. Unsupported device-wide peaks, actual kernel overlap, data
  age, eviction waste and attributed I/O counters remain null.

## Physical experiments

The task, reset XY range, 600-step horizon and strict lift/place/open/withdraw/
settle predicate were retained. No expert takeover, stage input, pretrained policy,
external training dataset or feedback-directed collection was introduced.

| Controller / learner | Expert qualification | Independent one-reset verification | Measured lift |
| --- | --- | --- | --- |
| Robot solver / MLP | 40/40 expert scenes | 0/10 complete | 2/10 |
| Robot solver / geometry GRU | Same solver teacher | 0/10 complete | 1/10 |
| Local DLS / MLP | 38/40 expert scenes | 0/10 complete | 10/10 |
| Local DLS / geometry GRU | Same DLS teacher | 0/10 complete | 5/10 |

Each student run used 2,000 updates and one live-generated accepted demonstration.
MLP batches contained 256 targets; GRU batches contained 64 sequences with 24
supervised targets, so these are qualification attempts, not compute-matched
model ablations. The selected checkpoint was verified on ten repeated training
reset states. Periodic checks used that reset once per checkpoint. DLS/GRU
checkpoints 250, 500 and 750 completed those isolated checks, but the selected
checkpoint then failed all ten verification rollouts. Those isolated successes
must not be presented as reliable task competence.

DLS meets the original expert minimum of 38/40. Qualification failures at seeds
10019 and 10030 are retained in its report. The original solver's 40/40 test
recorded zero IK failures but joint-target discontinuities up to 1.758 rad.

## What the failure investigation established

1. On the solver/MLP source episode, all 331 valid frames received supervision;
   512,000 sampled targets exposed each frame more than 1,400 times. Training-state
   translation-command error was 0.125 mm median and 0.750 mm at p95; gripper sign
   matched all 331 states. Small regression error did not imply physical success.
2. Source actions replayed successfully through ordinary `env.step()` without a
   planner. Student initial observations matched the source bitwise. However,
   target pose differences of 1.56–3.36 mm corresponded to solver joint-target
   differences of 0.88–0.99 rad before contact. This motivated bounded local DLS.
3. In the DLS/GRU verification, initial observations and actions matched the
   successful development trace bitwise. First-step cube pose/velocity then
   differed while robot state initially remained identical. Ten verification
   traces formed two alternating, bitwise-identical trajectory groups. CPU replay
   reconstructed network actions within 3.1e-6. Model reload/input mismatch is
   not supported as the explanation for those traces.
4. Native source documents a Direct-GPU pose/zero-dynamics ordering risk crossed
   by the current reset path. This is a hypothesis for the full task, not a
   demonstrated cause. An isolated two-cube experiment compared current batch
   reset against native teleport plus two zero-duration publications: both were
   finite and bitwise repeatable in 10/10 trials, moved no unrelated body and
   produced identical states after 1, 8 and 31 real steps. It did not reproduce
   the task's two-cycle. **No unproven reset workaround was landed.**

Additional bounded training-reset panels checked the previously successful
step-500 and step-750 DLS/GRU checkpoints: 5/10 and 3/10 complete successes,
respectively, with 10/10 lifts in both panels. Neither reached 9/10, so no fresh
verification or broader checkpoint search was performed. These results also show
that selecting from a single stochastic rollout is insufficient.

A separate four-demonstration diagnostic used 3,000 updates and four sampled
training reset states. The selected `step_1000.pt` completed
3/20 verification rollouts, with
15/20 lifts, below the 18/20 gate. It used
4,608,000 supervised targets; its state set and update budget differ from the
one-demonstration runs, so it is not a compute-matched causal comparison.

The next causal check is a robot-plus-cube reset/contact replay independent of
the learner. A separate four-demonstration diagnostic holds the learner/controller
fixed while changing training coverage; it is an ablation, not a declaration that
the one-reset gate passed. It cannot isolate reset state from coverage by itself.
Do not run formal 32/160-episode learning/efficiency sweeps merely to bypass a
failed qualification gate.

## Engineering verification

- A real 2-generation smoke updated model parameters and consumed both generations.
- Exact offline replay of the same corpus and 2,000 MLP minibatches produced
  **bitwise-identical model tensors** to the source run (maximum difference 0).
- A resumed geometry GRU advanced step 250 to 260 with restored optimizer and a
  new ODS source. Its 960 supervised targets included G1=576 and G2=384.
- Viser and TensorBoard both returned HTTP 200 during the resume run; both ports
  closed afterward. The evaluator exited 0 and producer cleanup completed.
- One earlier DLS/MLP run exceeded the engine's 5-second graceful-close bound and
  was correctly reported as forced/durability-unconfirmed. A reusable configurable
  `shutdown_timeout` now preserves the default 5 seconds; this showcase uses 30.
  Forced termination is still an error. The later resume run closed cleanly.
- A deadline probe produced a censored, unattained endpoint without any optimizer
  update. No post-deadline result can select an equal-budget checkpoint.
- Zero-padded quaternion conversion and CUDA recurrent resume training mode were
  corrected after real probes exposed them; failed artifacts remain available.

The final focused validation selection passed 407 checks, including pipeline,
controller, task, model, offline/reporter and project-context coverage. Public API
coverage was 2604/2604 and context/documentation checks passed. The full repository
suite was not required for this focused change. A native entity-balance warning
also appeared on some otherwise successful simulation teardown paths.

The host uses Python 3.11 / PyTorch 2.7 and native DexSim 0.5.1 build
`401968ebfd112a3c11ab9deea338e2418542fe6e`. As in earlier experiments, a temporary
Python compatibility overlay was required for stale Newton facade exports; default
physics was retained. Experiment invocations used dedicated free GPUs. Concurrent
qualification/diagnostics were not treated as controlled timing benchmarks.

## Retained evidence

- `/tmp/ods-v2-pose-qualified40`, `/tmp/ods-v2-dls-qualified40`
- `/tmp/ods-v2-overfit-one`, `/tmp/ods-v2-gru-overfit-one-finite`
- `/tmp/ods-v2-dls-overfit-one`, `/tmp/ods-v2-dls-gru-overfit-one`
- `/tmp/ods-v2-dls-gru-checkpoint-panels`, `/tmp/ods-v2-dls-gru-four-diagnostic`
- `/tmp/ods-v2-mlp-traces`, `/tmp/ods-v2-dls-gru-reset-audit.json`
- `/tmp/ods-v2-reset-probe` (exact driver and negative/inconclusive result)
- `/tmp/ods-v2-offline-smoke`, `/tmp/ods-v2-offline-matched-check`
- `/tmp/ods-v2-resume-viewer-fixed`, `/tmp/ods-v2-engineering-audit.json`
- `/tmp/ods-v2-engineering-comparison`, `/tmp/ods-v2-final-comparison` (preliminary; speedup withheld)
- `/tmp/ods-v2-final-smoke`, `/tmp/ods-v2-final-matched` (matching initialization, normalization and final weights)

See README.md for executable commands. No remote issue, PR, merge or deployment
was performed in this implementation step.
