# Ordinary offline BC diagnostics and learnability

The bounded first-eight learnability check completed 2,000 updates and failed
its preregistered grasp-phase gate. No phase-balanced sampling comparison,
full-corpus training or formal test followed. PR861 remains draft.

These experiments retain the PR828 model, target codec and original training
corpus. All phases and physical task truth are diagnostic or supervision data;
the policy receives only measured proprio27 and causal RGB histories. There is
no expert intervention, rule-based release or new training method.

## Main integration and preserved baseline

The BC branch normally merged `main@7cc143dd0`, including shared DDP, ODS,
RWM, TensorDict batch handling and robot/solver updates. Main owns shared SL,
simulation and robot configuration; this diagnostic work changes only the
showcase. The model, objective, reader and historical formal driver retain
their behavior. The original three formal seeds remain failed at 0/20 on the
development panels; independent 100-scene tests remain sealed.

After integration, actual PR828 forward/loss/gradient/AdamW parity and the old
2k checkpoint's 384 frames/48 replans, queues and decoded actions all had zero
maximum difference. This checks compatibility, not physical learning success.

## Fixed supervision audit

Execution source: `91054c9283b49f78112ca71135c467ccf098299d`.
The first 32 training episodes and 512 midpoint-stratified complete H2/F32
anchors were frozen before one inference pass per checkpoint. No update or
physical evaluation occurred. Full input hashes matched before and after.

| Checkpoint | First-target TCP mean | First-target p95 | First-target gripper sign errors |
| --- | ---: | ---: | ---: |
| Original 20k | 1.144 mm | 2.290 mm | 0/512 |
| Historical full-window continuation | 0.954 mm | 1.876 mm | 0/512 |
| Historical partial-window continuation | 1.027 mm | 2.031 mm | 0/512 |

Error-to-command-amplitude and direction measurements use the same anchor.
They do not support a blanket conclusion that command increments are obscured
by prediction noise. These are teacher-forced training samples, so they also
do not establish adequate accuracy in student-induced states.

All 256 withdrawn frames in the first-32 panel are excluded as complete-window
conditioning anchors; partial mode admits them. They already occur as future
labels. The previous full/partial comparison failed, so this coverage gap alone
is not a sufficient explanation or solution.

## Four-scene physical diagnostic

Execution source: `1b1d03976b7da90e232079ca0301e7b814bb6369`, with the same
source tree as integrated `393eb1fb4`. The original 20k checkpoint, H32
execution, auxiliary seed 20301025, four scenes and 600 steps per scene were
fixed before the single run. No checkpoint was selected using this panel.

The unchanged physical predicate produced 2/4 successes and 4/4 lifts. Across
2,215 steps and 71 actual forwards, independent same-time condition, stable
counter, success clock and queue checks found no mismatch. Observed and
physical TCP/cube positions matched exactly; decoded requests matched the
recorded clamped actions within 7.44e-7.

One failure stopped 37.19 mm from the goal without reaching its 30 mm region.
The other ended 4.83 mm from the goal but sustained every success condition for
only three steps (0.1 seconds), below the required nine steps (0.3 seconds).
Opening and withdrawal conditions frequently failed together. This locates
failure phases without establishing a unique learning cause. New scenes and
runtime prevent interpreting 2/4 as a gain over historical auxiliary panels.

Optional `evaluate --diagnostic --trace` now records distinct before/after
states, velocities, physical conditions, cached counters, measured inputs,
queue position and actual encoder features. Only the first two replans per
scene save RGB histories. Logging adds no policy forward or counter update.

## First-eight learnability check

Execution source: `ab64e40c7593885c6688e4df5849dc64414e33e3`.
Protocol SHA256: `cce6e8ddd59aa2b5da6f3becde34d8c4be175df2e1cf3747722fddbdb2c591cf`.

The derived manifest selects the first eight original episodes and references
their immutable files. Original first-32 normalization and 20k model weights
are retained. Fresh AdamW uses LR3e-5, weight decay1e-6 and gradient norm1;
seed 20302025, batch64, partial targets and 2,000 updates are fixed. The training
limit is 900 seconds. Resource preflight refused busy GPU4 before any worker;
a hardware-only addendum assigned idle GPU3 without changing the protocol.

Before and after evaluations use the same 488 anchors: 256 midpoint samples
plus each episode's final31 anchors, deduplicated. There are 11,648 valid future
positions; padding never enters target encoding or metrics. These are
training-distribution diagnostics, with repeated future positions, rather than
independent generalization samples.

The predeclared gate requires every phase to have samples, TCP p95 at most
5 mm, rotation p95 at most 5 degrees, gripper MAE at most 0.1 and strong-label
gripper sign agreement at least 95%.

| Target phase | TCP p95 before | TCP p95 after | Gate |
| --- | ---: | ---: | --- |
| Approach | 6.274 mm | 2.154 mm | Pass |
| Grasp | 12.353 mm | 7.506 mm | Fail |
| Transport | 2.868 mm | 2.298 mm | Pass |
| Release | 54.452 mm | 1.685 mm | Pass |
| Withdraw | 63.239 mm | 2.369 mm | Pass |

Training completed in 456.90 seconds. Fixed-panel normalized MSE fell from
0.00422974 to 0.0000687008, a 98.38% decrease. All phase rotation p95 values
were below 0.7 degrees, gripper MAE below 0.01 and strong-label sign agreement
100%. The grasp TCP threshold still failed; aggregate loss cannot override it.

Grasp error is concentrated at distant chunk positions: TCP p95 is 1.256 mm
at offset 0, 1.400 mm at offsets 1–7, 2.313 mm at 8–15 and 12.689 mm at 16–31.
This narrows a future investigation to long-horizon grasp prediction without
proving an architecture defect. No extra model inference or training was used
for this saved-output analysis.

All 2,000 updates, 47 AdamW state step counters, sampler replay, six unchanged
normalization buffers and bound input/source hashes were independently checked.
The owned process group exited and GPU3 returned idle. Only the final checkpoint
was assessed. The failed gate stops the planned sampling comparison and formal
training; the budget, seeds and thresholds were not expanded.

## Reproduction and artifacts

The private research commands require a clean Git checkout to bind their
execution provenance. The generic installed BC factory remains simulator-free
and works independently of these research commands.

```bash
embodichain run-example learning/bc offline-audit \
  --manifest /path/to/corpus/manifest.json \
  --checkpoint original20k=/path/to/original20k.pt \
  --checkpoint full2k=/path/to/full2k.pt \
  --checkpoint partial2k=/path/to/partial2k.pt \
  --output-dir /path/to/fresh/audit --prepare-only

embodichain run-example learning/bc learnability \
  --config examples/learning/bc/configs/learnability.candidate.yaml \
  --output-dir /path/to/fresh/learnability --prepare-only
```

The fixed learnability YAML names the historical local assets; adapt paths
and identities in a new protocol when reproducing elsewhere. Execution uses
an explicitly admitted device and a fresh output directory. Existing inference
or training outputs cannot be overwritten. These commands document the
completed bounded experiments, not a request to repeat the stopped protocol.

Detailed local artifacts are under `/root/outputs/bc-improvement-20261027/`:
`engineering/`, `offline-audit/`, `closed-loop/runtime-v1/`, `learnability/`,
`review/` and `wheel/`. Protocols, content hashes, raw predictions, phase and
offset metrics, physical traces, curves, worker logs and cleanup are retained
outside Git. Historical evidence remains under its original identities.
