# Revised ODS PickPlace experiments

Run date: 2026-10-17 (UTC). The revised default online learner still fails the
80% task-success target. Fixing causal-history handling, observation processing
and fresh-data exposure did not produce a successful complete-task showcase.
The engineering loop passes its checks; learning performance is a separate
result. Both online parameterizations completed 0/100 held-out tasks. The
residual frozen control completed 2/100, below the target. The earlier experiment
is preserved in [VALIDATION.md](VALIDATION.md).

## Changes and scope

- `sample_batch_with_history()` copies a supervised window and its entire
  uninterrupted causal prefix under the same ODS producer lock. Right padding
  is zeroed and excluded by `valid`; `loss_mask` selects 24 supervised steps.
  Prefixes stop at invalid frames or continuity changes.
- The learner reconstructs GRU history from the same origin as online inference,
  propagates gradients through that history, and biases sampling toward eligible
  episode-start windows with probability 0.25. This replaces the original
  32-step window with an unconditional eight-step loss exclusion, which never
  supervised episode steps 0–7. Supervised target count remains 24 steps per
  sequence; full-prefix processing increases computation.
- Equivalent quaternion signs are canonicalized and normalized to unit length.
  Quaternion inputs use unit scale rather than near-zero empirical variance.
  Other frozen training-only statistics have feature-specific physical lower
  bounds; normalized inputs are clipped to [-10, 10].
- Each subsequent online round overlaps its first 250 updates with refill,
  then waits for complete generation publication before its remaining 750
  updates. Supervised-generation counts are separate from history-input counts.
- Checkpoint selection maximizes development success, then measured lift, then
  minimizes final goal error. Startup arm-target deviation is also reported.
- An optional `residual` arm parameterization learns a controller lead bounded
  by 0.25 rad from measured joint positions. It uses the same GRU-128 and two
  head layers and still emits eight absolute joint targets. The gripper remains
  an absolute target. Checkpoints preserve observation and arm contracts;
  older checkpoints retain their original inference behavior.

The physical task, expert, 600-step limit and success predicate were unchanged.
Every full run qualified the physical expert on 40/40 independent scenes.
Students used measured state and recurrent memory without expert takeover,
planning calls, phase labels or feedback-directed collection. No replay
aggregation, pretrained policy or offline training dataset was introduced.

## Protocol

All full runs used seed 42, 32 ODS episode slots, five rounds of 1,000 updates,
64 sequences per batch, AdamW at 3e-4, gradient clipping at 1 and gripper loss
weight 5. Online production used generations 1–5; frozen controls retained G1.
Each online/frozen pair began with bitwise-identical network and normalization
tensors. The pairs used distinct engine run IDs.

Twenty fixed development scenes selected each run's checkpoint. The original
100 held-out scenes, seeds 30000–30099, evaluated the initial policy and selected
checkpoint after training. These fixed test scenes were reused for comparison
with the earlier experiment, without entering training or normalization.
Current test outcomes were not inspected while choosing the default mode.

At 06:40:18 UTC, both online variants had completed all development evaluations
without complete successes. The rule recorded before test inspection kept the
existing absolute default unless residual prediction demonstrated complete-task
improvement on the online development set. The residual mode remains optional. The decision
and development-report hashes are retained in
`/tmp/ods-revised-default-selection.json`. The later frozen-control development
success did not change this decision after test inspection.

## Controlled 1,000-update ablation

These four variants sampled the same initial generation from one live ODS
engine, with identical initial neural weights and equal update/target budgets.
They were evaluated only on the 20 development scenes.

| History/startup treatment | Physical observation treatment | Complete success | Measured lift | Mean first arm-target deviation |
| --- | --- | --- | --- | --- |
| Original window/burn-in | Original statistics | 0/20 | 2/20 | 0.6874 rad |
| Full history + startup bias | Original statistics | 0/20 | 1/20 | 0.1857 rad |
| Original window/burn-in | Canonical quaternion + physical scales | 0/20 | 1/20 | 0.7414 rad |
| Full history + startup bias | Canonical quaternion + physical scales | 0/20 | 0/20 | 0.1864 rad |

The history treatment bundles full-prefix reconstruction, gradients through
history and start-window bias. The observation treatment bundles quaternion
canonicalization, scale floors and clipping. This ablation does not isolate
each component's independent effect. Startup deviation is the largest absolute
arm-joint target difference from the reset positions, averaged across scenes.
The expert's first command holds those reset positions.

## Full-budget results

Every row has 5,000 optimizer updates. Test results belong to the selected
checkpoint, which may precede the last update.

| Arm prediction | Data | Selected step | Development complete success | Test complete success | Test measured lift |
| --- | --- | --- | --- | --- | --- |
| Absolute | Online | 3000 | 0/20 | 0/100 | 0/100 |
| Absolute | Frozen | 4000 | 0/20 | 0/100 | 1/100 |
| Residual | Online | 4000 | 0/20 | 0/100 | 25/100 |
| Residual | Frozen | 5000 | 1/20 | 2/100 | 20/100 |

All four initial policies had zero complete successes and zero measured lifts
on their 100 test scenes.
The residual-online checkpoint reached a 30% development lift rate, whereas
the residual-frozen run reached 60% lift at step 4000 and its first complete
development success at step 5000. Lift is not complete task success. These are
single-training-seed results, and they do not establish a general online-data
advantage or an 80% learning capability. No run met the target.

## Actual data exposure

The default online run generated 160 accepted expert episodes and 53,107 valid
frames, with zero rejected attempts. Its 320,000 sampled sequences supplied
7,680,000 supervised targets. Both online parameterizations used the following
supervised exposure:

| Generation | Supervised targets | Equivalent 64 × 24 minibatches |
| --- | --- | --- |
| G1 | 1,920,000 | 1250 |
| G2 | 1,536,000 | 1000 |
| G3 | 1,536,000 | 1000 |
| G4 | 1,536,000 | 1000 |
| G5 | 1,152,000 | 750 |

The earlier run trained on G5 in only one minibatch. Publication now precedes
the final 75% of each round's updates. Thus the latest generation was genuinely
trained on, although this scheduling repair did not improve default task
success. Frozen controls supervise the same total target count from G1 alone.
These exposure counts describe the complete training run. Earlier selected
checkpoints do not include later generations; their test results must not be
interpreted as evaluations of the step-5000 learner or of G5 alone.

The default run also consumed 44,761,520 valid recurrent input frames, including
repeated prefixes. That figure is not a count of unique generated data and is
not directly comparable to the previous fixed-window input count.

## Interpretation and retained failures

The original startup-supervision and quaternion-equivalence defects were real.
Their repair reduces startup errors and makes equivalent quaternion inputs
consistent, but they were insufficient explanations for complete-task failure.
All four runs had finite training losses and real parameter updates.
Small supervised error still failed to provide stable autonomous task execution.

Development probes found both missed grasps after approach/contact deviation
and sustained physical grasps followed by failed transport/release. Two lifted
residual-policy cases retained the cube between closed fingers while moving it,
then deviated from the target and remained closed through step 600. This is
evidence about those inspected failures, not a success-rate claim. Robustness
outside expert trajectories and transport/release accuracy remain unresolved;
their separate causal contributions were not isolated here.

The before/after movies use the same first three test seeds and retain all
failures. Labels identify the initial model, selected checkpoint step and seed.
The six recordings per arm parameterization are also retained independently.
The recordings are diagnostic artifacts, not a successful-only demonstration.

## Engineering validation

- 329 distinct focused sampling/BC, related Gym/context and documentation-checker
  checks passed across the proportional validation selections. The final BC selection has 12 tests,
  including immediate producer-error propagation on both sampling paths.
- A real two-generation smoke performed ten updates. Its second round consumed
  one old-generation minibatch followed by four new-generation minibatches.
- Resume continued step 5000 to 5010 with restored optimizer/model state and a
  fresh ODS identity, consuming generations 1 and 2. It did not restore replay.
- Viser and TensorBoard returned HTTP 200 during resume smoke and both ports
  closed afterwards. All full-run and movie workers exited; the final owned
  process-tree audit found no remaining processes, and experiment GPUs were free.
- API coverage was 2602/2602. Project-context and whitespace checks passed.
  Black 26.3.1 was run over the checkout before commits.
- The Sphinx dummy build succeeded with 770 warnings. Comparison with the prior
  build found no new warning messages; duplicate ODS API indexing was removed.

The full runs used default physics, one production environment and one separate
evaluation environment per run. Python was 3.11.14, PyTorch 2.7.0+cu128 and the
native DexSim build was 0.5.1+401968ebfd112a3c11ab9deea338e2418542fe6e.
The same temporary Python compatibility overlay described in VALIDATION.md
was used against that native build. No fallback physics was introduced.
GPU 5 was shared by the two independent frozen runs, so wall-time throughput
comparisons between configurations would not be controlled measurements.

## Reproduction and artifacts

With compatible installed runtime dependencies, run from the worktree root:

```bash
python examples/data_pipeline/ods_pick_place_showcase.py \
  --no-viewer --no-video --output /tmp/pick-place-revised-online
python examples/data_pipeline/ods_pick_place_showcase.py \
  --mode frozen --no-viewer --no-video --output /tmp/pick-place-revised-frozen
python examples/data_pipeline/ods_pick_place_showcase.py \
  --arm-command-mode residual --no-viewer --no-video \
  --output /tmp/pick-place-residual-online
python examples/data_pipeline/ods_pick_place_showcase.py \
  --arm-command-mode residual --mode frozen --no-viewer --no-video \
  --output /tmp/pick-place-residual-frozen
```

Use separate devices or unused viewer ports when running concurrent experiments.
Omit `--no-viewer` and `--no-video` to launch Viser, TensorBoard and recordings.
Each run saves resolved task configuration, runtime/commit metadata, immutable
checkpoints, per-scene reports and TensorBoard events. Implementation references
are `bc5e93b6` for the causal/data changes and `3a72a8fb` for optional residual
prediction; later validation/documentation changes do not change policy or
sampler behavior in those experiments.

Retained implementation-host outputs:

- `/tmp/ods-revised-online`, `/tmp/ods-revised-frozen`
- `/tmp/ods-revised-residual-online`, `/tmp/ods-revised-residual-frozen`
- `/tmp/ods-revised-ablation` (driver, common-source and initialization checks)
- `/tmp/ods-revised-resume-viewer`, `/tmp/ods-revised-engineering-checks.json`
- `/tmp/ods-revised-diagnostics`, `/tmp/ods-revised-lift-diagnostics`
- `/tmp/ods-revised-comparison-videos/absolute_before_after.mp4`
- `/tmp/ods-revised-comparison-videos/residual_before_after.mp4`
- `/tmp/ods-revised-learning.png`

These local artifacts include failures. They are not bundled training datasets.
