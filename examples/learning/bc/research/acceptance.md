# Frozen result and next controlled recipe

The historical `sl-bc-offline-pickplace-v1` learning target failed. All three
training seeds (`20261025`, `20261026`, `20261027`) completed 20,000 updates;
every development checkpoint scored 0/20 successes. The last-two 18/20 gate
failed and all independent 100-state tests remain sealed. Low supervised loss,
reference parity and software completion do not establish task learning.

## Preserve the historical evidence

The source checkout `/root/sources/EmbodiChain-sl-bc` remains pinned to
`f0c6e720feca700ea1580dfdf040b94ef09b57de`. Historical artifacts live under
`/data1/dyc/.codex-artifacts/embodichain-sl-bc-20261025/`:

- `corpus-v3/manifest.json`: SHA256
  `6b757b0f2d1343ead3d9812abf8652757825981a64fe9a84042dcd8e9c558093`;
  1,024 accepted episodes, 530,841 frames, frozen first-32 statistics.
- `formal-v1/run-manifest.json`, `aggregate.json`, `REPORT.md`: original source,
  robot/SDK/data identities, commands, hardware, failure results and cleanup.
- Each `formal-v1/seed-*/` retains checkpoints, metric histories, every
  development scene/log, selection and summary. No final-test file was created.
- `rgb-binding-audit/`: prior sampled spatial/vector assignment and moving-frame
  timing audit. It found no confirmed RGB/geometry binding defect; its scope
  does not prove every corpus frame correct.

The migrated source is a new source identity. Do not relabel the historical
results as runs of `embodichain_models.policy.bc`, overwrite these artifacts,
or edit the frozen source to make an old manifest match. New engineering
outputs belong to `/root/outputs/sl-convergence-20261026/bc/`.

## Migration and bounded diagnostic evidence

The new namespace was compared with actual PR828 source at
`51f1f0cd5a7602f23358c142997e9045323f547a`. Historical encoder and target imports
are loaded from those source files; the reference network is compiled from its
source AST without restoring retired core exports. CPU forward, loss, every
parameter gradient and one AdamW step matched exactly. The actual frozen 2k
checkpoint also matched 48 replans, 384 queue targets and decoded actions
exactly on three recorded episodes. These are numerical migration checks.

A separate three-update B64 engineering continuation from that 2k checkpoint
restored model/optimizer, scheduler and sampling progress. CPU recovery tests
are exact. GPU recovery was not bitwise equal: model maximum difference was
`1.49e-8`, optimizer `2.00e-11`; all values satisfy the existing PR828 FP32
`atol=1e-5, rtol=1e-4` tolerance. The strict failed bitwise report and its logs
remain alongside the separate tolerance assessment. No formal run was resumed.

The read-only diagnostic used only training episodes 0, 31 and 127, every 16th
complete H2/F32 anchor (92 anchors per checkpoint). Mean first-target TCP errors
were 4.20 mm at 4k and 1.20 mm at 20k; 20k first-target p95 was 2.17 mm. The
label encode/decode maximum action error was `1.19e-6`. These teacher-forced
samples do not cover student-induced states or establish generalization.

One auxiliary closed-loop scene used the frozen 20k seed-20261025 checkpoint,
seed `20296025`, 600 control steps, unchanged action horizon 8, and physical GPU4
for both simulation and policy. It completed with no expert initialization and
confirmed environment close/flush: 0/1 success, no lift, 0.352 m goal error.
This is a main-runtime compatibility/negative diagnostic, outside every formal
panel. It does not attribute failure to one cause.

No confirmed implementation bug was found. Queue and decode equivalence,
owned causal inputs, source identity, normalization and recovery checks rule
out the specific regressions they exercise. Remaining hypotheses include
closed-loop state drift, execution cadence and insufficient supervision around
recovery/release. The prior privileged-state control also failed and changed
several factors, so vision alone is not an established cause. Keep the PR draft.

## Next recipe: frozen-checkpoint cadence diagnostic v1

First isolate execution cadence with **zero training updates**, one fixed
checkpoint and one new auxiliary panel. This is a separately declared research
recipe, not a revision of the failed acceptance protocol:

| Item | Fixed value |
| --- | --- |
| Checkpoint | seed 20261025, step 20000, SHA256 `61d4df2456727d59fb7e910edb1a2c64116351b85daece911fd8445a33db85ef` |
| Inputs/model/normalization | Original H2/F32, two 256-pixel cameras, all saved buffers/weights unchanged |
| Panel | Four auxiliary states, seed `20297025`, same panel in both arms |
| Only changed factor | Executed target horizon 8 versus 1 |
| Budget | 600 steps/scene, four scenes/arm, 600 seconds/arm, physical GPU4 only |
| Measurements | Per-scene success, lift, goal error, termination, checkpoint/panel hashes, cleanup |
| Selection | No checkpoint selection or formal-test access; retain both arms, including failures |

After checking GPU4 is idle, set a fresh output directory and run the two arms
serially from the installed example runner. The checkpoint and output variables
below are task-specific; no frozen input is written.

```bash
BC_CHECKPOINT=/data1/dyc/.codex-artifacts/embodichain-sl-bc-20261025/formal-v1/seed-20261025/training/step_020000.pt
BC_DIAGNOSTIC_OUTPUT=/root/outputs/sl-convergence-20261026/bc/cadence-v1
mkdir -p "$BC_DIAGNOSTIC_OUTPUT"
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 600s \
  embodichain run-example learning/bc evaluate --diagnostic --action-horizon 8 \
  --checkpoint "$BC_CHECKPOINT" --scenes 4 --seed 20297025 \
  --sim-device cuda:0 --policy-device cuda:0 --output "$BC_DIAGNOSTIC_OUTPUT/h8.json"
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 600s \
  embodichain run-example learning/bc evaluate --diagnostic --action-horizon 1 \
  --checkpoint "$BC_CHECKPOINT" --scenes 4 --seed 20297025 \
  --sim-device cuda:0 --policy-device cuda:0 --output "$BC_DIAGNOSTIC_OUTPUT/h1.json"
```

This plan has not been executed. Diagnostic overrides require `--diagnostic`
and are recorded in the output; the formal driver supplies neither override.
Require equal checkpoint and scene digests before comparing arms. If cadence
changes behavior, inspect the failing scene transitions before declaring a new
learning recipe. If neither arm improves, do not repeat the 3×20k run. The next
learning design must first specify a single supervision/recovery change, a
separate development-only budget and new source/data signatures. Preserve the
original formal seeds, budgets, thresholds and sealed test population.
