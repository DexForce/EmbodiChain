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

## Executed paired cadence diagnostic v1

Before execution, `cadence-v1/protocol.json` froze source commit
`dfca55e92f85a473e20bc770f9a54e7f8517e49a`, source/config hashes, the checkpoint,
two auxiliary states and budget. Its SHA256 is
`aa576c9c44fd8de9301a401e95462aa1e299e3dba8ca7c21fdd87f5a04a41b77`.
The paired scene digest is
`ee481ae513bb5467a3b795bd4d385933b6327f36a770c253e3a0f000edf58030`.
Both arms used the frozen seed-20261025 20k checkpoint, reset seeds
`20296025`/`20296026`, 600 steps/scene and physical GPU4. The only changed
policy setting was execution horizon 8 versus 1. No training update or formal
panel was involved. Every step records measured geometry, geometric phase,
target/request, goal/TCP errors and actual model-forward count.

| Horizon | Auxiliary success | Lift | Mean final goal error | Actual forward calls | Wall time |
| --- | --- | --- | --- | --- | --- |
| 8 | 0/2 | 1/2 | 0.1803 m | 150 | 40.92 s |
| 1 | 0/2 | 0/2 | 0.3393 m | 1,200 | 41.61 s |

Both reports confirm identical checkpoint/panel digests, no expert
initialization and completed close/flush. Reducing the horizon did not improve
this pair; two auxiliary states do not estimate a general success rate.

The H8 second scene reached the goal (final error 8.49 mm) but had no withdrawn
step among its 291 near-goal steps: TCP–cube distance never reached the required
0.10 m there. Both fingers reached the required 0.035 m opening on only 23 of
those steps. At the last step they were about 0.0146 m open and TCP–cube distance
was 0.0553 m. The task therefore never completed release/withdraw. The first H8
scene remained in approach for all 600 steps (nearest TCP–cube distance 0.361 m).
Geometric phase labels are diagnostic measurements, not expert states or new
success criteria. These observations locate distinct failure phases without
establishing an implementation defect or a unique learning cause.

## Next bounded protocol: sequence execution coherence v2

Test whether longer execution of the already predicted F32 sequence changes
release/withdraw behavior. Use **zero training updates**, the same immutable
20k checkpoint and a new auxiliary panel. This tests a specific receding-horizon
hypothesis after the negative H1 result. It is not a revision of formal v1:

| Item | Fixed value |
| --- | --- |
| Checkpoint | seed 20261025, step 20000, SHA256 `61d4df2456727d59fb7e910edb1a2c64116351b85daece911fd8445a33db85ef` |
| Inputs/model/normalization | Original H2/F32, two 256-pixel cameras, all saved buffers/weights unchanged |
| Panel | Two new auxiliary states, seed `20297025`, same panel in both arms |
| Only changed factor | Executed target horizon 8 versus 32 |
| Budget | 600 steps/scene, two scenes/arm, 240 seconds/arm, physical GPU4 only |
| Measurements | Step traces, success/lift/error, finger opening and TCP withdrawal, actual forward calls, hashes and cleanup |
| Selection | No checkpoint selection or formal-test access; retain both arms, including failures |

This protocol has not been executed. After admitting an idle GPU4, copy this
protocol to a fresh output directory and record its SHA256 before either arm.
Run the arms serially; save both logs. The checkpoint and task-specific output
variables below do not write frozen inputs.

```bash
BC_CHECKPOINT=/data1/dyc/.codex-artifacts/embodichain-sl-bc-20261025/formal-v1/seed-20261025/training/step_020000.pt
BC_DIAGNOSTIC_OUTPUT=/root/outputs/sl-convergence-20261026/bc/cadence-v2
mkdir -p "$BC_DIAGNOSTIC_OUTPUT"
cp examples/learning/bc/research/acceptance.md "$BC_DIAGNOSTIC_OUTPUT/protocol.md"
sha256sum "$BC_DIAGNOSTIC_OUTPUT/protocol.md" > "$BC_DIAGNOSTIC_OUTPUT/protocol.sha256"
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 240s \
  embodichain run-example learning/bc evaluate --diagnostic --trace --action-horizon 8 \
  --checkpoint "$BC_CHECKPOINT" --scenes 2 --seed 20297025 \
  --sim-device cuda:0 --policy-device cuda:0 --output "$BC_DIAGNOSTIC_OUTPUT/h8.json" \
  > "$BC_DIAGNOSTIC_OUTPUT/h8.log" 2>&1
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 240s \
  embodichain run-example learning/bc evaluate --diagnostic --trace --action-horizon 32 \
  --checkpoint "$BC_CHECKPOINT" --scenes 2 --seed 20297025 \
  --sim-device cuda:0 --policy-device cuda:0 --output "$BC_DIAGNOSTIC_OUTPUT/h32.json" \
  > "$BC_DIAGNOSTIC_OUTPUT/h32.log" 2>&1
```

Require equal checkpoint and scene digests before comparing arms. Overrides and
traces require `--diagnostic`; the formal driver supplies neither override. A
reproducible improvement would require a separately frozen recipe and unchanged
physical success thresholds before broader evaluation. Do not open the sealed
test panel or repeat the original 3×20k run. If coherence does not improve
behavior, the next learning design must declare one supervision/recovery
change, its development-only budget and new source/data signatures before any
training. Preserve the original formal seeds, budgets, thresholds and failures.
