# Historical formal result and tail-conditioning protocol

This page preserves the original formal failure and completed tail-conditioning
comparison. The subsequent [supervision, closed-loop and first-eight learnability
diagnostics](improvement-v2.md) have their own source identities and stopping
decision; they do not replace these historical results.

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

## Executed sequence coherence diagnostic v2

The pre-execution manifest SHA256 is
`4b2c8587266ea24a5d492e86c0adc612a7f70b74ece3fcb14733cd156a9097e9`;
its source commit was `cf60f4dd17c524543de150beee2f32c1516630bd`.
Both arms used the same frozen 20k checkpoint, auxiliary reset seeds
`20297025`/`20297026`, 600 steps per scene and physical GPU4. Checkpoint,
initial-panel and source hashes matched; no expert or rule-based release/
withdraw intervention was introduced. Both arms closed/flushed successfully.

| Horizon | Auxiliary success | Lift | Mean final goal error | Forward calls |
| --- | --- | --- | --- | --- |
| 8 | 0/2 | 0/2 | 0.3773 m | 150 |
| 32 | 0/2 | 2/2 | 0.2031 m | 38 |

H32 improved lifting on this pair without achieving task success. Its first
scene remained in transport after lifting; its second reached 10.4 mm final
goal error but recorded only one withdrawn step among 70 near-goal steps.
This is a partial phase improvement on two auxiliary states, not evidence of
formal acceptance or generalization. No more cadence panels or training runs
were added. Full traces and the frozen manifest remain under
`/root/outputs/sl-convergence-20261026/bc/cadence-v2/`.

## Tail-conditioning evidence and opt-in reader

A read-only numeric audit of the first 32 training episodes found 256 frames
satisfying near-goal, opened, withdrawn and instantaneous stability conditions.
All occurred in the final 31 frames. The original H2/F32 full-future sampling
rule admitted **zero** of them as conditioning anchors. They still appeared
1,152 times as future target positions; successful-phase labels were not absent.
This is a specific coverage hypothesis for late-phase behavior and does not
explain every approach failure.

`BCWindowDataset(include_partial_targets=True)` now admits all 256 as complete-
history anchors. A bounded actual sampler check delivered two terminal-phase
anchors in 152 samples, with two and seven valid targets respectively, complete
H2 histories and finite zero padding. No parameter update was performed.
The checksummed corpus manifest remains unchanged. Numeric coverage does not
claim a new integrity scan of every RGB file; the diagnostic's sampling check
used declared immutable files without rehashing their RGB payloads.

The option defaults to false. Default real-corpus B4 samples, sampling RNG and
source identity remain bitwise identical. The frozen formal driver does not
pass the option. Partial mode groups target encoders by real valid length,
never supplies synthetic fields to a transform, right-pads finite zeros and
returns `target_valid`. It binds a different source/recipe identity; cross-mode
checkpoint recovery is rejected. Model, loss formula, normalization and formal
acceptance thresholds are unchanged.

## Preregistered candidate: one window-mode variable

The runnable configuration is
[`tail-conditioning.candidate.yaml`](../configs/tail-conditioning.candidate.yaml).
The private `window-candidate` adapter calls the installable generic SL factory;
it supplies only task-owned target construction and a declared weight-only warm
start. It does not introduce a trainer or alter the formal acceptance driver.

| Item | Fixed in both arms |
| --- | --- |
| Corpus | Original 1,024 episodes, manifest SHA256 `6b757b0f2d1343ead3d9812abf8652757825981a64fe9a84042dcd8e9c558093` |
| Initialization | Frozen seed-20261025 20k weights, checkpoint SHA256 `61d4df2456727d59fb7e910edb1a2c64116351b85daece911fd8445a33db85ef` |
| Normalization | Original first-32 tensors, compared exactly with checkpoint buffers |
| Model/objective | Original H2/F32 RGB model and masked normalized target MSE |
| Only experimental variable | `include_partial_targets=False` versus `True` |
| New training budget | 2,000 updates × batch64 per arm, seed `20298025`, serial GPU4 |
| Optimizer | Fresh AdamW in both arms, constant LR3e-5, weight decay1e-6, gradient norm1; new update count starts at zero |
| Execution | Horizon32 fixed in both arms, unchanged physical controller/success predicate |
| Held auxiliary panel | Four new states, seed `20300025`, final checkpoint only, 600 steps/scene, 600 seconds/arm |
| Exploratory reporting gate | Candidate ≥3/4 full physical successes and ≥2 paired successes above control; report all failures and phase metrics regardless |

The exploratory gate is not the formal 18/20 then 80/100 acceptance protocol.
No sealed test or original formal development panel is used. No checkpoint
selection is allowed on this new auxiliary panel. The historical prepare-only
commands built both recipes on CPU at **zero updates**. Their resolved
configurations differed only in output directory and the window flag. Those
zero-update files remain unchanged historical preparation records. The training
and held-panel evaluation have since completed and failed the auxiliary gate;
a next recipe remains pending. The original execution instructions below are
retained as the preregistered protocol, not a request to repeat the stopped pair.

From the repository root, prepare fresh outputs with:

```bash
CUDA_VISIBLE_DEVICES='' embodichain run-example learning/bc window-candidate \
  --config examples/learning/bc/configs/tail-conditioning.candidate.yaml \
  --no-include-partial-targets --prepare-only \
  --output-dir outputs/bc-tail-candidate/control/training --report outputs/bc-tail-candidate/control/prepared.json
CUDA_VISIBLE_DEVICES='' embodichain run-example learning/bc window-candidate \
  --config examples/learning/bc/configs/tail-conditioning.candidate.yaml \
  --include-partial-targets --prepare-only \
  --output-dir outputs/bc-tail-candidate/partial/training --report outputs/bc-tail-candidate/partial/prepared.json
```

For a separately authorized training execution, freeze this configuration,
source commit, all input hashes and the two resolved prepared reports first.
Admit GPU4, replace `--prepare-only` with `--train`, use distinct `fit.json`
reports and wrap each serial arm in `timeout 1800s` with
`CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. Each arm remains
bounded to exactly 2,000 new updates. Evaluate only its `training/last.pt` with
`run-example learning/bc evaluate --diagnostic --trace --scenes 4 --seed 20300025`
and the declared 600-second arm limit. Do not resume or overwrite a formal run.

If the pair misses the exploratory gate, retain both results and stop; do not
expand seeds/budgets, pick another checkpoint or open the sealed test. Any
later learning change requires a new single-variable protocol. Even a passing
auxiliary pair would require a separately frozen review before broader formal
evaluation. The failed original seeds, budgets, thresholds and test seal stay
unchanged.


## Completed tail-conditioning comparison: auxiliary gate failed

The actual execution source remains
`f881407ff2e8b0f233f816d52fabaea36e3b9226`. This subsequent documentation update
records the result and does not change the source identity of the executed
models or claim a rerun at the documentation revision.

- Fixed configuration SHA256:
  `58e943d5ecc37bc62549f4b0f7b0530cdd19e749dbb14b50b6e604a0e3333e73`.
- Pre-execution manifest SHA256:
  `789b73b8fafda35b6d4c542cc9382d9fdbd865aeca722f3693224c4087fb7e3b`.
- Frozen preregistration Markdown SHA256: `e41cc76e7116d8a1b4eb7cc1b0dd3ce7e538307c50e652487cd6c0179136c93f`.
  Its original bytes remain in the run's `protocol.md`; `protocol.yaml`, all
  frozen inputs and the 43 indexed artifacts remain unchanged.

Both arms used the same H32 execution, fixed 20k initialization checkpoint,
original statistics, fresh AdamW and training seed `20298025`. Each completed
exactly 2,000 new optimizer updates × batch64 (128,000 chunks). Both training
arms finished before either final checkpoint was evaluated. Only each arm's
`last.pt` was evaluated on the same four held auxiliary states (seed `20300025`);
all eight episodes reached the 600-step limit. There was no intermediate
checkpoint selection, seed change, retry or rule-based release/withdraw policy.

| Window mode | New updates | Success | Lift | Mean final goal error | Last100 mean target MSE | Train seconds | Evaluation seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full control | 2,000 | 0/4 | 2/4 | 0.180703 m | 0.00019117 | 550.26 | 68.79 |
| Partial | 2,000 | 0/4 | 3/4 | 0.099315 m | 0.00017464 | 555.22 | 72.29 |

The partial arm had **zero paired success gain** and failed the predeclared
≥3/4 successes plus gain ≥2 auxiliary gate. Lifting and mean goal error improved
on this small pair, but neither arm achieved full physical task success.
Training losses use different window/mask distributions and do not substitute
for the task predicate. Valid target positions were 4,096,000 for control and
3,975,986 for partial; all six input/target/ImageNet normalization buffers
remained bitwise equal to initialization.

Increasing terminal conditioning coverage from 0 to 256 anchors in the
first-32 audit was a verified data precondition, not a sufficient explanation
or solution for task success. The held traces still show intermittent
release/withdraw geometry; the recorded necessary geometric conditions do not
reconstruct every velocity/stability condition. No implementation bug was
confirmed. A next recipe remains pending: first review action-increment
signal-to-noise, phase transitions and the closed-loop state distribution,
without proposing unvalidated model changes or expanding this stopped run.

Execution completed in about 21 minutes. All four owned workers exited,
environment close/flush completed, GPU4 returned to no compute processes, and
source/input hashes matched the manifest. The historical prepare-only records
still show zero updates with their original hashes. The original three-seed
formal failures, budgets, thresholds and sealed test remain unchanged. These
new same-H32 auxiliary arms must not be treated as a single-variable comparison
against the original H8 formal results. Keep the PR draft and stop under the
predeclared negative-result rule.

The complete immutable run is under
`/root/outputs/sl-convergence-20261026/bc/tail-conditioning-run-v1/`:
`REPORT.md`, `summary.json`, `paired-results.json`, `manifest.json`,
`artifact-sha256.json`, both `training/last.pt` files, complete metrics/logs,
held step traces, `learning-curves.png` and `phase-traces.png`.
