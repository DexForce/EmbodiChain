# Working PickPlace showcase: diagnosis and validation

The default recipe uses 32 initial expert episodes, a small absolute-target
chunk BC policy, and two online generations. The physical task, initial robot
pose, cube randomization, contacts, 600-step horizon and complete-success
predicate are unchanged. **The development-selected step-5,000 policy passes 99/100 reserved scenes.**
The step-10,000 policy passes 65/100 and is not the selected output.
[Machine-readable results](WORKING_SHOWCASE_RESULTS.json) retain all failures.
The [learning and retention plot](/tmp/ods-v4/working_curves.png) shows both outcomes.

## Why the earlier learner failed

The failures had several contributing design problems. The experiments below
narrow them without claiming that one factor explains every failure.

1. **Four demonstrations did not cover the reset distribution.** Their initial
   X coordinates covered about 3 cm and Y about 1.6 cm, concentrated in one
   corner of the permitted 8 cm by 8 cm region. A source-matched LeRobot load
   produced the same seven model checkpoints and losses as frozen ODS. Therefore
   changing persistence format could not supply the missing state coverage or
   repair the learned controller. Increasing to 32 demonstrations improved some
   GRU checkpoints, but a four-scene diagnostic still fluctuated between 0/4 and
   2/4 complete successes. Data quantity alone did not make that recipe reliable.
2. **Low one-step error did not ensure a stable rollout.** On teacher states,
   losses were small. During student execution, millimetre-scale contact changes
   led to different grasp offsets and velocities, after which the recurrent
   policy could push down, lift too early or drop the cube. For one training
   reset, successful and failed student traces started with identical measured
   state and action; their post-contact trajectories diverged substantially.
   A full-scene probe replayed saved expert requests successfully in 6/6 runs,
   and reconstructed absolute references in 6/6. This establishes that the
   control chain can realize those motions. It does not prove the simulator has
   no reset-state sensitivity, nor identify an internal backend defect.
3. **The expert used an unnecessarily difficult wrist orientation.** The fixed
   initial TCP was approximately 177.7 degrees from the old top-down grasp.
   The cube admits a parallel-jaw grasp rotated 180 degrees about its vertical
   axis, only 45.7 degrees from the same initial TCP. The old 32-episode corpus
   included wrist positions near -3.016 radians; the new orientation pilot kept
   that joint between roughly 0.05 and 0.94 radians. The new teacher qualified
   40/40, versus the previous 38/40, with mean final goal error 6.77 mm over all
   40 qualification scenes. This changes the demonstration, not the task or
   physical acceptance conditions. Both 32-episode orientation corpora have
   bitwise-identical initial cube XY positions.
4. **Short per-step control exposed the learner to contact and timing errors.**
   The teacher contains a 30-step closed-gripper dwell. A policy must produce a
   coherent close/hold/lift/transport/release/withdraw trajectory when observed
   states remain similar during that dwell. The replacement predicts 64 future
   absolute TCP/gripper targets and executes 16 before replanning from the two
   most recent measured observations. Low-level control still corrects each
   target against the live measured TCP. This retains feedback while reducing
   independently predicted incremental-action drift. No teacher phase, progress
   label, episode clock or success signal enters the model.

These are engineering comparisons, not a fully isolated model ablation. The new
learner has more parameters and supervises more future targets per update than
the old GRU. Changing horizon, representation, augmentation and data coverage
prevents attributing all improvement to any one of them.

## Why the second online round regressed

The reserved result is 99/100 for the development-selected step 5,000 and 65/100
for step 10,000. Both were selected for verification before test outcomes could
influence model choice. The selected model lifted 100/100 cubes; its only failure
(seed 50060) ended 3.37 cm from the goal and was correctly rejected by the unchanged
3 cm limit. Mean final error over all 100 cases was 11.03 mm.

A post-training audit evaluates both frozen models on every archived training
anchor, without adding test data or making an update:

| Training generation | Arm MSE at 5,000 | Arm MSE at 10,000 | Gripper MSE at 5,000 | Gripper MSE at 10,000 |
| --- | --- | --- | --- | --- |
| G1 |0.000526|0.000918|0.000847|0.001416|
| G2 |0.001100|0.000407|0.002763|0.000462|

Fit improves on the new generation and worsens on the old one. This is evidence
consistent with forgetting when the bounded cache is replaced and optimization
continues at a fixed learning rate. It does not isolate data replacement from
optimizer dynamics or explain every contact failure. The existing development
checkpoint selection protects the delivered policy; it does not solve continual
learning stability. The default entry evaluates and reports the selected
`summary.json.best_checkpoint`, and retains the later regressed checkpoint.
No replay aggregation or test-based retuning was introduced to hide this result.

## Implemented recipe

- MLP-512 with two hidden layers and a chunk head, 643,712 trainable parameters.
- Inputs: two consecutive 51-feature measured geometry observations. Missing
  initial history repeats the first observation.
- Prediction: 64 targets, each containing arena TCP position, six-dimensional
  rotation and normalized finger request. Output normalization is frozen from
  initial data, alongside observation normalization.
- Targets are reconstructed from the recorded **pre-action** TCP and the actual
  requested relative command. They are not measured future displacements.
- Execution: 16 queued targets, then replan. Absolute target error is converted
  to the same bounded 7D relative requests used by the teacher and existing DLS
  action manager. Episode reset clears history and queued predictions.
- Training: AdamW, learning rate 3e-4, weight decay 1e-6, gradient norm limit 1;
  0.01 normalized observation noise during training only; batch 256; two online
  rounds of 5,000 updates, with immutable checkpoints and development selection.
- ODS: CPU cache of 32 episodes; independent producer/evaluator; manual refill;
  actual generation provenance copied with the sampled tensors. The new context
  sampler admits episode-start/end anchors, masks missing future targets, and
  never crosses validity gaps or continuity boundaries.

This is a plain PyTorch BC implementation. The short observation history and
receding action-chunk organization are informed by the official
[ManiSkill state imitation baseline](https://github.com/haosulab/ManiSkill/blob/main/examples/baselines/diffusion_policy/train.py)
and its [controller documentation](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html).
It does not implement diffusion or claim ManiSkill's published performance.
The longer 64/16 horizons are this task's empirically checked configuration.

## Evidence and limitations

| Experiment | Complete success | Interpretation |
| --- | --- | --- |
| Previous four-demo GRU, repeated training resets | 3/20 | Historical failed fitting/robustness baseline |
| Previous four-demo GRU, independent development | 0/20 | Historical generalization failure |
| Raw expert request replay / absolute reference replay | 6/6 each | Physical control feasibility diagnostic |
| Old teacher, 32-demo absolute chunk prototype | 18/20 development | Viable learner before teacher improvement |
| Revised teacher qualification | 40/40 | Meets original physical expert requirement |
| Revised teacher, 32-demo chunk prototype | 20/20 development | Prototype result, separate from integrated path |
| Integrated offline entry, 32 demos / 10,000 updates | 20/20 development | Actual production example path |
| Integrated online entry, step 5,000 | 20/20 development | Selected by development success/lift/error |
| Integrated online entry, step 10,000 | 18/20 development | Consumed both generations; regression retained |
| Revised one-demo diagnostic | 6/10 repeated training resets | Failed 9/10 diagnostic, retained as a limitation |
| Revised four-demo diagnostic | 8/20 repeated training resets | Failed 18/20 diagnostic, retained as a limitation |
| Selected step 5,000, reserved test | **99/100** | Passes original 80/100 complete-task target |
| Step10,000 after two generations, reserved test | **65/100** | Fails 80/100; no validated T80 claim |

The reference recipe requires multiple demonstrations. It does not claim robust
one-shot learning. The one-demo failure is not relabeled as a pass because the
32-demo policy succeeds. Four-demo and held-out test results, including all failures, are retained in the
machine-readable report.

The online run generated 64 successful episodes, published two generations and
supervised 91,732,262 generation-1 targets plus 56,588,907 generation-2 targets.
This proves fresh data was consumed. It does not show that refresh improved
quality: the development-selected checkpoint is step 5,000, before the second
round. Both that checkpoint and the step-10,000 checkpoint confirming two
consecutive >=80% development panels were frozen before reserved-test outcomes
were used. Their 100-scene tests use seeds 50000–50099 and cannot select or tune a
replacement model. Full failures are retained.

Final test evaluation uses independent B=1 evaluators distributed over three
GPUs of the same model for turnaround. It is a correctness/learning check, not a
single-GPU timing benchmark. The official entry remains one producer and one
independent evaluator. No multi-seed ODS-versus-offline efficiency advantage is
claimed. The offline corpus is data-ready; its generation cost must not be
omitted from a cold-start comparison.

The integrated training runs used `--final-test-scenes 0` before the separate,
frozen-checkpoint test pass. Their original summaries consequently leave the
in-run learning-target flag false. The attached 100-scene reports establish the
selected model's final target result; the original summaries were not rewritten.

## Display, recovery and reproducibility

The resume/viewer smoke restored model, optimizer, input/output normalization
and RNG state, advanced step 10,000 to 10,010, consumed both new generations and
completed the engineering evaluation. Viser and TensorBoard returned HTTP 200.
The saved initial/final smoke videos remain under `/tmp/ods-v4/resume-viewer`.
The ordinary same-scene comparison uses the online initialization and step-5,000
policy at engineering seed 90000: initialization fails and the trained policy
succeeds. Its full failure is retained; the successful recording is held on its
last frame when composing the side-by-side video.

- [Before/after video](/tmp/ods-v4/comparison-video/before_after.mp4)
- Online run: `/tmp/ods-v4/integrated-online`.
- Offline run: `/tmp/ods-v4/integrated-offline`.
- Expert qualification: `/tmp/ods-v4/yaw-qualified40`.
- Frozen test-candidate hashes: `/tmp/ods-v4/frozen_test_candidates.json`.
- Selected / two-generation tests: `/tmp/ods-v4/final-test100`, `/tmp/ods-v4/threshold-test100`.

A LeRobot 0.4.4 export of the same working 32-episode corpus is retained at
`/tmp/ods-v4/yaw-lerobot32`; SDK read-back validates exact numeric equality. The
full integrated offline run reported above used the numeric corpus. No native
LeRobot minibatch-throughput result is inferred from that equivalence.

On the experiment host, use the existing Python 3.11 runtime and compatibility
overlay for the installed native package's stale Newton facade exports. The task
uses default physics. This trains from scratch with live ODS and opens both pages:

```bash
cd /root/sources/EmbodiChain-ods-pick-place-showcase
PYTHONPATH=.:extensions:extensions/embodichain_tasks:/tmp/ods-dexsim-runtime \
EMBODICHAIN_SIM_EXIT_PROCESS=0 \
/root/miniconda3/envs/py311/bin/python examples/data_pipeline/ods_pick_place_showcase.py \
  --sim-device cuda:1 --train-device cuda:1 \
  --qualification-report /tmp/ods-v4/yaw-qualified40/expert_qualification.json \
  --output /tmp/ods-pick-place-demo
```

Choose an idle GPU and a fresh output directory. Omit the qualification-report
argument to repeat all 40 qualification scenes. The default final evaluation
also runs the initialization on the reserved panel for comparison; this costs
additional simulation time. The separate accelerated verification driver used
for this report is `/tmp/ods-v4/final_evaluate.py`, with its threshold counterpart
alongside it. No external pretrained model or dataset is required.

## Validation accounting

Focused validation passed 283 tests, API coverage 2604/2604 and the project-context
check. Independent engine, AI and robotics reviews are recorded in
[the review report](WORKING_REVIEW.md). All owned training/evaluation processes
exited normally.

The initial integrated online/offline runs began before two reporting fixes:
the TensorBoard per-window target rate counted contexts rather than all masked
future targets, and their normalization fingerprint omitted target buffers.
Their model updates and primary `EfficiencyTracker` target counts are correct;
the old TensorBoard rate must not be used. The current implementation counts
actual valid supervision and fingerprints all input/output normalization plus
chunk settings. The later resume smoke exercises this corrected path. No timing
or matched-policy efficiency claim relies on the earlier artifacts.

Native DexSim's existing entity-balance teardown warning is retained in logs.
Worker exit status, force-termination status and service shutdown are checked
separately. The physical success predicate was not relaxed and evaluation never
calls the teacher or a task-stage controller.
