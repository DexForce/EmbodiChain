# ODS PickPlace: qualify lightweight BC before streaming

Status: design baseline, 2026-10-17. Core interfaces are now implemented;
[implementation evidence](IMPLEMENTATION.md) records the failed one-reset
repeatability gate and current limitations. See [review findings](REVIEW.md) and
[previous experiments](EXPERIMENTS.md) for evidence and limitations.

The [offline comparison and efficiency protocol](BENCHMARK.md) defines the
acquisition-inclusive/data-ready baselines, metrics, resource budgets and
multi-run data-reuse accounting. It is the owner of efficiency comparisons.

## Decision

Keep Franka single PickPlace, its reset distribution, real contact, 30 Hz control,
240 Hz default physics, 600-step limit and strict physical success predicate.
Keep training from scratch and independent student execution. Replace the
assumption that GRU joint-target BC is already viable with explicit learning
qualification gates. Prefer a small state MLP with measured geometry, then
qualify Cartesian pose commands through the same teacher/student controller.
Only a competent frozen learner advances to the ODS streaming showcase.

The initial default candidate is **51 current-state features → 256 ReLU →
256 ReLU → 7 normalized action outputs**, approximately 81,000 parameters.
It is a candidate to validate, not a claim of reaching 80% success. Preserve the
current joint GRU as a diagnostic baseline. Do not begin with ACT, diffusion,
pretraining, or a larger general imitation-learning framework.

## Official references and what transfers

Sources inspected on 2026-10-17; repository references are pinned below.

| Reference | Verified implementation | Use in this showcase |
| --- | --- | --- |
| [ManiSkill BC](https://github.com/mani-skill/ManiSkill/blob/62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3/examples/baselines/bc/bc.py) | Two 256-unit ReLU layers, linear action output, MSE, Adam at 3e-4; explicit demo/evaluation controller match | First lightweight model and objective reference; retain ODS as the data source |
| [Isaac Lab Franka low-dimensional BC](https://github.com/isaac-sim/IsaacLab/blob/b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/config/franka/agents/robomimic/bc_rnn_low_dim.json) | Robomimic, two-layer LSTM-400, five-mode GMM, sequence/horizon 10; eef pose, gripper and object observations | Reference for geometry and temporal/multimodal fallback; substantially larger than our GRU-128, not the first lightweight candidate |

The ManiSkill reference uses offline files; this proposal adopts its small policy
structure, not its storage or experiment infrastructure. The existing GRU/head
has about 78,000 parameters: the MLP mainly simplifies inference and sampling,
rather than reducing parameter count. The Isaac config enables GMM; its L2
configuration fields must not be read as proof that this policy uses plain MSE.

[ManiSkill controller documentation](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html)
distinguishes position and pose deltas. Its position-only controller preserves
current orientation; that is not a fixed top-down controller. Our physical
release/withdraw/settle success predicate remains authoritative, and results on
other frameworks' tasks are not performance predictions for this task.

## Policy and environment contract

### Current measured observations, version 2

| Feature | Width | Convention |
| --- | --- | --- |
| Robot qpos and qvel | 18 | Existing nine-joint ordering, including both fingers |
| TCP position and rotation | 9 | Arena xyz; first two rotation-matrix columns, column-major flattening |
| Cube position and rotation | 9 | Same arena and rotation representation |
| Goal position | 3 | Arena xyz |
| Cube minus TCP; goal minus cube | 6 | Arena vectors |
| Cube linear and angular velocity | 6 | Arena m/s and rad/s |
| Total | 51 | Current measurement only |

Record measured TCP at collection time so consumers need not instantiate a
robot. The same feature builder runs for teacher data and student evaluation.
Rotation columns retain unit scale; continuous quantities use frozen G1-only
per-feature statistics with physical scale floors. Count observation clipping;
do not normalize near-constant features by arbitrarily tiny variance. Never
use development/test states for statistics. Schema metadata fixes ordering,
frame, units and normalization. These observation additions are explicit changes
from the original 28-feature plan.

Do not expose expert segment, elapsed segment time, target waypoint, success,
grasp Boolean or planner progress to the policy. A fallback may use previous
measured observations and previous policy requests; its reset/padding contract
must be identical during training and inference.

### Seven policy actions, version 2

The student predicts normalized arena-frame translation delta (3), arena-frame
rotation-vector delta (3), and absolute active-finger opening (1). At each step:

```text
u = clamp(network(observation), -1, 1)
p_target = p_measured + translation_scale * u[0:3]
R_target = Exp(rotation_scale * u[3:6]) @ R_measured
finger_target = 0.02 * (u[6] + 1) meters
```

Initial controller qualification candidates are 0.02 m and 0.10 rad per component
per control step. They are provisional controller settings, not tuned test-set
hyperparameters. Freeze them only after training-side support/tracking audits and
expert qualification. There is no accumulated hidden pose target. Rotation
composition, scale and clipping are serialized in the action descriptor.

Reuse the absolute arena pose/IK behavior of `EefPoseAction`, with a small generic
relative-pose action adapter; reuse continuous `ParallelGripperAction`. IK uses
measured joints as seed and ordinary drives. Failed IK holds measured arm
positions and emits a diagnostic; no teacher substitution is permitted. Audit
joint-target discontinuity even when IK succeeds. Invalid/nonfinite requests
fail explicitly; clipping and IK failures are counted in every evaluation.

The current reset is not top-down. Seven-dimensional control preserves that reset
and allows the policy to learn orientation. A four-dimensional XYZ/gripper variant
would require a separate, disclosed reset or orientation-controller change and
fresh qualification; it is not the proposed default.

### Teacher commands and causal labels

Keep handwritten expert planning and physical acceptance. Adapt the expert to
produce Cartesian references, compute bounded requests from actual measured
state, and execute through the exact student adapter at 30 Hz. Planning the
second segment still uses the measured post-grasp state. Expert-only waypoints
and phase logic remain private.

Record `(observation_t, requested_action_t)` before applying the action. Store
processed eight-joint targets separately if needed for diagnostics. Adjacent
measured state differences are plant responses, not command labels. FK of old
planned joint targets may initialize references but is not qualified data until
execution through the new controller succeeds physically.

Audit teacher dwell durations and conflicting near-identical observations. Where
blind timing produces unlearnable transitions, use measured tracking/contact
settling to advance the teacher reference, with bounded timeouts. This is an
explicit teacher-design change to validate, not a claim that every dwell is bad.
Do not add student queries or feedback-driven demonstrations in this milestone.

### Recording and ODS schema must agree

The existing `actions` stream stores eight expert controller targets;
`_validate_worker_dimensions()` checks `expert_action_spec.width`. Merely changing
`action_dim` to seven is incorrect. First inspect/reuse the environment's existing
`policy_request` recording route, then expose the resolved recorded-action
descriptor to ODS allocation and validation. Keep legacy expert-command mode
unchanged; do not relabel eight-joint data as seven policy requests.

The schema records semantics, feature order, dimensions, frames, scales, control
period and controller version, not just tensor width. Worker startup checks the
actual observation and recorded-action descriptors before publishing samples.
Checkpoint loading rejects incompatible schemas. Test causal pairing through
the real recorder and producer, not only a handcrafted tensor batch.

## Learner and minimal diagnostic comparisons

Start with ManiSkill-style MLP, linear output and MSE in normalized **controller
request space**, Adam 3e-4, batch 256 independent valid frames and gradient norm
limit 1. Inference clamps at the adapter; report out-of-range predictions before
clamping. Baseline loss is the mean over seven normalized components; log
translation, rotation and gripper terms separately. Do not inherit gripper weight
5 without an ablation. Synthetic save/load and numerical tests remain necessary
but are not learning qualification.

Use one live-generated diagnostic snapshot and common seeds for these ordered
comparisons. Run only enough variants to resolve the next uncertainty:

1. Existing residual GRU versus the same model trained on arm targets
   `(expert_target - measured_qpos) / 0.25`, with loss in that residual space.
   Audit target support first; never silently clip supervised targets. This is
   the cheapest test of the loss-scale hypothesis.
2. Add geometry while retaining that joint action contract and model. Then
   compare MLP with GRU on the same observations/actions/data. This separates
   geometry and model effects; both must complete the same physical task.
3. Qualify the new Cartesian teacher/controller, then compare joint and Cartesian
   candidates using the same reset panel and physical predicate. Recollect labels
   through each controller; these are controller-system comparisons, not a claim
   of changing only network outputs on identical trajectories.
4. If current-state MLP cannot resolve transitions, inspect action ambiguity and
   try four-frame measured history, then the existing small GRU. Short history
   must stop at episode/continuity boundaries. A larger Isaac-style GMM recurrent
   policy is a later evidence-driven option, not an automatic escalation.

For comparisons, report supervised-target counts and wall time, not only updates.
The old 64-by-24 sequence batch contains 1,536 targets; the proposed frame batch
contains 256. Equal optimizer updates across them are not equal supervision.

## Qualification gates and stopping rules

These thresholds are proposed engineering decisions; only the final 80% target
comes from the original acceptance plan. All rollout gates use the unchanged
complete physical predicate and independent student execution.

| Gate | Work and bounded initial budget | Required result / next action |
| --- | --- | --- |
| G0: contracts | Unit checks plus same-reset recorded-command replay; no optimizer | Correct causal pairing, units, ordering, reset and failure behavior; replay reproduces accepted task execution |
| G1: teacher | 40 independent qualification scenes through the exact student controller | At least 38 complete successes; failures retained; otherwise fix controller/teacher before BC |
| G2a: one episode | One freshly generated accepted episode; up to 2,000 updates; evaluate every 250 | Student completes its training reset in at least 9/10 fresh rollouts; repeat trials check repeatability, not generalization |
| G2b: four episodes | Four training reset states; up to 3,000 updates | At least 18/20 complete rollouts, five per reset; if missed, stop and diagnose instead of scaling data |
| G3: frozen generalization | Live ODS G1 with 32 episodes; up to 5,000 MLP updates, 20 fixed dev states every 500 | At least 16/20 dev successes before streaming; an explicitly reported 128-episode frozen ablation may diagnose coverage, but changes the diagnostic data budget |
| G4: streaming | Five generations, 32 slots, 1,000 updates/round, matched frozen control | Parameter changes, at least two generations used in supervision, independent execution, reload/resume/closure and provenance gates pass |
| G5: final | Freeze design/selection rule, then evaluate selected checkpoints on 100 newly reserved states | At least 80/100 complete successes; report failures and uncertainty, not only lift |

If a gate times out, save the best checkpoint and failures, identify whether
teacher replay, fitting, transition ambiguity or free-rollout drift failed, and
change one hypothesis before rerunning. No silent budget extension, smaller test
distribution, longer task horizon, expert takeover or relaxed success predicate.
Small-data diagnostic snapshots are generated by ODS during the experiment;
they are not external offline datasets or a new generic replay facility.

## Streaming organization and controls

Keep a CPU shared cache of 32 accepted episodes, one expert producer and one
independent evaluation process. The parent owns the learner and manual refill.
Keep the current lock/copy/validity/publication/error contracts. The MLP consumes
`sample_batch(batch_size=256, chunk_size=1)` and no recurrent prefixes.

Use initial G1 in round one. For later rounds request one refill, perform up to
250 overlap updates, wait for the requested generation to complete, then perform
750 updates. Record the actual generation of every supervised frame: refill may
publish rows incrementally, so overlap batches need not be exclusively old data.
Keep a bounded 30-second no-sample-window wait distinct from a configurable
full-generation deadline, which is frozen from qualified producer timings before
training. Producer failure terminates immediately. Deadline failure is explicit.

Use the [benchmark matrix](BENCHMARK.md#experiment-arms): a strong offline baseline
precollects all 160 episodes and may load them into RAM; matched offline delivery
and serialized ODS isolate mechanisms. Frozen G1 remains a data-refresh ablation,
not the sole offline comparator. Report cold and data-ready timing separately,
including hardware and repeated-dataset reuse costs. Optional snapshots used to
match supervision have their memory and I/O costs recorded.

Keep diagnostic accounting in the runner: distinct episodes/frames sampled,
per-generation exposure, data age, accepted frames never sampled before overwrite,
production/learner/wait durations, memory and clipping/IK failures. Identity uses
engine run ID, episode ID, generation and a within-episode frame index; expose
the frame index in the same locked copy if absent. Count only valid supervised
frames. Do not confuse repeated prefixes or sample counts with unique coverage.

## Evaluation, display and reproducibility

Keep the existing 20 development states for iteration. Previously examined test
seeds 30000–30099 now serve historical comparison only. Reserve a fresh disjoint
100-state manifest before final redesigned runs; do not use it for normalization,
hyperparameters or checkpoint selection. Inspect it only after the design is
frozen. Run the qualified online/frozen pair with three training seeds; report
each result and variation. Claim robust attainment only if every reported online
seed meets the 80/100 target; a single qualifying run is labeled as such.

Select by development complete success, then lift and final goal error, including
the initial/resumed checkpoint in the candidate set. Keep per-scene reports and
binomial uncertainty for test success; do not treat repeated trials on the same
reset as independent scene coverage.

Viser shows the last evaluated immutable checkpoint and its version. TensorBoard
shows complete success, lift, goal error, episode length, action-component losses,
IK/clipping, expert failures, generation exposure and costs. Report transport,
release and settling failures as diagnostics only; no phase labels enter the
student. Before/after videos use the same predetermined scenes and retain failure.

Checkpoints include model type, feature/action schemas, controller parameters,
normalization, optimizer/RNG, step, source IDs and resolved configuration. Resume
recreates ODS with a new run ID; no exact shared-cache recovery is claimed. Old
28-to-8 checkpoints remain loadable through their original explicit model path,
never silently converted to the new policy/controller.

## Implementation order and ownership

1. Fix the two [confirmed review findings](REVIEW.md) with focused regression
   coverage. Add residual-loss support/scale diagnostics and small-data rollout
   modes to the existing showcase runner.
2. Add measured geometric observations to the official task; run the minimal
   joint-model comparisons before assuming Cartesian conversion is necessary.
3. Implement the generic relative-pose action term and explicit recorder/ODS
   action-schema integration. Requalify teacher replay and 40-scene performance.
4. Add the MLP and conditional history comparator in the example. Pass G2/G3,
   then activate G4 streaming, controls and G5 reserved evaluation.
5. Update the one-entry launch instructions and retain a complete failed/successful
   experiment report. No new `train-il` command or RL registry integration.

Task owns physics, expert and success. Action manager owns generic control. ODS
owns production, bounded storage, publication and provenance. The example owns
the small learner, experimental schedule and reporting. Add focused production
tests for coordinate/rotation composition, IK failure, partial reset, schema
mismatch, causal recording, padding IDs and provenance. Validate model loss and
checkpoint behavior deterministically; validate learning through physical runs.

No DAgger, student-feedback collection, cross-generation aggregate replay,
pretrained models or RLInf integration is included. Those remain separate scope
decisions after this showcase is demonstrably competent.
