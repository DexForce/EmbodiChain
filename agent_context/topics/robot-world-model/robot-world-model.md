# Robot world model

## Ownership and entry points

`embodichain_models.dynamics.rwm` is the canonical model package. Its `model.py`
owns body-frame state increments; `kinematic.py` owns `KinematicWorldModel`,
with heading-frame increments and full-body FK features. `robot_state.py` owns the
canonical state representation and SO(3) transforms. They have no simulation or example
dependency. Shared MLPs/normalizers stay in core `learning.models`; optimizer
updates, run artifacts and recovery stay in [SL](../supervised-learning/supervised-learning.md).

`supervised.py` owns the autoregressive objective and replay-window adaptation.
`predictor.py` retains graph-connected prediction state. `sl_recipe.py` exports
`build_rwm_trainer` for `train-sl`; the same symbol is exported from the model
family. `examples/learning/rwm/` owns controlled corpus generation and acceptance
through `embodichain run-example learning/rwm`; private support is not a model API.
The [RWM guide](../../../docs/source/features/neural_dynamics/rwm.md) owns the
user-facing principles, tensor layout, prediction and replay examples.

`examples/learning/motion_tracking/fit_world_model.py` collects official G1 physical
transitions and fits the kinematic variant through the same `SupervisedLearner`.
Task/reference data remain outside the model package.

## State, graph and data boundaries

Canonical state is position, row-major body-to-world rotation, world linear and
angular velocity, then ordered joint positions and velocities. Both variants use
right-multiplied exponential rotation coordinates. The kinematic variant uses
heading axes for position/linear-velocity changes and each state's own body axes
for angular-velocity changes. Its input features share
`compute/kinematics/features.py` with tracking tasks.
Fixed control period, action semantics and joint order enter model/checkpoint
identity. Normalization changes only from explicitly selected measured training
transitions and is frozen during prediction differentiation.

The objective retains the full multi-step graph and excludes invalid suffix
padding before model/error arithmetic. Predictor reset allocates replacements,
preserving outstanding results and gradients through unselected rows. Freeze
model parameters and normalization for action/state differentiation, and finish
dependent backward work before changing weights. The objective supports explicit
feature weights and temporal mean/sum reduction; complete-window adaptation
rejects episode-crossing or unusable transitions before graph construction.

`replay.py` owns fixed-byte-budget device storage. Append copies detached,
owned physical transitions before reset. Windows/history respect episode
boundaries and ring overwrite; sampling returns owned tensors. Declared auxiliary
fields preserve task progress and controls within the allocation budget. Use
`history_complete` to exclude starts whose required history was overwritten;
real episode-start padding remains complete. Acquisition, reset and optional
replay refill remain caller-owned.

## Validation

`tests/models/dynamics/rwm/` covers state transforms, replay ownership/boundaries,
NaN padding, state/action gradients, model identity and exact offline recovery.
The controlled three-seed fitting protocol and its unchanged acceptance gate
live in `examples/learning/rwm/README.md`; historical reports retain their
recorded source/data hashes. These runs qualify fitting/interfaces, not robot
policy performance.

## Online motion learning

`embodichain_models.policies.motion_tracking` owns the goal encoder/decoder actor
and scalar critic. The official task's `tracking/motion_model.py` composes shared
FK, reference queries and reward/continuation rules into a tensor-only environment.
Replay starts include clean history, previous actions, physical context and frame
progress. Clip suffixes use validity masks; a complete horizon retains gradients
until the trainer finishes backward, then `detach_state()` preserves values and
reference progress without drawing new noise.

The online recipe keeps `trainer / policy / algorithm` blocks and a task-owned
`workflow` block. `trainer.learning_env` resolves `MotionTrackingModelRL` through
`build_learning_policy_runtime(resources=...)`; `resources` shares the live WM,
motion library, task and Replay start provider without copying their state.
`MotionTrackingPolicy.from_config` owns network configuration and observation/action
space validation. Collection and evaluation use separate `workflow.collection_env`
and `workflow.evaluation_env` paths. Checkpoints retain the full recipe, and the
entry normalizes earlier flat recipe fields before compatibility checks.

`learning/dynamics/online.py` owns prefill/warmup/update counters and fractional
update credit. `examples/learning/motion_tracking/train.py` composes physical
collection, shared SL and `DifferentiableTrainer.update_once()`. It freezes the WM
around the full policy update. The recipe checkpoint includes all three networks,
optimizers, schedule, normalizers, RNGs and reference sampling statistics. Replay
is refilled from a newly reset physical environment after restoring learning state.
The task's `agents/model_based*.yaml` configs distinguish functional tests,
four-clip development, complete 40-clip convergence and frozen-WM policy tests.
The full recipe uses 60,000 collection iterations, no model-only warmup and
24,000 Actor / 240,000 WM / 480,000 Critic updates. Its default physical route is
`env.newton.yaml` with the `mujoco_warp` solver; `env.dexuni.yaml` retains the
DexUni alternative. `save_every: 0` retains final checkpoints only.
`--wm-checkpoint` initializes the fixed-model recipe from fitted or online WM
weights, while `--no-evaluate-final` separates successful training from evaluation.
The installed entry is `embodichain run-example learning/motion_tracking/train.py`.

The full recipe enables `compile_mode: reduce-overhead` for the actor, complete
WM objective, tensor-only model transition and reference-window FK. Sampling and episode bookkeeping
remain outside compiled regions. `compute/kinematics/matrix.py` shares fused compiled FP32/FP64 small-matrix
products across FK, model integration and task features; eager calls retain Torch
matrix kernels. Tracking orientation error computes the squared shortest angle
directly from rotation trace and antisymmetric entries. The model environment has
no auto-reset and supplies next observations without a redundant terminal copy. Actor compilation
uses `nn.Module.compile`, preserving parameter ownership and checkpoint keys.
Physical collection starts its own CUDA graph step before copying measured data
into Replay. Partial-reset features use dynamic compilation without CUDA graph
capture; full-batch task analytics retain the selected compilation mode.
The motion recipe uses fused Adam on CUDA for WM, Actor and Critic, with the
existing CPU Adam implementation on CPU.
The differentiable trainer releases consumed rollout snapshots after gradient
accumulation and normalization updates, before the next collection or Critic fit.
`episode_context_in_info` controls JSON provenance output independently of episode
validity checks. Both options are runtime settings outside checkpoint identity.

`evaluate_online.py` loads a checkpoint in a separate physical process and runs
all selected clips from their first frame, with deterministic actions, no observation
or reset noise and a full-clip horizon. Optional trajectories contain measured successor state, the matching reference,
active masks, joint names and tracked-link names for visualization. Per-clip failures and normalized full-clip return measure
convergence. The [online recipe guide](../../../examples/learning/motion_tracking/README.md)
owns commands and output semantics. `latest.pt` retains complete learning state;
`policy-NNNNNN.pt` retains evaluation weights at configured save/evaluation intervals.

Quality evaluation defaults to `--mode quality`; `--mode physics-regression`
requests Warp run-to-run determinism. `--max-steps` bounds a functional check and
marks the result as a partial-horizon evaluation. Evaluation records its own
hardware, software, precision and data provenance.

`evaluate_world_model.py` loads a frozen online checkpoint and collects fresh
physical episodes under policy or reference-PD control. It compares open-loop
predictions and a state-hold diagnostic at 1/8/16/32 steps, respecting episode
boundaries and retaining model/normalizer values. State, reward and termination
errors measure each prediction endpoint; the supervised training loss aggregates
weighted errors over its fitted window. Training results also expose actual Adam
step ranges and per-invocation first-call durations, including lazy initialization.

The motion example's `prepare_lafan1.py` downloads the pinned G1 retargeting
source and preserves original-data, retargeting-project and download-mirror
notices. `download_pretrained.py` resolves a model revision once and downloads
the G1 inference bundle from `DexForceAI/embodichain_model`; the record stores
the immutable revision. Evaluation loads it with `weights_only=True`.
`replay.py` consumes the self-contained evaluation NPZ and uses public
`RobotMarkers` and `SimulationManager.sync_render_state()` for FastRT playback
without physics steps. The visualization interfaces are supplied by #833.
