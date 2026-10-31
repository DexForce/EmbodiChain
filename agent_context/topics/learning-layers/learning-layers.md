# Differentiable learning layers

The [Differentiable Kinematics guide](../../../docs/source/overview/learning/differentiable_kinematics.md)
owns user-facing FK/IK setup, training examples and gradient/state contracts.

## Owners and dependency direction

`compute/kinematics/{dls,pose_error,serial_chain,types}.py` owns fixed-step
DLS, SE(3) residuals, copied PK geometry, and shared configuration/results.
These modules do not import learning or simulation. `SerialChainKinematics`
returns TCP poses and geometric Jacobians from one FK pass, retaining dJ/dq.

`learning/layers/fk.py` owns `DifferentiableFKLayer`: joints to root-frame TCP
poses through a pure FK path; `evaluate` also returns geometric Jacobians from
one FK pass. `DifferentiableIKLayer` composes this FK layer with fixed-step DLS.
IK keeps its `kinematics` child name for checkpoint compatibility.
`learning/layers/toppra.py` owns `DifferentiableRetimingLayer`, composing the
simulation-independent public `compute.trajectory.retime_joint_paths()` API.
It checkpoints fixed motion-limit buffers and numerical settings, pads quantity
outputs to a fixed width and loads on demand so FK/IK imports do not load Warp.
`JointPathTimingResult.sample_lengths` distinguishes failed rows (zero) from
stationary successes (two). Callers resolve robot caps, preserve input graphs
and own geometric validation. Gradients are first-order and local to fixed
filtering, sampling topology and active constraints.
`learning/models/` owns general MLPs, feature normalization
and configuration-driven MLP construction. These shared components do not import
RL, TensorDict or simulation. Concrete NeRD models, codecs and the shared-SL
recipe live in the bundled `embodichain_models.dynamics.nerd` extension;
teacher collection, historical training and research belong to
`examples/learning/nerd/`. See
[neural dynamics](../neural-dynamics/neural-dynamics.md) for model, SL integration,
dataset and checkpoint ownership. RL policy interfaces, actor-critic composition and
policy registration/builders belong to `learning/rl/policies/`; RL code consumes
the shared networks and normalizer. See [RL learning](../rl-learning/rl-learning.md)
for those contracts.

Training, task-parameter optimization and the differentiable benchmark compose
FK or IK layers directly. Use `torch.no_grad()` for forward-only inference.
There is no registered Robot solver adapter: callers provide root-frame targets
and explicitly configure the layer's TCP and limits. Existing Robot solver binding
and arena conversion remain in [IK solvers](../ik-solvers/ik-solvers.md).

Supervised optimizer and offline-run ownership belongs to
[supervised learning](../supervised-learning/supervised-learning.md).

## Gradient and state boundaries

Targets are root-frame TCP matrices; seed batch axes match the target or contain
only a joint axis. Training inputs must match model device and dtype. Both float32
and float64 are supported. FK inputs follow the model's joint order; FK does not
project limits. Caller-owned pose losses can train joint predictors through FK.
FK and Jacobian derivatives remain connected through
all iterations. Final tolerances produce boolean metadata; no convergence-based
Python exit occurs. Failure returns the final iterate and continuous residuals.
Hard limits have piecewise gradients; disabling projection does not disable the
physical-limit success check. The shortest rotation residual has a pi branch cut.
Only first-order input gradients are part of the public contract.

Layers own fixed TCP/limit buffers, copy geometry, and explicitly migrate PK
internals during `.to()`. Checkpoints verify fixed-model identity and restore
numerical settings and buffers; construct the same geometry before loading.
`load_state_dict(assign=True)` also migrates fixed PK geometry to the loaded
buffers' device/precision, including loads through an enclosing network.
They contain no optimizer or recurrent seed. `.eval()` preserves gradients.
Training code owns state, reset and detach boundaries. No call may overwrite
another outstanding graph's intermediates.

## Training and validation

Model and layer callers own task losses and optimization. Formal task deployments
belong to the task extension. See [RL learning](../rl-learning/rl-learning.md)
for trainer/collector lifecycle.

`examples/learning/differentiable_fk/learn_pose_reaching.py` trains an example-local
joint predictor through fixed FK with TCP pose supervision and the shared SL
trainer. Generating joint solutions are discarded; validation selects weights
before independent local test evaluation. Its viewer compares initial and frozen
trained predictions using one CPU worker and rejects stale target edits. Example
smoke commands own launch/artifact checks; production tests own FK numerics.

`examples/learning/differentiable_ik/learn_grasp_yaw.py` trains an example-local
conditional yaw predictor through fixed-step IK, preserving grasp position and
downward approach. Validation alone selects checkpoints; independent target
and current-state holdouts compare constant, learned, teacher-supervised and
numerical methods. `view_learned_yaw.py` evaluates frozen checkpoints and discards
stale asynchronous results after target edits. Numerical reference baselines are
part of the training/evaluation entry point. The demo does not execute physical
motion; keep task predictors out of production policy registries until a reusable
contract exists.

`examples/learning/differentiable_toppra/optimize_path.py` optimizes free Franka
path nodes through physical duration with fixed joint targets. Its Viser viewer
compares connected and detached objectives and shares the native physical clock.
The example owns projected first-order updates, sampled checks and artifact
replay; it does not train a network or imply policy-quality improvements.

Tests live under `tests/compute/kinematics/` and `tests/learning/layers/`.
Validate FK, dJ/dq, full IK and recurrent-state gradients separately; include
nonzero TCP offsets, repeated solves, checkpoint migration and import boundaries. GPU validation runs serially with `--run-gpu`.
The unified kinematic benchmark's `differentiable` mode measures training and
forward-only baselines, including FK/Jacobian and DLS component costs.
