# Differentiable learning layers

## Owners and dependency direction

`compute/kinematics/{dls,pose_error,serial_chain,types}.py` owns fixed-step
DLS, SE(3) residuals, copied PK geometry, and shared configuration/results.
These modules do not import learning or simulation. `SerialChainKinematics`
returns TCP poses and geometric Jacobians from one FK pass, retaining dJ/dq.

`learning/layers/` owns `KinematicsLayer` and `DifferentiableIKLayer` Torch
module interfaces. `learning/models/` owns MLP, policy interfaces, actor-critic
models, normalization, and policy registration/builders. The registry is
`learning/models/registry.py`; callers import models from `embodichain.learning.models`.
Model exports load on demand so standalone MLP/layers do not load RL, TensorDict
or simulation. RL algorithms and trainers consume these models.

Training, task-parameter optimization and the differentiable benchmark call
`DifferentiableIKLayer` directly. Use `torch.no_grad()` for forward-only inference.
There is no registered Robot solver adapter: callers provide root-frame targets
and explicitly configure the layer's TCP and limits. Existing Robot solver binding
and arena conversion remain in [IK solvers](../ik-solvers/ik-solvers.md).

## Gradient and state boundaries

Targets are root-frame TCP matrices; seed batch axes match the target or contain
only a joint axis. Training inputs must match model device and dtype. Both float32
and float64 are supported. FK and Jacobian derivatives remain connected through
all iterations. Final tolerances produce boolean metadata; no convergence-based
Python exit occurs. Failure returns the final iterate and continuous residuals.
Hard limits have piecewise gradients; disabling projection does not disable the
physical-limit success check. The shortest rotation residual has a pi branch cut.
Only first-order input gradients are part of the public contract.

Layers own fixed TCP/limit buffers, copy geometry, and explicitly migrate PK
internals during `.to()`. Checkpoints verify fixed-model identity and restore
numerical settings and buffers; construct the same geometry before loading.
They contain no optimizer or recurrent seed. `.eval()` preserves gradients.
Training code owns state, reset and detach boundaries. No call may overwrite
another outstanding graph's intermediates.

## Training and validation

Model and layer callers own task losses and optimization. Formal task deployments
belong to the task extension. See [RL learning](../rl-learning/rl-learning.md)
for trainer/collector lifecycle.

Tests live under `tests/compute/kinematics/` and `tests/learning/layers/`.
Validate FK, dJ/dq, full IK and recurrent-state gradients separately; include
nonzero TCP offsets, repeated solves, checkpoint migration and import boundaries. GPU validation runs serially with `--run-gpu`.
The unified kinematic benchmark's `differentiable` mode measures training and
forward-only baselines, including FK/Jacobian and DLS component costs.
