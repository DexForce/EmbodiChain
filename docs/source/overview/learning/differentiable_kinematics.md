# Differentiable Kinematics

Use robot kinematics and joint-path retiming inside a Torch network or task loss
without starting a simulation environment. `DifferentiableFKLayer` maps joints
to TCP poses. `DifferentiableIKLayer` composes the FK layer with fixed-step damped
least squares (DLS) to map target TCP poses to joints. Both support supervised
learning, task-parameter optimization and forward-only inference.
`DifferentiableRetimingLayer` applies TOPPRA to a joint path so a loss can use
its execution duration.

| Call | Input | Output | Input gradients |
| --- | --- | --- | --- |
| `fk(qpos)` / `fk.forward(qpos)` | Joints `(..., dof)` | TCP poses `(..., 4, 4)` | Joint positions |
| `fk.evaluate(qpos)` | Joints `(..., dof)` | TCP poses and geometric Jacobians `(..., 6, dof)` | Joint positions, including Jacobian derivatives |
| `fk.jacobian(qpos)` | Joints `(..., dof)` | Geometric Jacobians `(..., 6, dof)` | Joint positions |
| `ik(target_xpos, qpos_seed)` / `ik.forward(...)` | TCP targets and optional joint seeds | Joints `(..., dof)` | Targets and supplied seeds |
| `ik.solve(target_xpos, qpos_seed)` | Same as `ik.forward` | Final joints, success mask and pose residuals | Floating outputs retain target and seed gradients; the mask is boolean |
| `retime(path, velocity_limits, acceleration_limits)` | Joint waypoints `(B, N, J)` and shared motion limits | Joint trajectory, arrival intervals and validity | Waypoints and tensor motion limits |

The layers are standalone `nn.Module` components. For simulator-backed Robot
solving and differential pose control, see the
{doc}`motion solvers </overview/sim/motion/solvers/index>`. Full signatures and
configuration fields are in the
{doc}`learning API </api_reference/embodichain/embodichain.learning>` and
{doc}`kinematics compute API </api_reference/embodichain/embodichain.compute>`.

## Shared initialization

The FK and IK examples use the same fixed Franka model on CPU. They run
independently after this shared initialization.

```python
from pathlib import Path

import torch

from embodichain.compute.kinematics import (
    DifferentiableIKCfg,
    SerialChainKinematics,
    pose_error,
)
from embodichain.data import get_data_path
from embodichain.learning.layers import DifferentiableFKLayer, DifferentiableIKLayer

model = SerialChainKinematics.from_urdf(
    Path(get_data_path("Franka/Panda/PandaWithHand.urdf")).read_text(),
    root_link_name="base",
    end_link_name="fr3_hand_tcp",
    device="cpu",
    dtype=torch.float32,
)
tcp = torch.eye(4, device="cpu", dtype=torch.float32)
fk = DifferentiableFKLayer(model, tcp=tcp)
```

`end_link_name` selects the model's terminal frame. `tcp` is the fixed transform
from that frame to the task's tool frame; identity uses the terminal frame
directly. The FK layer copies the model and owns its TCP and physical-limit
buffers. Configure a different TCP or physical limits at construction when the
task requires them.

## Differentiable FK

Train a joint predictor using TCP pose supervision. This example creates a small
predictor and illustrative reachable labels, then backpropagates position and
rotation errors through FK into the predictor's parameters. Replace the example
observations and labels with task data for training.

```python
predictor = torch.nn.Linear(
    3, fk.dof, device=fk.tcp.device, dtype=fk.tcp.dtype
)
observation = fk.tcp.new_ones(8, 3)
with torch.no_grad():
    target_pose = fk(fk.default_seed().expand(8, -1))

q_pred = fk.default_seed() + 0.1 * predictor(observation)
pose_pred = fk(q_pred)
position_error, rotation_error = pose_error(pose_pred, target_pose)
loss = (
    position_error.square().sum(-1)
    + rotation_error.square().sum(-1)
).mean()
loss.backward()  # Gradients reach predictor.weight and predictor.bias.
```

`fk(qpos)` computes poses without constructing an explicit geometric Jacobian.
Use `fk.evaluate(qpos)` when the loss also needs Jacobians: it returns both from
one FK pass. Geometric Jacobians are expressed in the chain root frame, with
linear rows preceding angular rows.

FK evaluates the supplied joints without clipping them to physical limits. The
training task owns loss weights, joint-limit handling and any posture loss.
TCP pose supervision alone need not select a unique joint posture for a redundant
robot; use an output mapping or additional objective when the task needs one.

## Differentiable IK

IK takes a target pose and an optional starting joint configuration. The example
below creates its own seed and nearby target, then differentiates a pose-residual
loss through the unrolled solver.

```python
ik = DifferentiableIKLayer(fk, cfg=DifferentiableIKCfg(max_iterations=16))
qpos_seed = fk.default_seed()[None].clone().requires_grad_()
target_pose = fk(qpos_seed.detach() + 0.01).detach().requires_grad_()

result = ik.solve(target_pose, qpos_seed=qpos_seed)
loss = (
    result.position_error.square().mean()
    + result.rotation_error.square().mean()
)
loss.backward()  # Both qpos_seed.grad and target_pose.grad are connected.
```

Use `ik(target_pose, qpos_seed=...)` when a network needs only the joint tensor.
`ik.solve(...)` additionally returns a boolean `success` mask and continuous
position/rotation residuals. Seeds are `(dof,)` or match all target batch axes;
omitting a seed uses bounded joint midpoints, or projected zero for unbounded
joints. There is one seed per target and no multi-start search.

Each DLS iteration computes FK and the TCP Jacobian once, retaining their
derivatives, then applies:

```text
q_next = q + step_size * J.T @ solve(J @ J.T + damping**2 * I, error)
```

The iteration count is fixed. Position and rotation tolerances classify the final
iterate without stopping the loop early. `success` also requires finite values
and physical-limit validity. A failed solve returns the last iterate and its
differentiable residuals. DLS is a local solver; it does not guarantee global
reachability.

Joint-limit projection applies to the seed and every update by default. Its
gradient is zero outside a clamp interval. Set `enforce_joint_limits=False` for
an unconstrained objective with caller-owned limit handling; the final success
check still uses physical limits. Numerical options belong to
`DifferentiableIKCfg` in the compute API.

## Shared gradient and state contracts

These contracts apply to the FK and IK layers.

- **Frames and units:** poses are valid finite SE(3) matrices in the configured
  chain root frame and refer to the configured TCP. Callers own world-to-root
  conversion. Joint inputs follow `model.joint_names`. Translation and prismatic
  coordinates use metres; revolute joints and rotation vectors use radians.
- **Batch, device and precision:** FK joints are `(..., dof)` and poses are
  `(..., 4, 4)`; leading batch dimensions are retained. Inputs must match the
  layer's device and dtype. Float32 and float64 are supported. `.to(...)` migrates
  copied PK geometry as well as registered buffers.
- **Gradients:** only first-order input gradients are part of the public
  contract. Geometry and TCP are fixed configuration, not trained parameters.
  `pose_error` returns translation and shortest root-frame rotation vectors;
  the zero-rotation derivative is finite, while the shortest rotation has a
  branch cut at pi. `.eval()` preserves gradients; use `torch.no_grad()` for
  forward-only inference.
- **Caller-owned state:** neither layer owns an optimizer or recurrent seed.
  Training code owns losses, rollout state, reset and detach boundaries.
  Outstanding calls retain independent computation graphs.
- **Checkpoints:** reconstruct the same robot geometry before loading a state
  dict. Checkpoints store TCP/limit buffers, geometry identity and, for IK,
  numerical settings. `load_state_dict(assign=True)` also migrates fixed
  geometry to the loaded buffers' device and dtype, including nested loads.
  IK retains the `kinematics` constructor argument and child name for existing
  state dicts.

## Differentiable TOPPRA

`DifferentiableRetimingLayer` fits a cubic spline through the retained joint
waypoints and uses TOPPRA (time-optimal path parameterization by reachability
analysis) to compute a rest-to-rest speed profile under joint velocity and
acceleration limits. It samples each moving trajectory uniformly in physical
time. Gradients through spline fitting and timing let a duration loss guide
changes to the waypoint geometry.

This standalone CPU example differentiates duration with respect to the two
interior waypoints. The start and goal are fixed joint configurations.

```python
import torch

from embodichain.learning.layers import DifferentiableRetimingLayer

start = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
free_nodes = start.new_tensor([[0.25, 0.4], [0.70, -0.15]], requires_grad=True)
goal = start.new_tensor([[1.0, 0.2]])
path = torch.cat((start, free_nodes, goal), dim=0).unsqueeze(0)
velocity_limits = start.new_tensor([1.0, 0.6])  # rad/s
acceleration_limits = start.new_tensor([2.0, 1.0])  # rad/s²

retime = DifferentiableRetimingLayer(sample_count=65, grid_size=100)
q, qd, qdd, dt, success, lengths = retime(path, velocity_limits, acceleration_limits)
if not success.item():
    raise RuntimeError("The example path could not be retimed.")

duration = dt.sum(-1)  # Seconds, one value per path.
duration.sum().backward()  # Gradient in free_nodes.grad, in seconds/radian.
```

### Inputs and results

Paths have shape `(B, N, J)` for `B` paths, `N` waypoints and `J` joints, with
at least two waypoints per path. The in-tree Warp solver supports float32 and
float64 on CPU and CUDA. Floating outputs use the waypoint dtype. Each motion
limit tensor is finite and shared across the batch. Supply a nonnegative scalar,
`(J,)` nonnegative magnitudes or signed `(J, 2)` lower and upper bounds containing
zero. For padded inputs, pass an integer `path_lengths` tensor of shape `(B,)`
with valid prefix lengths between `2` and `N`.

| Output | Shape | Meaning |
| --- | --- | --- |
| `q`, `qd`, `qdd` | `(B, M, J)` | Joint position, velocity and acceleration at each sample |
| `dt` | `(B, M)` | Seconds since the previous sample, with a zero first entry |
| `success`, `lengths` | `(B,)` | Boolean solver status and integer valid output sample count |

`sample_count` fixes `M` (default `64`, minimum `2`). `grid_size` sets the
minimum timing grid size (default `100`, minimum `3`). The solver can enlarge
the grid to fit the path. Use `dt.cumsum(-1)` for sample arrival times and
slice by `lengths` to exclude padding. A stationary path succeeds with two valid
samples and zero duration. Padding repeats the final position with zero
derivatives and intervals. Failed paths have length zero and zero-filled
outputs. Check `success` before reducing durations into a batch loss.

### Gradients and module state

Floating outputs retain first-order gradients to waypoints and tensor limits.
Backward holds waypoint filtering, timing-grid and sampling choices, and active
constraints fixed. Gradients can be discontinuous or undefined at those switches.
Failed paths contribute zero gradients. For stationary paths, gradients flow
only through the retained position.

The layer's state dict is empty. Keep `sample_count` and `grid_size` in the
caller's configuration and register fixed limits as buffers on the enclosing
model. `.eval()` preserves autograd. Use `torch.no_grad()` for forward-only
retiming. Solver success describes timing under the supplied motion limits.
Callers supply robot-specific limits and validate joint position bounds and
collision clearance for the resulting path.

For a function interface with time-based sampling via `sample_dt`, see
`retime_joint_paths` in the
{doc}`compute API </api_reference/embodichain/embodichain.compute>`.

For retiming a simulated robot through the motion planning interface, see
{doc}`ToppraPlanner </overview/sim/motion/planners/toppra_planner>`.

## Examples and measurement

- [FK pose reaching](https://github.com/DexForce/EmbodiChain/blob/main/examples/learning/differentiable_fk/README.md)
  trains a joint predictor through TCP pose loss and compares initial and learned
  predictions on held-out cases and edited targets.
- [IK grasp orientation](https://github.com/DexForce/EmbodiChain/blob/main/examples/learning/differentiable_ik/README.md)
  trains a conditional yaw predictor through IK and compares frozen predictions
  against heading and numerical references.
- The CPU Franka demo uses duration gradients to optimize free waypoints and
  compares the original and optimized trajectories on a shared physical clock
  in Viser. [TOPPRA path optimization](https://github.com/DexForce/EmbodiChain/blob/main/examples/learning/differentiable_toppra/README.md)

The READMEs own runnable commands, dependencies, artifacts and evaluation
protocols. These are kinematic examples; collision, contact, dynamics and physical
execution are outside their validation scope.

The unified kinematic benchmark measures differentiable IK:

```bash
python -m scripts.benchmark.robotics.kinematic_solver.run_benchmark -s differentiable
```

`differentiable` measures IK forward and forward/backward passes, FK/Jacobian
and DLS components, and FEP/Pytorch forward-only baselines. Reports
describe workloads, error metrics, timing and memory-accounting scope; benchmark
results do not establish performance for a different task or device.
