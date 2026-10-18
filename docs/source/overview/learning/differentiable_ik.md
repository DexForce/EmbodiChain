# Differentiable IK

Use `DifferentiableIKLayer` for training, task-parameter optimization and
forward-only inference. It calls the fixed-step DLS implementation in
`embodichain.compute.kinematics` without importing the simulation runtime.

## Training without simulation

```python
from pathlib import Path
import torch
from embodichain.compute.kinematics import SerialChainKinematics, DifferentiableIKCfg
from embodichain.data import get_data_path
from embodichain.learning.layers import KinematicsLayer, DifferentiableIKLayer

model = SerialChainKinematics.from_urdf(
    Path(get_data_path("Franka/Panda/PandaWithHand.urdf")).read_text(),
    end_link_name="fr3_hand_tcp", root_link_name="base", device="cuda",
)
kinematics = KinematicsLayer(model)
ik = DifferentiableIKLayer(kinematics, DifferentiableIKCfg(max_iterations=16))
seed = kinematics.default_seed()[None].clone().requires_grad_()
target = kinematics(seed.detach() + 0.01).detach().requires_grad_()
result = ik.solve(target, seed)
loss = result.position_error.square().mean() + result.rotation_error.square().mean()
loss.backward()  # Both seed.grad and target.grad are connected.
# q_des = ik(target, seed) returns just the joint tensor.
```

Targets are valid finite SE(3) transforms in the chain root frame and refer to the
configured TCP. Their shape is `(..., 4, 4)`. Seeds are `(dof,)` or `(..., dof)`;
results retain all leading dimensions. Positions use metres, revolute joints and
rotation vectors use radians, and prismatic coordinates use metres.

The layer requires matching dtype/device and supports float32 and float64. A model
copy owns the geometry; `layer.to(...)` migrates its PK tensors as well as registered
buffers. Checkpoints contain numerical settings, TCP, limits and a geometry
fingerprint. Reconstruct the same chain before loading. Geometry and TCP are fixed
configuration, not learned parameters. `eval()` does not disable gradients.

## Forward-only inference and frame ownership

```python
with torch.no_grad():
    result = ik.solve(target, seed)
    success, joints = result.success, result.qpos
```

The layer takes one seed per target and does not run a multi-start search. It is
an `nn.Module`, not a registered Robot solver. Callers own world-to-root frame
conversion and must configure TCP and physical joint limits explicitly when
constructing `KinematicsLayer`. Existing simulation solvers and robot defaults
remain unchanged.

## Numerical and gradient contract

Each iteration computes FK and the TCP Jacobian once, then applies
`q_next = q + step_size * J.T @ solve(J @ J.T + damping**2 * I, error)`.
FK uses standard autograd, and the derivative of the Jacobian with respect to joints
is retained. The implementation uses `solve_ex(check_errors=False)` without
per-iteration convergence synchronization. No previous seed is stored across calls.

Defaults are 16 iterations, damping 0.05, step size 0.5, position tolerance 0.0005 m
and rotation tolerance 0.0005 rad. Tolerances classify the final iterate, never stop
iterations early. Hard limit projection applies to the seed and every update;
outside a clamp interval its derivative is zero. Set `enforce_joint_limits=False`
for unconstrained experiments, and supply your own limit loss; success still checks
physical limits. Unbounded joints default to projected zero; bounded joints use
their midpoint.

Rotation errors use shortest root-frame axis-angle vectors with a finite derivative
at zero. The pi branch cut and hard limit transitions are not globally smooth.
Only first-order input gradients are supported as a public contract. DLS is a
local solver, not a guarantee of global reachability. Failed solves return the last
iterate and differentiable residuals rather than substituting the seed.

## Measurement

```bash
python -m scripts.benchmark.robotics.kinematic_solver.run_benchmark -s differentiable
```

The benchmark measures forward, forward+backward, FK/Jacobian and DLS components on
identical local and distant Franka targets. FEP and PytorchSolver are forward-only
baselines. It reports errors including failed rows, throughput, CPU memory and
Torch GPU allocations; Warp allocations are not included in that GPU counter.
