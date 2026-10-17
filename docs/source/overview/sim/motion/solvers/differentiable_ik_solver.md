# Differentiable IK

Use the Torch layer for training and the Solver adapter for a Robot control part.
Both call the same fixed-step DLS implementation in `embodichain.compute.kinematics`.
The existing `DifferentialSolver` retains its control-oriented interface and behavior.

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

## Robot adapter

```python
from embodichain.lab.sim.motion.solvers import DifferentiableIKSolverCfg

cfg = DifferentiableIKSolverCfg(
    urdf_path=get_data_path("Franka/Panda/PandaWithHand.urdf"),
    root_link_name="base", end_link_name="fr3_hand_tcp",
    ik=DifferentiableIKCfg(max_iterations=16), dtype="float32",
)
solver = cfg.init_solver(device="cuda")
success, joints = solver.get_ik(target, seed)  # (N,), (N, dof)
valid, candidates = solver.get_ik(target, seed, return_all_solutions=True)
# (N, 1), (N, 1, dof): one candidate, no random multi-start search.
```

The adapter supports live `set_tcp`, `set_qpos_limits`, and Robot limit intersections.
`get_fk` includes TCP; `get_jacobian` defaults to the same TCP. Optional `locations`
specifies a shared or per-sample flange-frame translation. Use layer/solver direct
calls for float64 validation; Robot's arena conversion currently uses float32.
Continuous candidate-path selection is not supported.

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

## Training and measurement

```bash
python -m examples.learning.differentiable_ik.train --device cuda
python -m examples.learning.differentiable_ik.train --device cpu --batch-size 8 --horizon 3 --ik-iterations 4 --updates 20
python -m scripts.benchmark.robotics.kinematic_solver.run_benchmark -s differentiable
```

The example first fits a small MLP to synthetic joint-tracking transitions, then
freezes its parameters and trains an actor using the existing complete-rollout APG
trainer. It is not a physical Franka dynamics model. Frozen model forward calls
remain in grad mode so policy gradients propagate through its inputs. State becomes
the next IK seed without detaching inside the rollout. Outputs include held-out
metrics, trainer checkpoint, state-model weights and IK settings.

The benchmark measures forward, forward+backward, FK/Jacobian and DLS components on
identical local and distant Franka targets. FEP and PytorchSolver are forward-only
baselines. It reports errors including failed rows, throughput, CPU memory and
Torch GPU allocations; Warp allocations are not included in that GPU counter.
