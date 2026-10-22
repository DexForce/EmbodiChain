# Simple Newton solver

This example installs a real `dexsim.newton_solvers` provider named `simple`
from EmbodiChain. It demonstrates an ordinary `SimulationManager.update()`,
independent differentiable rollout, parameter/input gradients, publication into
the live Scene, and partial functional/native reset in two environments.

## Prerequisite

Use Python 3.10 or 3.11 with Torch and a DexSim build containing the
`FunctionalDynamics` service revision of
[DexSim MR !1483](http://192.168.3.16/Engine/dexverse/-/merge_requests/1483).
The released DexSim pin in EmbodiChain does not yet provide that API. Normal
EmbodiChain installations remain supported; this optional example requires the
integration SDK. The older callback revision of !1483 is also insufficient.

In a dedicated EmbodiChain development environment, install this checkout and
then the matching SDK wheel. Use `--no-deps` only for the EmbodiChain checkout,
so it does not reinstall the released SDK pin. Let the SDK wheel resolve its
own dependencies; the qualified build requires Newton 1.7.0 and Warp 1.17.0.
This integration environment intentionally overrides EmbodiChain's released
SDK pin:

```bash
python -m pip install --no-deps --no-build-isolation -e .
python -m pip install /path/to/compatible/dexsim_engine.whl
python examples/simple_newton_solver/demo.py --device cpu
# Optional, on a supported CUDA installation:
python examples/simple_newton_solver/demo.py --device cuda:0
```

Reinstall editable metadata after changing entry points. The command runs
headlessly and prints a JSON result with positions and nonzero gradients. It
checks numerical equality and state isolation before reporting success.

For feature acceptance, first require the SDK capability, then run the provider
tests. Ordinary CI using the released SDK skips this optional test module; that
skip is not validation of the solver:

```bash
python -c 'from dexsim.engine.newton_physics import FunctionalDynamics'
python -m pytest tests/sim/physics/test_simple_solver.py -q --run-gpu
```

## Dynamics and supported model

For each independent prismatic leaf, with coordinate `q`, velocity `v`, input
force `u`, trainable scale `g`, and effective mass `m + armature`:

```text
f = clamp(g * u, -effort_limit, effort_limit)
a = (f - stiffness * q - (damping + joint_damping) * v) / effective_mass
v_next = clamp(v + dt * a, -velocity_limit, velocity_limit)
q_next = clamp(q + dt * v_next, lower_limit, upper_limit)
```

A position-limit clamp stops the slider. Body acceleration is the realized
velocity change divided by `dt`. Gradients through the clamps are piecewise;
the example stays away from these boundaries. Native and functional stepping
call the same transition implementation. Body poses use the fixed joint frames
and world-space slider axis, preserving arena placement and body orientation.

The model must have fixed ancestors and independent prismatic leaf bodies,
zero gravity, no particles, mimic constraints, friction or position/velocity
drives. Moving-body collisions are rejected. Static ground is allowed because
the moving bodies have collision disabled. The demo uses an explicit collision
overlay: URDF visual geometry can otherwise become collision geometry during
import. Unsupported explicit control targets and external body forces remain
errors. This eager solver does not support CUDA Graph capture.

## API boundaries

`NewtonPhysicsCfg(solver_cfg={"solver_type": "simple", "stiffness": 2.0})`
uses installed metadata discovery; no global registration call is needed.
`requires_grad=True` enables the functional path. After `sim.prepare()`, use
the public `sim.spawn_result` Scene:

- `create_functional_state()` clones physical state.
- `functional_parameters()` returns the model-scoped trainable `force_scale`.
- `functional_step(..., publish=False)` computes an isolated trajectory.
- `functional_step(..., publish=True)` also publishes detached physical state
  and advances the live clock once. Do not also call `sim.update()` for that
  transition. The returned tensors retain their gradient graph.
- `reset_functional_state(..., arenas=["arena_0"])` resets only the selected
  environment; the global `default` arena is a separate selection.

The service is stateless: recurrent history is empty, history commit is a no-op,
and DexSim owns native buffers, clocks and lifecycle. A topology rebuild creates
a new provider; save and restore learned parameters explicitly across rebuilds.
Training loops and optimizers remain downstream. The example adds no manager or
registry of its own.

The demonstrated gradients are with respect to effort and `force_scale`.
Functional state includes both joint and body coordinates. To differentiate a
Cartesian loss with respect to initial joint positions, initialize body poses
through a differentiable FK graph consistent with those positions. Promoting
only `joint_q` to a trainable tensor while keeping the initial `body_q` detached
instead differentiates a redundant state with its initial body pose held fixed.
