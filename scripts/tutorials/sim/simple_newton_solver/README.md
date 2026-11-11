# Tutorial: a custom Newton solver

This directory owns the complete spring-slider example: `solver.py`, `demo.py`,
and `slider.urdf`. The provider is tutorial code; it is not part of the
`embodichain` package or its solver registry. `demo.py` passes the local
`SimpleSolverProvider` directly to DexSim. It demonstrates `SimulationManager.update()`,
independent differentiable rollout, parameter/input gradients, publication into
the live Scene, and partial functional/native reset in two environments.

The [narrative tutorial](../../../../docs/source/tutorial/simple_newton_solver.rst)
walks through the configuration, shared transition, and expected output.

## Prerequisite

Use the current DexSim extension SDK with `ProviderNewtonSolverCfg`,
`FunctionalDynamics`, `Scene.dynamics_capabilities()` and explicit control,
particle and native acceleration capabilities. The released dependency pin
may lag these development APIs; selecting the provider reports an actionable
error when the SDK is insufficient. The provider can still be inspected without
activating the extension.

The example was validated with DexSim `0.6.0+c3e92584`, Newton 1.7.0 and Warp
1.17.0 in the conda `open` environment. Run this checkout against that environment
without reinstalling its editable package or SDK:

```bash
conda activate open
export PYTHONPATH="$PWD:$PWD/extensions${PYTHONPATH:+:$PYTHONPATH}"
python scripts/tutorials/sim/simple_newton_solver/demo.py --device cpu
python scripts/tutorials/sim/simple_newton_solver/demo.py --device cuda:0
python scripts/tutorials/sim/simple_newton_solver/demo.py --device cpu \
    --preview-path simple_newton_solver_preview.png
python scripts/tutorials/sim/simple_newton_solver/demo.py --device cpu \
    --viser --viser-seconds 60
```

For a separate development installation, install the checkout with `--no-deps`
and install a compatible DexSim wheel with its dependencies. The direct factory
example requires no entry-point metadata refresh.

The command runs headlessly and prints a JSON result with positions and nonzero
gradients. It checks numerical equality and state isolation before reporting
success. The two slider positions after publication are about `0.02087 m` and
`0.03131 m`; resetting the first arena leaves the second at `0.03131 m`.
Pass `--preview-path` to save a PNG with the measured functional trajectory and
the selected-arena reset. The plot is generated only when this option is used.

Pass `--viser` to see the actual EmbodiChain scene at `http://127.0.0.1:8080`.
The script exports both slider links, replays 30 native simulation steps, and
keeps the browser scene open for 30 seconds by default. Use `--viser-seconds 60`
to leave more time to inspect the result, or `--viser-port PORT` if 8080 is busy.
The JSON includes the visible link count and final replay positions. Viser
shows the URDF cube mesh; the `base` has no visual geometry. This mode uses the
existing `VisualizationCfg` and `SimulationManager.capture_visualization()`
without requiring a native render window.

For local verification, first require the SDK capability, then run the tutorial
on CPU or CUDA. The default package test suite does not exercise this example:

```bash
python - <<'PY'
from dexsim.engine.newton_physics import FunctionalDynamics
from dexsim.engine.newton_physics import ProviderNewtonSolverCfg
from dexsim.scene import Scene
assert callable(getattr(Scene, "dynamics_capabilities", None))
PY
python scripts/tutorials/sim/simple_newton_solver/demo.py --device cpu
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

Python configuration passes the provider factory directly:

```python
from dexsim.engine.newton_physics import ProviderNewtonSolverCfg
from embodichain.lab.sim.cfg import NewtonPhysicsCfg
from solver import SimpleSolverProvider  # local to this tutorial directory

physics = NewtonPhysicsCfg(
    solver_cfg=ProviderNewtonSolverCfg(
        provider_factory=SimpleSolverProvider,
        options={"stiffness": 2.0},
    ),
)
```

Pass a class or zero-argument factory, not an already-created provider/solver.
DexSim creates a fresh provider for every model build. The direct factory path
does not scan installed entry points. To try name-based registration in a
separate project, package and register your own provider under
`dexsim.newton_solvers`; this tutorial deliberately does not register one.
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
