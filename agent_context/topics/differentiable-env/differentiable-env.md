# differentiable-env

> Topic: Differentiable environment for analytic policy gradient (APG) —
> `DifferentiableEmbodiedEnv` + the `embodichain.lab.sim.diff` Warp-tape
> ↔ PyTorch-autograd bridge.

## Overview

EmbodiChain supports analytic policy gradient (APG) via
`embodichain.lab.gym.envs.differentiable_env.DifferentiableEmbodiedEnv`.
The bridge wraps a Warp tape around one EmbodiChain physics step and
exposes a `torch.autograd.Function`
(`embodichain.lab.sim.diff.NewtonStepFunc`) so PyTorch-side `action`
tensors get a gradient from `tape.backward()`.

## Required configuration

- `NewtonPhysicsCfg(requires_grad=True, solver_cfg={"solver_type": "semi_implicit"})`
- `use_cuda_graph=False` (forced by dexsim when grad mode is on)

The default backend and any other Newton solver are rejected at
construction time by `DifferentiableEmbodiedEnv._validate_diff_cfg`.

## Subclass contract

Task authors implement two methods on `DifferentiableEmbodiedEnv`:

- `_apply_action_kernel(action_wp, tape)` — launch a Warp kernel that
  writes joint/body targets into `nm._control` while the tape is open.
  The `action_wp` argument is a `wp.array(dtype=wp.float32,
  requires_grad=True)` of shape `[num_envs * action_dim]`.
- `_read_outputs(final_state)` — build the `obs` / `reward` /
  `terminated` / `truncated` outputs as torch tensors via `wp.to_torch`
  so the tape can record the dependency. Must return a dict with
  `_order` (tuple of output keys) and `_grad_track` (mapping from output
  key to the Warp array that backs its gradient, or `None` for outputs
  that don't need grad).

Optionally override `_make_step_fn()` to swap the per-substep advance
function. The default uses `dexsim.engine.newton_physics.DifferentiableStepper.step`;
the Franka APG example overrides it to call `newton.eval_fk` directly
(see "FK bypass" below).

See `embodichain/lab/gym/envs/tasks/special/franka_reach_apg.py` for
the canonical example.

## Why reward must be computed inside the tape

`NewtonStepFunc.forward` keeps the `wp.Tape` open while
`obs_reward_fn(final_state)` runs. Reward must be computed by a Warp
kernel that writes into a `wp.zeros(..., requires_grad=True)` array
inside the tape; `wp.to_torch(reward_wp)` then returns a torch tensor
that carries the tape's gradient. Computing reward in pure torch *after*
the tape closes would detach it from the autograd graph and
`action.grad` would come back as `None`.

The same rule applies to any observation that needs to be
grad-tracked: build it from `wp.to_torch` of a tape-tracked Warp array.

## FK bypass for the Franka task

The `semi_implicit` Newton solver does not propagate gradient through
`joint_target_pos` to `body_q` (verified empirically; the reference
implementation at `/root/sources/analytic_policy_gradients/envs/franka_reach_env.py`
hits the same limitation and uses the same workaround). The Franka APG
example overrides `_make_step_fn()` to call `newton.eval_fk(model,
new_joint_q, joint_qd, fk_state)` directly, bypassing the dynamics
solver. The grad path is then:

    action → new_joint_q (action kernel) → eval_fk → body_q → reward kernel → reward_wp → tape.backward → action.grad

The default `_make_step_fn` still uses the differentiable stepper, so
envs whose reward depends on dynamics (not just FK) can use it — but
they should verify the solver actually propagates grad for their
control inputs before relying on it.

## Functor autograd compatibility

Reward/observation functors that compose torch operations on tensors
obtained via `wp.to_torch` are automatically autograd-compatible.
Functors that detour through CPU / NumPy break the graph; those need
torch-only reimplementations for the differentiable path. For now, the
differentiable env computes reward via a dedicated Warp kernel rather
than reusing the standard reward-manager functors — a future task can
audit and port functors as needed.

## Memory

Each step records `sim_steps_per_control` substeps into the tape. For
long horizons or large `num_envs`, pass `truncate_backward_at=K` on the
env config to split the tape and detach at chunk boundaries.

## Source of truth

- `embodichain/lab/gym/envs/differentiable_env.py` —
  `DifferentiableEmbodiedEnv` base class.
- `embodichain/lab/sim/diff/bridge.py` — `NewtonStepFunc`,
  `tape_context`, `differentiable_step`.
- `embodichain/lab/gym/envs/tasks/special/franka_reach_apg.py` —
  example task.
- `embodichain/lab/sim/sim_manager.py` —
  `SimulationManager.create_differentiable_stepper` /
  `create_gradient_rollout` delegators.
- `/root/sources/dexsim/python/dexsim/engine/newton_physics/differentiable_stepper.py`
  — the underlying dexsim primitive.

## Related topics

- env-framework
- rl-training
