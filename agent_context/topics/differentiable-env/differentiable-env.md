# differentiable-env

> Topic: Newton-backed kinematic environments for analytic policy gradient
> (APG) through the Warp-tape ↔ PyTorch-autograd bridge.

## Public entry point

Use
`embodichain.lab.gym.envs.differentiable_env.DifferentiableEnv`.
It inherits `EmbodiedEnv`, preserves its scene lifecycle, and replaces the
normal physics step with a task-defined kinematics callback recorded on a
Warp tape.

The resolution path is:

    DifferentiableEnv.step(action)
      → _functional_state_tensors()
      → NewtonStepFunc.apply(action, sim_state, *state_tensors)
      → _apply_action_kernel(action_wp, *state_wps, tape=tape)
      → _make_kinematic_step_fn()()
      → _read_outputs(final_state)
      → Warp tape backward → action.grad and optional functional-state grads

## Invariants

- The configured physics backend must be `NewtonPhysicsCfg`.
- `NewtonPhysicsCfg.requires_grad` must be `True`.
- Default physics and other backends fail during `DifferentiableEnv`
  construction.
- `DifferentiableEnv` always supplies a named kinematics callback; its bridge
  contract has no dynamics mode, solver substeps, or control buffer.
- The environment never invokes the configured Newton solver or collision
  pipeline.
- Newton gradient configuration still selects the semi-implicit solver, but
  `DifferentiableEnv` never advances it.
- Gradient mode disables Newton CUDA graph capture.
- `NewtonStepFunc.apply(action, sim_state, *state_tensors)` accepts optional
  functional state tensors. They are appended to the action-kernel arguments
  as Warp arrays and receive gradients in backward, which lets downstream
  tasks retain a complete recurrent closed-loop graph without storing tensors
  in the metadata-only `sim_state` mapping.
- `DifferentiableEnv.step(action)` obtains those tensors from
  `_functional_state_tensors()` and forwards them through the same bridge.

## Subclass contract

Task authors implement three required hooks and may override one state hook:

- `_functional_state_tensors()` optionally returns recurrent PyTorch state in
  the order expected by the action hook. The default returns an empty tuple.
  `_read_outputs` must expose a next-state tensor through `_order` and
  `_grad_track` before retaining that same tensor for the following step.
- `_apply_action_kernel(action_wp, *state_wps, tape=tape)` launches Warp work
  that maps the PyTorch action and functional-state bridge arrays into
  task-owned kinematic state. `tape` is keyword-only.
- `_make_kinematic_step_fn()` returns a zero-argument callback such as
  `newton.eval_fk(...)`. The callback returns the state consumed by the output
  hook.
- `_read_outputs(final_state)` returns `obs`, `reward`, `terminated`, and
  `truncated`, plus `_order` and `_grad_track` metadata used by
  `NewtonStepFunc`.

There is no public `_apply_dynamics_action_kernel` or
`differentiable_step_mode` extension point. Differentiable dynamics are a
future feature, not a `DifferentiableEnv` capability.

## Autograd and reset rules

Action, kinematics, and gradient-producing output kernels must execute while
the Warp tape is open. Each tracked output names its backing Warp array in
`_grad_track`; an output mapped to `None` does not seed Warp backward.

A terminal step tracked through either the action or functional state returns
the terminal observation and exposes
`requires_reset_after_backward` plus `deferred_reset_ids` in `info`. Reset
those rows only after backward. A no-grad terminal step resets them
synchronously.

## Franka reference task

`embodichain_tasks.special.franka_reach_apg.FrankaReachApgEnv` is the canonical
example. Its path is:

    action → new_joint_q → newton.eval_fk → body_q → reward kernel → action.grad

The task snapshots live joint positions before opening the tape and writes the
detached next joint state back after the bridge returns. It does not exercise
Newton dynamics.

## Dynamics boundary

The public differentiable package exposes no solver stepper, trajectory, or
gradient-rollout API. Add those capabilities as a separate future design when
Newton dynamics are ready for end-to-end validation.

## Source of truth

- `embodichain/lab/gym/envs/differentiable_env.py`
- `embodichain/lab/sim/cfg/simulation.py`
- `embodichain/lab/sim/diff/`
- `embodichain_tasks/embodichain_tasks/special/franka_reach_apg.py`

## Focused validation

- `tests/gym/envs/test_differentiable_embodied_env.py`
- `tests/sim/test_sim_manager_cfg.py`

## Related topics

- env-framework
- rl-learning
