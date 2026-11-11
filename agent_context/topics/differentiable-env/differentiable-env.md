# differentiable-env

> Topic: One Newton-backed environment entry point for pure Warp kinematics
> and Scene functional solver dynamics.

## Public entry point

Use
`embodichain.lab.gym.envs.differentiable_env.DifferentiableEnv`.
It inherits `EmbodiedEnv` and selects an explicit Python constructor
`step_mode="kinematic"|"dynamics"`. Kinematics is the default; solver names
never select the mode. Both paths share terminal handling while preserving
their different graph/reset lifetimes.

The kinematic resolution path is:

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
- Kinematic mode supplies a named callback through `NewtonStepFunc`; that
  bridge has no solver substeps or control buffer and never invokes Newton
  dynamics. Its callback runs once per control step without advancing physics.
- Dynamics mode uses `FunctionalDynamics` through the public manager/Scene
  path. The task maps actions to controls held over `sim_steps_per_control`;
  DexSim owns solver substeps and native publication/time. There is no second
  World update.
- Both modes require a compatible Newton configuration. Dynamics validates
  provider support through the Scene; kinematic deployments retain their
  semi-implicit setup without stepping it.
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

Dynamics tasks instead implement `_action_to_control(action, state)` and pure
`_read_outputs(state)` over `FunctionalSolverState`. The output hook also runs
for observation, reset and detach. Configured action/observation/reward functors
and native interval events are rejected in dynamics mode. The Scene's physics
and history tensors remain separate from task hooks and live native buffers.
`dynamics_state` and `dynamics_parameters()` are available only in that mode.
There is no parallel environment class or mode field inside `NewtonStepFunc`.

## Autograd and reset rules

Action, kinematics, and gradient-producing output kernels must execute while
the Warp tape is open. Each tracked output names its backing Warp array in
`_grad_track`; an output mapped to `None` does not seed Warp backward.

In kinematic mode, a terminal step tracked through either the action or recurrent state returns
the terminal observation and exposes
`requires_reset_after_backward` plus `deferred_reset_ids` in `info`. Reset
those rows only after backward. A no-grad terminal step resets them
synchronously. Kinematic tasks own their recursive tensors and must implement
their own `detach_state()` boundary for segmented policy training.

Dynamics mode owns independent physical/history tensors and auto-resets
finished rows while retaining terminal reward/observation graphs. The full
pre-reset batch is in `info["terminal_observation"]` for Actor-Critic bootstrap.
Unselected rows preserve their histories and graphs. `detach_state()` cuts
history without resetting, sampling or changing values. Finish backward before
updating model parameters. Shared storage alone never connects Torch/Warp
autograd; the FK bridge owns that derivative.

## Franka reference task

`embodichain_tasks.special.franka_reach_apg.FrankaReachApgEnv` is the canonical
example. Its path is:

    action → new_joint_q → newton.eval_fk → body_q → reward kernel → action.grad

The task snapshots live joint positions before opening the tape and writes the
detached next joint state back after the bridge returns. It does not exercise
Newton dynamics.

## Dynamics boundary

`sim.diff.FunctionalDynamics` exposes registered solver dynamics through the
public manager/Scene path. `NewtonStepFunc` retains the pure-kinematic Warp
contract. The environment selects between these paths without passing solver
modes into the Warp bridge. NeRD binding and physical identity belong to
[neural dynamics](../neural-dynamics/neural-dynamics.md).

## Source of truth

- `embodichain/lab/gym/envs/differentiable_env.py`
- `embodichain/lab/sim/cfg/physics.py`
- `embodichain/lab/sim/diff/`
- `extensions/embodichain_tasks/embodichain_tasks/special/franka_reach_apg.py`

## Focused validation

- `tests/gym/envs/test_differentiable_embodied_env.py`
- `tests/gym/envs/test_differentiable_dynamics_env.py`
- `tests/gym/envs/tasks/test_neural_dynamics_tasks.py`
- `tests/sim/test_sim_manager_cfg.py`

## Related topics

- env-framework
- rl-learning
