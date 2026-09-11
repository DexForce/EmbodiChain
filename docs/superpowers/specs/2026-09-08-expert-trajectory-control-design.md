# Expert Trajectory Timing and Joint Command Design

## Goal

Provide one source-independent expert-trajectory execution contract that uses
planner timing correctly in standalone simulation and Gym environments. Gym
data generation must optionally command and record joint position plus joint
velocity while preserving the existing position-only behavior and dataset
schema by default.

The design covers expert trajectories produced by `MotionGenerator`, Task
Programs, handwritten joint sequences, replay files, and compatible external
planners. It does not change the timing semantics of ordinary policy-driven
`env.step()` calls.

## Invariants

1. A planner's `dt` describes the planner time axis. It never changes the
   simulator's configured `physics_dt`.
2. A Gym transition always advances exactly one environment control period:
   `step_dt = physics_dt * sim_steps_per_control`.
3. A standalone simulation playback uses a fixed command period that must be
   an integer multiple of `physics_dt`.
4. Joint velocity targets are defined on the time grid that is actually
   executed. Resampling or retiming positions invalidates source velocities and
   requires velocity recomputation.
5. Expert control and expert recording use the same action layout. A dataset
   must contain every target that affected the simulated robot.
6. Position-only remains the default, including its existing action width and
   LeRobot schema.

## Configuration Ownership

Add a source-independent environment configuration under
`EmbodiedEnvCfg`:

```python
@configclass
class ExpertTrajectoryCfg:
    joint_command_mode: Literal["position", "position_velocity"] = "position"


class EmbodiedEnvCfg(EnvCfg):
    expert_trajectory: ExpertTrajectoryCfg = ExpertTrajectoryCfg()
```

The public YAML form is:

```yaml
expert_trajectory:
  joint_command_mode: position_velocity
```

`ExpertTrajectoryCfg` belongs to
`embodichain/lab/gym/envs/expert_trajectory.py`. It is deliberately not owned
by `MotionGenOptions`, a Task Program preset, or a dataset functor:

- Motion generation is only one possible trajectory source.
- Task Programs are only one possible expert producer.
- Dataset backends persist the effective command and must not select physics
  control behavior.
- The mode must be known during environment construction, before expert
  buffers and LeRobot features are allocated.

`DemoExecutionCfg` remains per-run persistence configuration. A single dataset
run cannot override or mix expert action schemas.

## Source-Neutral Expert Trajectory

Introduce a validated value type in `expert_trajectory.py`:

```python
@dataclass(frozen=True, slots=True)
class ExpertJointTrajectory:
    positions: torch.Tensor
    velocities: torch.Tensor | None = None
    dt: torch.Tensor | None = None
```

Its tensors are environment-batched. `positions` and optional `velocities` use
shape `(B, N, D)`; explicit `dt` uses `(B, N)`. An omitted `dt` means the input
already contains one sample per destination control step. This supports a
handwritten position sequence without pretending that it was planned by
`MotionGenerator`.

Adapters convert `PlanResult`, replay input, and finite handwritten sequences
to this type before execution. Existing lazy `DemoSegment.actions` remain
supported. In `position_velocity` mode, a lazy per-frame action must provide
both targets because a single frame lacks the neighboring samples required to
derive velocity safely.

## Fixed-Grid Retiming

Add a pure computation to `embodichain.compute.trajectory` that prepares a
timed path for a fixed command period. It accepts `(positions, dt,
control_dt)` and returns retimed positions, arrival intervals, velocities, and
per-row valid sample counts.

For each environment row with source duration `T`:

```text
K = ceil(T / control_dt)
execution_duration = K * control_dt
```

For `K > 0`, source time is sampled at uniformly spaced phase points
`j * T / K`, for `j = 0..K`. Returned arrival intervals are exactly zero for
the first sample and `control_dt` for subsequent valid samples. This uniformly
stretches the trajectory by less than one control step instead of accelerating
it, so time quantization does not create a new velocity-limit violation.

Rows are padded to the largest `K` in a batch by repeating the terminal
position and using zero velocity. Explicit valid counts distinguish motion
samples from rectangular batch padding. A zero-duration stationary input
produces one hold sample with zero velocity.

When no retiming is needed and a valid native velocity trajectory is present,
the native values may be retained. Otherwise velocities are recomputed from
the executed positions and fixed grid. The recomputed first and last velocity
targets are zero, stationary intervals are zero, and all output values are
finite. The integration layer validates the result against robot joint limits;
it reports violations instead of silently clamping them.

The existing `resample_in_time()` remains a duration-preserving sampling
primitive. The new operation has different semantics because it intentionally
quantizes the duration to a destination execution clock.

## Standalone Simulation Playback

Add a reusable playback API under
`embodichain/lab/sim/motion/execution.py`. Its configuration contains a fixed
`control_dt` and `joint_command_mode`. `control_dt=None` resolves to
`physics_dt`; any explicit value must satisfy:

```text
control_dt / physics_dt == positive integer
```

Playback first prepares the expert trajectory for that command grid. Each
command writes qpos and, in `position_velocity` mode, qvel, then advances the
simulation with the unchanged `physics_dt` for the resolved integer number of
physics steps. It never passes planner intervals as a replacement physics
timestep. The terminal command and all held or inactive rows use zero qvel.

The motion-generator tutorial is migrated to this API and selects
`position_velocity` explicitly.

## Gym Expert Execution

The Gym destination control period is always `env.step_dt`. Expert trajectory
normalization happens before actions are yielded to the common demo executor.
`BaseEnv.step()` and `EmbodiedEnv.step()` retain their existing sequence and
always call:

```python
sim.update(env.physics_dt, env.cfg.sim_steps_per_control)
```

The expert mode affects only controller command contents:

- `position`: emit the existing qpos tensor command.
- `position_velocity`: emit a controller-ready `TensorDict` containing `qpos`
  and `qvel` on the environment device.

`JointPositionGymTransportEncoder` receives the environment's expert command
mode. In position mode it retains its current output exactly. In
position-velocity mode it composes full-robot qpos and qvel targets, writes
only addressed joints for active rows, holds inactive qpos, and zeros inactive
qvel. A missing `JointPositionPayload.velocities` is an error in this mode; the
transport layer must not guess timing from an isolated command frame.

The existing `MotionPolicy.velocity_targets` remains a trajectory-reference
policy (`auto` or explicit zero). It is separate from the destination command
mode. For example, `position_velocity` plus `velocity_targets="zero"` is valid
and records explicit zero velocity targets.

At demo start, abort, completion, and reset, the expert execution boundary
zeros velocity targets for affected joints so a previous feed-forward target
cannot leak into a hold or later position-only command.

## Expert Action Recording

The environment's policy `action_space` remains unchanged. Expert collection
uses a separately derived expert action specification:

| Mode | Stored `action` width | Layout |
|---|---:|---|
| `position` | `D` | `[qpos]` |
| `position_velocity` | `2D` | `[qpos, qvel]` |

Here `D` is the number of active joints. The canonical stored representation is
a flat tensor so existing rollout, async-recorder, and LeRobot paths continue
to operate on one standard action feature. In position-velocity mode the
LeRobot names are ordered as all `<joint>.position` values followed by all
`<joint>.velocity` values.

Both the expert rollout buffer and optional `.pt` trajectory buffer are
allocated from the expert action specification rather than blindly from the
policy action space. A shared encoder converts the effective controller action
to the canonical stored vector, so persistence cannot silently discard qvel.

Episode and trajectory metadata include:

- `expert_trajectory_schema_version`
- `joint_command_mode`
- `step_dt`
- ordered active joint names
- explicit qpos and qvel slices within the action vector

Measured `observation.qvel` remains distinct from target `action` qvel.
Position mode preserves the old feature width, names, and values.

## Compatibility and Errors

- Existing environment configs default to `position` and require no changes.
- Existing raw policy actions and `env.action_space` retain their current
  meaning.
- Existing position-only expert datasets retain their action schema.
- Structured qpos/qvel controller actions remain accepted by the low-level
  environment boundary independently of expert collection.
- A position-velocity expert run rejects missing, malformed, wrong-device, or
  non-finite qvel targets.
- An expert source with explicit timing rejects negative intervals, position
  changes at zero intervals, and a nonzero first arrival offset.
- An explicit standalone `control_dt` that is not representable by an integer
  number of physics steps is rejected without rounding.
- One environment/dataset instance has one immutable expert action schema;
  changing the mode requires constructing a new environment and dataset run.

## Implementation Boundaries

Expected primary changes:

- `embodichain/compute/trajectory/timing.py`: fixed-grid retiming primitive.
- `embodichain/lab/sim/motion/execution.py`: standalone fixed-cadence playback.
- `embodichain/lab/gym/envs/expert_trajectory.py`: config, value type, action
  specification, and source-neutral normalization.
- `embodichain/lab/gym/envs/embodied_env.py`: configuration wiring, expert
  buffer allocation, effective-action recording, and metadata.
- `embodichain/lab/gym/envs/task_program/bridge.py`: mode-aware qpos/qvel
  command encoding.
- `embodichain/lab/task_program/integrations/environment.py`: pass the
  environment-owned expert mode into the bridge encoder.
- `embodichain/lab/gym/envs/managers/datasets.py`: mode-aware LeRobot feature
  names and lossless action conversion.
- focused context pages for motion planning, environment execution, and data
  persistence.

No planner backend needs to know `env.step_dt`, and no dataset backend chooses
the robot control mode.

## Validation

Focused tests must prove:

1. Exact-grid retiming is unchanged; off-grid duration is rounded upward and
   uniformly slowed; endpoints, valid counts, padding, and zero-duration rows
   are correct.
2. Recomputed qvel uses destination `control_dt`, with zero terminal and
   stationary targets.
3. Standalone playback never changes `physics_dt`, advances an integer number
   of physics steps per command, and applies qpos/qvel together.
4. `env.step()` advances the same number of physics steps in both expert
   command modes.
5. Default position mode produces the existing qpos action and dataset schema.
6. Position-velocity mode encodes, applies, buffers, and exports both targets
   in the documented order.
7. Task Program commands and non-MotionGenerator expert trajectories use the
   same normalization and recording behavior.
8. Missing qvel in a lazy position-velocity command fails before physics
   advances.
9. Inactive, aborted, completed, and reset rows cannot retain nonzero target
   velocity.

Simulator-backed tracking validation should compare position-only and
position-velocity modes at the same fixed physics and command cadence; it is a
behavioral benchmark, not a universal assertion that velocity feed-forward
always lowers tracking error.

## Out of Scope

- Changing planner backend time-parameterization algorithms.
- Changing ordinary policy action spaces or learning-policy output semantics.
- Velocity-only or effort-control expert modes.
- Variable `physics_dt` playback.
- Mixing expert action schemas inside one dataset run.
- Real-device controller timing and interpolation.
