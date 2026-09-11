# Expert Trajectory Timing and Joint Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Execute planner and source-neutral expert joint trajectories on a fixed simulator control grid, optionally command and persist qpos plus qvel in Gym expert-data generation, and preserve position-only behavior by default.

**Architecture:** Keep planner `dt`, simulator `physics_dt`, and environment `step_dt` as separate clocks. A pure compute primitive retimes timed paths to a fixed command grid; simulator and Task Program integrations consume that result without changing physics stepping. `EmbodiedEnvCfg.expert_trajectory` owns the destination command/recording schema, while dataset code only persists the already selected schema.

**Tech Stack:** Python 3.10+, PyTorch, TensorDict, Gymnasium, DexSim, pytest, Black 26.3.1.

**Spec:** `docs/superpowers/specs/2026-09-08-expert-trajectory-control-design.md`

## Global Constraints

- Work only in `/root/sources/EmbodiChain-expert-trajectory-qvel` on `codex/expert-trajectory-qvel`.
- Preserve `EmbodiedEnv.step()` physics advancement: one step always calls the simulator with configured `physics_dt` and `sim_steps_per_control`.
- Preserve the default `position` command path and its existing dataset action width and names.
- Do not change policy `action_space`; derive a separate expert action specification for expert buffers and persistence.
- Reject malformed or missing velocity targets in `position_velocity` mode before calling `env.step()`.
- Use focused tests before implementation changes and commit each coherent layer separately.
- At completion, push `HEAD` to `origin/codex/timed-trajectory-velocity-targets`, the existing head branch of PR #595.

## Task 1: Add fixed-control-grid retiming

**Files:**

- Modify: `embodichain/compute/trajectory/timing.py`
- Modify: `embodichain/compute/trajectory/__init__.py`
- Modify: `tests/compute/test_trajectory_timing.py`

**Step 1: Write failing tests**

Add tests for a public `retime_to_control_grid(positions, dt, control_dt)` operation that returns positions, velocities, arrival intervals, and per-row valid sample counts. Cover:

- an exact-grid path whose destination intervals remain exactly `control_dt`;
- an off-grid path with `K = ceil(T / control_dt)`, endpoint preservation, and uniform phase sampling over the original duration;
- recomputed velocities using the destination command period, with zero first/last velocity;
- mixed-duration batches padded by repeated terminal positions, zero intervals, and zero velocities;
- a zero-duration path yielding one valid hold sample;
- rejection of non-positive/non-finite `control_dt`, nonzero first arrival offsets, and invalid timing.

Run:

```bash
pytest -q tests/compute/test_trajectory_timing.py
```

Expected: new retiming tests fail because the function is absent.

**Step 2: Implement the pure primitive**

In `timing.py`, add strict validation and an implementation with this contract:

```python
def retime_to_control_grid(
    positions: torch.Tensor,
    dt: torch.Tensor,
    control_dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Retiming result: positions, velocities, dt, valid sample counts."""
```

Compute each row's duration from `dt[:, 1:]`, calculate the number of command intervals with tolerance-aware ceiling, query the source at `j * T / K`, and write output intervals as `[0, control_dt, ...]`. Reuse `differentiate_positions()` on the rectangular result, then force the first, last valid, and padded velocities to zero. Export the function from `embodichain.compute.trajectory`.

**Step 3: Run and format**

```bash
pytest -q tests/compute/test_trajectory_timing.py
black embodichain/compute/trajectory/timing.py embodichain/compute/trajectory/__init__.py tests/compute/test_trajectory_timing.py
```

Expected: focused timing tests pass.

**Step 4: Commit**

```bash
git add embodichain/compute/trajectory/timing.py embodichain/compute/trajectory/__init__.py tests/compute/test_trajectory_timing.py
git commit -m "feat: retime trajectories to fixed control grids"
```

## Task 2: Retime Atomic Action planner output to the environment clock

**Files:**

- Modify: `embodichain/lab/sim/atomic_actions/trajectory_ops.py`
- Modify: `embodichain/lab/sim/atomic_actions/primitives/move_end_effector.py`
- Modify: `embodichain/lab/sim/atomic_actions/primitives/move_held_object.py`
- Modify: `embodichain/lab/sim/atomic_actions/primitives/move_joints.py`
- Modify: `embodichain/lab/sim/atomic_actions/primitives/pour.py`
- Modify: `tests/sim/atomic_actions/test_trajectory_ops.py`

**Step 1: Write failing integration tests**

Extend trajectory-operation tests so `to_full_robot_trajectory(..., control_dt=...)`:

- retimes an off-grid `PlanResult` to exact control intervals;
- embeds retimed qpos/qvel into the full robot DoF layout;
- keeps uncontrolled joint positions constant and their velocities zero;
- reports correct valid counts through `TimedTrajectory`'s existing padding/timing representation;
- rejects missing planner `dt` before building runtime commands.

Run:

```bash
pytest -q tests/sim/atomic_actions/test_trajectory_ops.py
```

Expected: tests fail because `control_dt` is not accepted.

**Step 2: Wire the retimer**

Add required `control_dt: float` to `to_full_robot_trajectory`. Retime the controlled-joint positions and planner timing before embedding; use the recomputed destination-grid qvel and discard source acceleration because it is no longer valid after retiming. Pass `context.control_dt` from all four primitive call sites. Ensure segment bookkeeping in `move_held_object` and `pour` uses the retimed waypoint count already returned by the helper.

**Step 3: Verify and commit**

```bash
pytest -q tests/sim/atomic_actions/test_trajectory_ops.py tests/sim/atomic_actions/test_sim_adapter.py
black embodichain/lab/sim/atomic_actions tests/sim/atomic_actions/test_trajectory_ops.py
git add embodichain/lab/sim/atomic_actions tests/sim/atomic_actions/test_trajectory_ops.py
git commit -m "feat: align atomic trajectories with control timing"
```

## Task 3: Add source-neutral expert trajectory configuration and encoding

**Files:**

- Create: `embodichain/lab/gym/envs/expert_trajectory.py`
- Modify: `embodichain/lab/gym/envs/__init__.py`
- Modify: `embodichain/lab/gym/envs/embodied_env.py`
- Create: `tests/gym/envs/test_expert_trajectory.py`
- Modify: `tests/gym/envs/test_env_timing.py`

**Step 1: Write failing configuration and encoding tests**

Test:

- `ExpertTrajectoryCfg` defaults to `joint_command_mode="position"` and rejects unknown modes;
- `ExpertJointTrajectory` validates batched positions, optional velocities, and optional timing;
- normalization retimes explicitly timed trajectories to `step_dt` and derives qvel from the executed grid;
- untimed position-velocity trajectories require explicit velocities;
- expert action specification is width `D` for position and `2D` for position-velocity;
- canonical action encoding returns `[active qpos, active qvel]`, validates shape/device/finiteness, and fails before stepping when qvel is absent;
- environment physics advancement is identical for qpos-only and qpos+qvel controller actions.

Run:

```bash
pytest -q tests/gym/envs/test_expert_trajectory.py tests/gym/envs/test_env_timing.py
```

Expected: new module/config tests fail.

**Step 2: Implement the expert contract**

Create `expert_trajectory.py` with:

```python
JointCommandMode = Literal["position", "position_velocity"]

@configclass
class ExpertTrajectoryCfg:
    joint_command_mode: JointCommandMode = "position"

@dataclass(frozen=True, slots=True)
class ExpertJointTrajectory:
    positions: torch.Tensor
    velocities: torch.Tensor | None = None
    dt: torch.Tensor | None = None
```

Add small source-neutral helpers for trajectory preparation, expert action width/names/layout metadata, and canonical conversion of tensor/TensorDict controller actions. Keep these helpers independent of dataset backends and planner classes.

Add `expert_trajectory: ExpertTrajectoryCfg = ExpertTrajectoryCfg()` to `EmbodiedEnvCfg`. Validate it during environment setup. Use the canonical encoder only at expert-recording boundaries; do not modify low-level action preprocessing or policy `action_space`.

**Step 3: Verify and commit**

```bash
pytest -q tests/gym/envs/test_expert_trajectory.py tests/gym/envs/test_env_timing.py
black embodichain/lab/gym/envs/expert_trajectory.py embodichain/lab/gym/envs/embodied_env.py tests/gym/envs/test_expert_trajectory.py tests/gym/envs/test_env_timing.py
git add embodichain/lab/gym/envs tests/gym/envs/test_expert_trajectory.py tests/gym/envs/test_env_timing.py
git commit -m "feat: define expert trajectory joint command modes"
```

## Task 4: Make Task Program expert commands mode-aware

**Files:**

- Modify: `embodichain/lab/gym/envs/task_program/bridge.py`
- Modify: `embodichain/lab/task_program/integrations/environment.py`
- Modify: the Task Program environment adapter factory that constructs `TaskProgramEnvironmentAdapter`
- Modify: `tests/gym/envs/task_program/test_bridge.py`
- Modify: Task Program integration tests covering adapter construction

**Step 1: Write failing bridge tests**

Test that:

- default position mode preserves the existing tensor qpos result byte-for-byte;
- position-velocity mode returns a TensorDict containing full-width `qpos` and `qvel`;
- addressed active rows receive payload positions and velocities;
- inactive rows and holds retain qpos and use zero qvel;
- missing/non-finite/wrong-shaped payload velocity raises before an environment step;
- adapter construction forwards `env.cfg.expert_trajectory.joint_command_mode` to the built-in encoder.

Run:

```bash
pytest -q tests/gym/envs/task_program/test_bridge.py
```

Expected: mode-aware tests fail.

**Step 2: Implement bridge wiring**

Give `JointPositionGymTransportEncoder` an immutable command mode. In position-velocity mode, construct controller-ready TensorDict values and explicitly zero qvel for hold/inactive rows. Update `RuntimeCommandFrameEncoder` to install this built-in transport with the requested mode while preserving custom transport ordering. Pass the environment-owned mode through the adapter factory/integration assembly.

**Step 3: Verify and commit**

```bash
pytest -q tests/gym/envs/task_program/test_bridge.py tests/lab/task_program/integrations
black embodichain/lab/gym/envs/task_program/bridge.py embodichain/lab/task_program/integrations/environment.py tests/gym/envs/task_program/test_bridge.py
git add embodichain/lab/gym/envs/task_program/bridge.py embodichain/lab/task_program/integrations tests/gym/envs/task_program
git commit -m "feat: encode expert qpos and qvel commands"
```

## Task 5: Persist the effective expert action without changing policy spaces

**Files:**

- Modify: `embodichain/lab/gym/envs/embodied_env.py`
- Modify: `embodichain/lab/gym/envs/managers/datasets.py`
- Modify: `embodichain/lab/gym/utils/gym_utils.py` only if buffer allocation requires an explicit action shape helper
- Modify: `tests/gym/envs/managers/test_dataset_functors.py`
- Modify: existing rollout/trajectory persistence tests for `EmbodiedEnv`

**Step 1: Write failing persistence tests**

Cover both modes:

- position mode keeps current action dimension, values, and joint names;
- position-velocity mode allocates expert action width `2D` independently of policy `action_space`;
- rollout and `.pt` trajectory buffers store the canonical concatenated target vector;
- LeRobot action features are ordered as every `<joint>.position` then every `<joint>.velocity`;
- metadata records schema version, command mode, `step_dt`, ordered joint names, and qpos/qvel slices;
- direct structured action conversion never silently discards qvel;
- measured observation qvel is not substituted for target action qvel.

Run the narrowest existing persistence test modules discovered next to these implementations. Expected: qvel-specific assertions fail.

**Step 2: Allocate and record from the expert specification**

During `EmbodiedEnv` initialization, derive one immutable expert action specification from active joints and the configured mode. When demo/expert buffers are created, use that specification rather than policy action width. At every expert write path, canonicalize the effective controller action before appending it. Add layout metadata to episode and `.pt` trajectory metadata. Keep non-expert policy recording on its existing action-space contract.

Update LeRobot feature construction and frame conversion to use the same expert specification. Position mode must preserve the old schema. Position-velocity conversion must either preserve both qpos/qvel or reject an incomplete structured action.

**Step 3: Verify and commit**

```bash
pytest -q tests/gym/envs/managers/test_dataset_functors.py
black embodichain/lab/gym/envs/embodied_env.py embodichain/lab/gym/envs/managers/datasets.py embodichain/lab/gym/utils/gym_utils.py tests/gym/envs/managers/test_dataset_functors.py
git add embodichain/lab/gym/envs/embodied_env.py embodichain/lab/gym/envs/managers/datasets.py embodichain/lab/gym/utils/gym_utils.py tests/gym/envs
git commit -m "feat: record expert position and velocity targets"
```

## Task 6: Add fixed-cadence standalone simulation playback

**Files:**

- Create: `embodichain/lab/sim/motion/execution.py`
- Modify: `embodichain/lab/sim/motion/__init__.py`
- Modify: `scripts/tutorials/sim/motion_generator.py`
- Create: `tests/sim/motion/test_execution.py`

**Step 1: Write failing playback tests**

Use fakes for robot and simulator to prove:

- omitted `control_dt` resolves to `physics_dt`;
- explicit `control_dt` must be a positive integer multiple of `physics_dt`;
- planner timing is retimed before command application;
- every simulator update receives unchanged `physics_dt` and the resolved integer physics-step count;
- position mode writes qpos only and position-velocity mode writes both targets;
- terminal and padded commands use zero qvel.

Run:

```bash
pytest -q tests/sim/motion/test_execution.py
```

Expected: tests fail because playback API is absent.

**Step 2: Implement and migrate the tutorial**

Create a reusable fixed-cadence playback configuration/API that accepts the source-neutral trajectory data or raw timed tensors without importing Gym into simulation. Validate clock divisibility, use `retime_to_control_grid`, command the robot, and call `sim.update(physics_dt, steps_per_command)`. Replace the tutorial's variable-physics-dt loop with this API and select `position_velocity` explicitly for the qvel example.

**Step 3: Verify and commit**

```bash
pytest -q tests/sim/motion/test_execution.py
black embodichain/lab/sim/motion/execution.py embodichain/lab/sim/motion/__init__.py scripts/tutorials/sim/motion_generator.py tests/sim/motion/test_execution.py
git add embodichain/lab/sim/motion scripts/tutorials/sim/motion_generator.py tests/sim/motion/test_execution.py
git commit -m "feat: play timed trajectories at fixed simulation cadence"
```

## Task 7: Update project context and public API coverage

**Files:**

- Modify matched pages under `agent_context/topics/motion-planning/`
- Modify matched pages under `agent_context/topics/env-framework/`
- Modify matched pages under `agent_context/topics/data-pipeline/` if the dataset schema route is documented there
- Modify API reference pages only when the API coverage checker identifies the new public exports

**Step 1: Inspect affected context**

```bash
python .agents/skills/project-dev-context/scripts/context.py affected --base 1fbf914df483f59779360fd7d941704fa1a11507
```

Update only the routed context pages that own the changed behavior. Document clock separation, expert config ownership, qpos/qvel canonical action layout, and fixed-cadence playback.

**Step 2: Check context and API coverage**

```bash
python .agents/skills/project-dev-context/scripts/context.py check
python docs/scripts/check_api_docs.py
```

If the API checker reports missing pages for newly exported public APIs, use the project's API-doc synchronization workflow to add only those pages.

**Step 3: Commit**

```bash
git add agent_context docs/source
git commit -m "docs: describe expert trajectory timing controls"
```

Omit `docs/source` from `git add` when the API checker requires no changes.

## Task 8: Run proportional validation and update PR #595

**Step 1: Format the entire tree**

```bash
black .
git status --short
```

Review every formatting change and retain only task-relevant changes.

**Step 2: Run focused and broader validation**

```bash
pytest -q tests/compute/test_trajectory_timing.py tests/sim/motion/test_execution.py tests/sim/atomic_actions/test_trajectory_ops.py tests/sim/atomic_actions/test_sim_adapter.py tests/gym/envs/test_expert_trajectory.py tests/gym/envs/test_env_timing.py tests/gym/envs/task_program/test_bridge.py tests/gym/envs/managers/test_dataset_functors.py
python .agents/skills/project-dev-context/scripts/context.py affected --base 1fbf914df483f59779360fd7d941704fa1a11507
python docs/scripts/check_api_docs.py
```

Run the repository's proportional pre-commit checks for every changed Python file. Record any dependency/environment failure separately from code failures, including the exact failing import or binary mismatch.

**Step 3: Inspect the complete PR delta**

```bash
git diff --check 1fbf914df483f59779360fd7d941704fa1a11507..HEAD
git diff --stat 1fbf914df483f59779360fd7d941704fa1a11507..HEAD
git log --oneline 1fbf914df483f59779360fd7d941704fa1a11507..HEAD
```

Confirm no unrelated user changes are included and all configuration defaults remain backward compatible.

**Step 4: Push to the existing PR branch**

```bash
git push origin HEAD:codex/timed-trajectory-velocity-targets
gh pr view 595 --repo DexForce/EmbodiChain --json headRefOid,url,state
```

Expected: PR #595 stays open and its `headRefOid` matches the pushed worktree head.
