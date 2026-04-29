# ATOMIC_ACTION

## Objective

Use the existing atomic action pipeline to replace the motion generation part of the Robochallenge `stack_bowls` task, while preserving the original task rule:

- Keep the original bowl stacking goal and success criterion.
- Keep the original arm-selection rule based on distance.
- Keep the original place offsets and return-to-home logic.
- Replace the intermediate motion generation with atomic actions only.

Target implementation file:

- [robot_challenge_tasks/robot_challenge_tasks/tasks/stack_bowls_atomic.py](/home/dex/桌面/EmbodiChain/robot_challenge_tasks/robot_challenge_tasks/tasks/stack_bowls_atomic.py)

## Confirmed Decisions

- A new environment will be added instead of overwriting the existing `StackBowls-v1`.
- The original offset logic and return logic from [stack_bowls.py](/home/dex/桌面/EmbodiChain/robot_challenge_tasks/robot_challenge_tasks/tasks/stack_bowls.py) will be reused directly.
- `force_reannotate` should remain exposed in the new implementation so it can be changed manually later.
- The current default plan is to keep `force_reannotate=True` in the new task code and construct a fresh affordance/semantics object for each pick stage so manual affordance selection can still happen per planning stage.
- Although the environment is dual-arm, planning will be done for one arm at a time, consistent with the current scripted task.

## Reference Files

- Existing task:
  - [robot_challenge_tasks/robot_challenge_tasks/tasks/stack_bowls.py](/home/dex/桌面/EmbodiChain/robot_challenge_tasks/robot_challenge_tasks/tasks/stack_bowls.py)
- Target task:
  - [robot_challenge_tasks/robot_challenge_tasks/tasks/stack_bowls_atomic.py](/home/dex/桌面/EmbodiChain/robot_challenge_tasks/robot_challenge_tasks/tasks/stack_bowls_atomic.py)
- Robochallenge environment entry:
  - [robot_challenge_tasks/scripts/run_env.py](/home/dex/桌面/EmbodiChain/robot_challenge_tasks/scripts/run_env.py)
- Robochallenge config:
  - [robot_challenge_tasks/configs/stack_bowls/aloha_stack_bowls.json](/home/dex/桌面/EmbodiChain/robot_challenge_tasks/configs/stack_bowls/aloha_stack_bowls.json)
- Atomic action example:
  - [scripts/tutorials/sim/atom_action.py](/home/dex/桌面/EmbodiChain/scripts/tutorials/sim/atom_action.py)
- Bowl-specific atomic action validation:
  - [scripts/tutorials/sim/atom_test_for_bowls.py](/home/dex/桌面/EmbodiChain/scripts/tutorials/sim/atom_test_for_bowls.py)
- Atomic action engine:
  - [embodichain/lab/sim/atomic_actions/engine.py](/home/dex/桌面/EmbodiChain/embodichain/lab/sim/atomic_actions/engine.py)
- Atomic action implementations:
  - [embodichain/lab/sim/atomic_actions/actions.py](/home/dex/桌面/EmbodiChain/embodichain/lab/sim/atomic_actions/actions.py)
- Affordance lifecycle:
  - [embodichain/lab/sim/atomic_actions/core.py](/home/dex/桌面/EmbodiChain/embodichain/lab/sim/atomic_actions/core.py)

## Current Understanding

The existing `stack_bowls.py` task generates a full joint-space scripted trajectory in `create_demo_action_list()`:

1. Select the arm for `bowl_mid`.
2. Pick `bowl_mid` and place it into `bowl_max`.
3. Select the arm for `bowl_min`.
4. If the arm changes, return the previous arm first.
5. Pick `bowl_min` and place it into `bowl_max`.
6. Return the final working arm.

The atomic action system already provides the three primitives needed for this task:

- `pick_up`
- `place`
- `move`

However, the current atomic engine only returns trajectory segments for the chosen control part. The Robochallenge environment still expects full robot actions over all active joints, so the new task must explicitly expand single-arm atomic trajectories into full-environment actions.

## Implementation Plan

### Phase 1: Environment Skeleton

- Create `StackBowlsAtomicEnv` in `stack_bowls_atomic.py`.
- Register it under a new gym id.
- Export the new task from `robot_challenge_tasks/tasks/__init__.py`.

### Phase 2: Shared Task Logic

- Reuse the original arm-selection logic from `stack_bowls.py`.
- Reuse the original pick offsets, place offsets, and return poses.
- Reuse the original success check logic.

### Phase 3: Atomic Action Integration

- Build left-arm and right-arm atomic action engines separately.
- Configure `pick_up`, `place`, and `move` for each arm with the correct arm and gripper control parts.
- Use the existing bowl mesh and rigid object entity to build `ObjectSemantics`.

### Phase 4: Affordance Handling

- Build a fresh `ObjectSemantics + AntipodalAffordance` object for each pick stage.
- Keep `force_reannotate` as an explicit interface in the new task implementation.
- Default implementation will not hardcode behavior beyond preserving the interface and current agreed baseline.

### Phase 5: Action Packing

- Convert returned atomic trajectories into full robot actions.
- Keep the inactive arm fixed at its current state while the selected arm executes.
- Return a final action list compatible with `EmbodiedEnv.create_demo_action_list()`.

### Phase 6: Validation

- Check that the generated action list is non-empty and has valid dimensions.
- Run the new environment through the Robochallenge runner.
- Manually select affordance regions for each bowl-pick stage.
- Verify that both `bowl_mid` and `bowl_min` are stacked successfully on `bowl_max`.

## Risks And Constraints

- The atomic action engine does not automatically generate full-environment actions; this must be handled in task code.
- The current atomic action examples are single-arm examples, while the target environment is dual-arm.
- No API should be invented. If any missing Dexsim or EmbodiChain interface is encountered during implementation, the next step is to stop and confirm before proceeding.

## Progress Status

- `Done`: Read the existing Robochallenge stack bowls task and extracted its control logic.
- `Done`: Read the bowl atomic-action tutorial and the single-bowl validation script.
- `Done`: Verified the real `AtomicActionEngine`, `PickUpAction`, `PlaceAction`, and `MoveAction` APIs from source.
- `Done`: Verified the actual `AntipodalAffordance.force_reannotate` behavior from source.
- `Done`: Confirmed with the user that a new environment should be added, and that original offset/return logic can be reused.
- `Done`: Created and initialized this `ATOMIC_ACTION.md` as the task progress log.
- `Done`: Implemented the first version of `stack_bowls_atomic.py` with atomic `pick_up`, `place`, and `move` stages.
- `Done`: Registered and exposed the new environment and added a dedicated atomic config entry.
- `Done`: Completed source-level validation by `py_compile` and package import in the `embodichain` conda environment.
- `Done`: Fixed the two-bowl annotation lifecycle issue by explicitly stopping the Viser server after each annotation and separating bowl annotation ports.
- `Done`: Replaced the pose-IK home return stage with deterministic return-to-initial-joint-space interpolation to avoid false affordance attribution when the actual failure is in home IK.
- `Done`: Changed the atomic pick path to explicitly resolve the annotated affordance into a concrete grasp pose, validate `pre_grasp` and `grasp` IK first, and then pass that exact validated pose into `pick_up`.
- `Done`: Reworked `stack_bowls_atomic.py` into a readable task-level rollout builder that creates per-arm atomic engines, expands single-arm trajectories into full dual-arm actions, and preserves the original arm-selection and place-offset rules.
- `Pending`: Run task-level Dexsim validation for the full stacking rollout.

## Operation Log

Every new implementation step should append to:

- `Progress Status`
- `Operation Log`

At minimum, each future update should record:

- what file was changed
- what behavior was added or modified
- whether the step was only code completion or also runtime-validated

- `2026-04-29`:
  - Changed file: `robot_challenge_tasks/robot_challenge_tasks/tasks/stack_bowls_atomic.py`
  - Behavior: implemented the full two-stage bowl stacking rollout with `pick_up`, `place`, and `move`, added full-robot trajectory packing for single-arm atomic plans, and kept per-stage affordance annotation with configurable `force_reannotate` and annotator ports.
  - Validation: source-level only (`py_compile` and import in the `embodichain` environment), not yet runtime-validated in Dexsim.
