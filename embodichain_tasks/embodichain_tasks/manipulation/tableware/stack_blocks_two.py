# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Stack two blocks with one atomic-action demonstration segment."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import partial
from typing import TYPE_CHECKING, Any

import torch

from embodichain.lab.gym.envs import DemoSegment, EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.utils.registration import register_env
from embodichain.utils import logger

if TYPE_CHECKING:
    from embodichain.lab.sim.atomic_actions import (
        AtomicActionEngine,
        CompiledTrajectory,
        ObjectSemantics,
    )
    from embodichain.lab.sim.motion.expansion import (
        TrajectoryAugmentationCfg,
        TrajectoryPhase,
    )
    from embodichain.lab.sim.objects import RigidObject

__all__ = ["StackBlocksTwoEnv"]

BASE_BLOCK_UID = "block_1"
STACK_BLOCK_UID = "block_2"
CONTROL_PART = "right_arm"
HAND_CONTROL_PART = "right_eef"
BLOCK_HEIGHT = 0.05
GRASP_OFFSET = (0.02, 0.0, -0.025)
HAND_OPEN_QPOS = 0.05
HAND_CLOSE_QPOS = 0.0
PICK_SAMPLE_INTERVAL = 90
PLACE_SAMPLE_INTERVAL = 90
HAND_INTERP_STEPS = 10
GRASP_HOLD_STEPS = 45
SETTLE_STEPS = 30

TRAJECTORY_VARIANT_EXTENSION = "trajectory_variants"
"""Optional ``extensions`` key enabling trajectory variant expansion."""

PATH_OPERATORS = ("joint_residual", "via_points", "nullspace_residual")
"""Operators that reshape a free phase without changing its sample count."""

RETIMED_OPERATORS = PATH_OPERATORS + ("retime",)
"""Operators allowed only after the grasp closes, where sample counts may change."""


@register_env("StackBlocksTwo-v1", max_episode_steps=600)
class StackBlocksTwoEnv(EmbodiedEnv):
    """Pick up ``block_2`` and place it on ``block_1`` as one segment."""

    def __init__(self, cfg: EmbodiedEnvCfg | None = None, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)

        base_block = self.sim.get_rigid_object(BASE_BLOCK_UID)
        stack_block = self.sim.get_rigid_object(STACK_BLOCK_UID)
        if base_block is None or stack_block is None:
            raise RuntimeError(
                "StackBlocksTwo-v1 requires rigid objects 'block_1' and 'block_2'."
            )
        self._base_block: RigidObject = base_block
        self._stack_block: RigidObject = stack_block
        self._arm_joint_ids: list[int] = list(
            self.robot.get_joint_ids(name=CONTROL_PART)
        )
        self._initialize_atomic_actions()
        self._initialize_trajectory_variants()

    def _initialize_atomic_actions(self) -> None:
        """Create the right-arm atomic-action engine and object semantics."""
        from embodichain.lab.sim.atomic_actions import (
            Affordance,
            AtomicActionEngine,
            ControlPartCommandProfile,
            ObjectSemantics,
        )
        from embodichain.lab.sim.motion.motion_generator import (
            MotionGenCfg,
            MotionGenerator,
        )
        from embodichain.lab.sim.motion.planners import ToppraPlannerCfg

        hand_dof = len(self.robot.get_joint_ids(name=HAND_CONTROL_PART))
        hand_open_qpos = torch.full(
            (hand_dof,), HAND_OPEN_QPOS, dtype=torch.float32, device=self.device
        )
        hand_close_qpos = torch.full(
            (hand_dof,), HAND_CLOSE_QPOS, dtype=torch.float32, device=self.device
        )
        motion_generator = MotionGenerator(
            cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=self.robot.uid))
        )
        self._action_engine: AtomicActionEngine = AtomicActionEngine(
            motion_generator,
            control_profiles={
                HAND_CONTROL_PART: ControlPartCommandProfile.joint_positions(
                    open=hand_open_qpos,
                    grasp=hand_close_qpos,
                )
            },
        )
        self._stack_block_semantics: ObjectSemantics = ObjectSemantics(
            affordance=Affordance(),
            geometry={},
            label=STACK_BLOCK_UID,
            entity_id=STACK_BLOCK_UID,
        )

    def _initialize_trajectory_variants(self) -> None:
        """Decode the optional trajectory-variant extension.

        The feature is opt-in. Without an ``extensions.trajectory_variants`` entry
        the task plans exactly one trajectory per environment, as before.
        """
        from embodichain.lab.sim.motion.expansion import TrajectoryAugmentationCfg

        self._variant_cfg: TrajectoryAugmentationCfg | None = None
        self._variant_jacobian_columns: list[int] = []
        self._variant_solver_columns: list[int] = []
        self._variant_episode = 0
        settings = getattr(self, TRAJECTORY_VARIANT_EXTENSION, None)
        if settings is None:
            return
        if not isinstance(settings, Mapping):
            raise TypeError(
                f"'{TRAJECTORY_VARIANT_EXTENSION}' must be a mapping of augmentation "
                "settings."
            )
        decoded = dict(settings)
        if not decoded.pop("enabled", True):
            return
        cfg = TrajectoryAugmentationCfg.from_mapping(decoded)
        if cfg.factors.ik.enabled:
            self._resolve_jacobian_columns()
        self._variant_cfg = cfg

    def _resolve_jacobian_columns(self) -> None:
        """Map the arm solver's Jacobian columns onto this task's joint order.

        ``BaseSolver.get_jacobian`` orders its columns by the solver's own joint
        names, which need not match the robot's public joint order. Resolving
        the permutation here keeps the redundancy operator's ``(N, R, C)``
        contract aligned with ``controlled_joint_indices``.
        """
        solver = self.robot.get_solver(CONTROL_PART)
        solver_names = list(getattr(solver, "joint_names", None) or ())
        if not solver_names:
            raise RuntimeError(
                "Redundancy expansion needs a kinematic solver for control part "
                f"'{CONTROL_PART}' that declares its joint names."
            )
        robot_names = list(self.robot.joint_names)
        controlled = [robot_names[index] for index in self._arm_joint_ids]
        missing = [name for name in controlled if name not in solver_names]
        if missing:
            raise RuntimeError(
                f"Control part '{CONTROL_PART}' joints {missing} have no column in "
                "its solver Jacobian; redundancy expansion cannot align them."
            )
        self._variant_solver_columns = [
            robot_names.index(name) for name in solver_names
        ]
        self._variant_jacobian_columns = [
            solver_names.index(name) for name in controlled
        ]

    def _reference_jacobians(self, trajectory: torch.Tensor) -> list[torch.Tensor]:
        """Return one row-aligned task Jacobian sequence per environment.

        The solver Jacobian is expressed in the arm's base frame, so dropping
        its angular rows selects base-frame rotational freedoms. This task
        grasps from directly above, which makes the angular-z row the gripper's
        own spin about its approach axis.

        Evaluation stays on the robot's own device; the caller moves the result
        wherever the augmentation operators run.
        """
        from embodichain.compute.kinematics import select_jacobian_rows

        solver = self.robot.get_solver(CONTROL_PART)
        rows = list(self._variant_cfg.factors.ik.task_rows)
        jacobians = []
        for env_id in range(trajectory.shape[0]):
            arm_qpos = trajectory[env_id][:, self._variant_solver_columns]
            full = solver.get_jacobian(arm_qpos)
            jacobians.append(
                select_jacobian_rows(full, rows)[:, :, self._variant_jacobian_columns]
            )
        return jacobians

    def _variant_phases(
        self,
        pick_compiled: CompiledTrajectory,
        place_compiled: CompiledTrajectory,
        *,
        hold_steps: int,
        place_offset: int,
    ) -> tuple[TrajectoryPhase, ...]:
        """Annotate the compiled trajectory with fixed and variable phases.

        Phase boundaries come from the atomic actions' own named segments, not
        from re-deriving their internal step arithmetic. The gripper-close and
        gripper-open segments are contact phases and the dwell after closing is
        a hold, so no operator can move a grasp, a placement, or the settling
        pause that follows them.

        Retiming is confined to phases at or after the lift. The executor clears
        the held block's dynamics at one shared step index, so changing how long
        the approach takes would desynchronize that event across environments.
        """
        from embodichain.lab.sim.motion.expansion import TrajectoryPhase

        approach = pick_compiled.segment(0, "approach")
        close = pick_compiled.segment(0, "close")
        lift = pick_compiled.segment(0, "lift")
        expected_close_stop = (
            round((PICK_SAMPLE_INTERVAL - HAND_INTERP_STEPS) * 0.6) + HAND_INTERP_STEPS
        )
        if close.stop != expected_close_stop:
            raise RuntimeError(
                "The grasp-hold insertion index no longer matches the compiled "
                f"close segment ({close.stop} vs {expected_close_stop}); update "
                "_insert_grasp_hold before annotating contact phases."
            )
        phases = [
            TrajectoryPhase(
                "pick_approach", approach.start, approach.stop, "free", PATH_OPERATORS
            ),
            TrajectoryPhase("grasp_close", close.start, close.stop, "contact"),
        ]
        # A trajectory too short for the dwell keeps no hold phase; an empty
        # sample range is not a valid phase.
        if hold_steps > 0:
            phases.append(
                TrajectoryPhase(
                    "grasp_hold", close.stop, close.stop + hold_steps, "hold"
                )
            )
        phases.append(
            TrajectoryPhase(
                "lift",
                lift.start + hold_steps,
                lift.stop + hold_steps,
                "free",
                RETIMED_OPERATORS,
            )
        )
        for name, kind in (
            ("approach", "free"),
            ("release", "contact"),
            ("retract", "free"),
        ):
            segment = place_compiled.segment(0, name)
            phases.append(
                TrajectoryPhase(
                    f"place_{name}",
                    segment.start + place_offset,
                    segment.stop + place_offset,
                    kind,
                    RETIMED_OPERATORS if kind == "free" else (),
                )
            )
        return tuple(phases)

    def _expand_trajectory_variants(
        self,
        trajectory: torch.Tensor,
        phases: tuple[TrajectoryPhase, ...],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Give each parallel environment a different way of running one plan.

        Every environment keeps the waypoints its own affordance produced; only
        the free motion between them changes. Environment zero always replays
        its unmodified reference so the trusted trajectory stays in the dataset.

        Successive episodes start further along the list of combinations instead of
        repeating the first ones, and each episode draws its own residuals because the
        reference identity carries the episode index.

        Rows whose proposal is rejected fall back to their own reference rather
        than being dropped, and both the assignment and the fallbacks are
        reported in the segment metadata.
        """
        from embodichain.lab.sim.motion.expansion import (
            SceneCase,
            TrajectoryTemplate,
            expand_row_variants,
        )

        samples = int(trajectory.shape[1])
        intervals = torch.full(
            (samples,), float(self.step_dt), dtype=torch.float32, device="cpu"
        )
        intervals[0] = 0
        joint_names = tuple(self.robot.joint_names)
        positions = trajectory.detach().to(device="cpu", dtype=torch.float32)
        episode = self._variant_episode
        templates = [
            TrajectoryTemplate(
                source_id="stack_blocks_two_atomic",
                source_revision=f"pick{PICK_SAMPLE_INTERVAL}_place{PLACE_SAMPLE_INTERVAL}",
                template_id=f"env_{env_id}_episode_{episode}",
                joint_names=joint_names,
                positions=positions[env_id],
                dt=intervals,
                phases=phases,
                allowed_operators=RETIMED_OPERATORS,
                controlled_joint_indices=tuple(self._arm_joint_ids),
            )
            for env_id in range(self.num_envs)
        ]
        limits = self.robot.get_qpos_limits()[0].detach().to(device="cpu")
        jacobians = None
        if self._variant_cfg.factors.ik.enabled:
            jacobians = [
                value.detach().to(device="cpu", dtype=torch.float64)
                for value in self._reference_jacobians(trajectory)
            ]
        result = expand_row_variants(
            templates,
            cfg=self._variant_cfg,
            cases=[
                SceneCase(
                    scene_case_id="stack_blocks_two",
                    initial_state_id=f"env_{env_id}_episode_{episode}",
                    scene_signature=f"{BASE_BLOCK_UID}+{STACK_BLOCK_UID}",
                    task_id="StackBlocksTwo-v1",
                    robot_profile_id=self.robot.uid,
                )
                for env_id in range(self.num_envs)
            ],
            joint_limits=limits,
            control_dt=float(self.step_dt),
            task_jacobians=jacobians,
            ordinal_offset=episode * max(self.num_envs - 1, 1),
        )
        self._variant_episode = episode + 1
        batch = result.candidates
        report = {
            "episode_index": episode,
            "requested_variants": self.num_envs,
            "accepted_variants": len(result.variants),
            "distinct_variants": len({variant.ordinal for variant in result.variants}),
            "fallbacks": sum(result.rejected.values()),
            "rejected": dict(sorted(result.rejected.items())),
            "row_variants": [dict(row) for row in batch.factors],
            "row_lengths": batch.valid_length.tolist(),
            # Retiming moves phase boundaries per row, so a dataset consumer
            # cannot recover contact windows from the nominal annotation alone.
            "row_phases": [
                [
                    [phase.phase_id, phase.start_index, phase.stop_index, phase.kind]
                    for phase in row
                ]
                for row in batch.phases
            ],
        }
        if report["fallbacks"]:
            logger.log_warning(
                "Trajectory variant expansion reused the reference trajectory for "
                f"{report['fallbacks']} environment(s): {report['rejected']}."
            )
        expanded = batch.positions.to(device=trajectory.device, dtype=trajectory.dtype)
        return expanded, report

    def create_demo_segments(self, **kwargs: Any) -> tuple[DemoSegment]:
        """Plan the complete stacking task as exactly one semantic segment."""
        del kwargs
        plan_success, trajectory, source_pose, target_pose, variants = (
            self._plan_stack()
        )
        metadata: dict[str, Any] = {
            "segment_index": 0,
            "segment_count": 1,
            "planning_success": plan_success.detach().cpu().tolist(),
            "source_pose": source_pose.detach().cpu().tolist(),
            "target_pose": target_pose.detach().cpu().tolist(),
            "atomic_actions": ["pick_up", "place"],
        }
        if variants:
            metadata["trajectory_variants"] = variants
        return (
            DemoSegment(
                actions=self._iter_segment_actions(trajectory),
                name="stack_block_2_on_block_1",
                target_uid=STACK_BLOCK_UID,
                instruction="Pick up block 2 and place it on top of block 1.",
                progress_total_steps=int(trajectory.shape[1]) + SETTLE_STEPS,
                metadata=metadata,
                validator=partial(self._validate_stack, plan_success.detach().clone()),
            ),
        )

    def _plan_stack(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Plan PickUp then Place while threading the held-object state.

        When trajectory-variant expansion is enabled the planned batch is replaced
        by one distinct execution variant per environment before it is returned.
        """
        from embodichain.lab.sim.atomic_actions import (
            ActionInvocation,
            EntityState,
            GraspGoal,
            MotionPolicy,
            PickUpOptions,
            PlaceGoal,
            PlaceOptions,
            SceneSnapshot,
        )

        source_pose = self._stack_block.get_local_pose(to_matrix=True).to(
            device=self.device, dtype=torch.float32
        )
        base_pose = self._base_block.get_local_pose(to_matrix=True).to(
            device=self.device, dtype=torch.float32
        )
        grasp_pose = self.robot.compute_fk(
            qpos=self.robot.get_qpos()[:, self.robot.get_joint_ids(name=CONTROL_PART)],
            name=CONTROL_PART,
            to_matrix=True,
        )
        grasp_pose[:, :3, 3] = source_pose[:, :3, 3] + torch.tensor(
            GRASP_OFFSET, dtype=torch.float32, device=self.device
        )
        endpoints = {
            "primary": {
                "motion": CONTROL_PART,
                "grasp": HAND_CONTROL_PART,
            }
        }
        pick_binding = self._action_engine.bind_control_parts(
            "pick_up",
            endpoints,
        )
        place_binding = self._action_engine.bind_control_parts(
            "place",
            endpoints,
        )
        pick_compiled = self._action_engine.compile(
            (
                ActionInvocation(
                    skill_id="pick_up",
                    goal=GraspGoal(
                        self._stack_block_semantics,
                        grasp_xpos=grasp_pose,
                    ),
                    binding=pick_binding,
                    motion_policy=MotionPolicy(sample_count=PICK_SAMPLE_INTERVAL),
                    skill_options=PickUpOptions(
                        pre_grasp_distance=0.12,
                        lift_height=0.15,
                        hand_interp_steps=HAND_INTERP_STEPS,
                    ),
                ),
            ),
            self._action_engine.initial_context(
                scene=SceneSnapshot(
                    timestamp=0.0,
                    version=0,
                    entities={STACK_BLOCK_UID: EntityState(source_pose)},
                ),
                control_dt=self.step_dt,
            ),
        )
        pick_success = pick_compiled.plan_success
        pick_trajectory = pick_compiled.trajectory.positions
        picked_context = pick_compiled.projected_context
        pick_trajectory = self._insert_grasp_hold(pick_trajectory)

        target_pose = source_pose.clone()
        target_pose[:, :3, 3] = base_pose[:, :3, 3]
        target_pose[:, 2, 3] += BLOCK_HEIGHT
        held = picked_context.get_held_object(CONTROL_PART)
        if held is None or not bool(pick_success.all().item()):
            return (
                torch.zeros_like(pick_success, dtype=torch.bool),
                self._ensure_nonempty_trajectory(pick_trajectory),
                source_pose,
                target_pose,
                {},
            )

        place_eef_pose = torch.bmm(target_pose, held.object_to_eef)
        place_compiled = self._action_engine.compile(
            (
                ActionInvocation(
                    skill_id="place",
                    goal=PlaceGoal(place_eef_pose),
                    binding=place_binding,
                    motion_policy=MotionPolicy(sample_count=PLACE_SAMPLE_INTERVAL),
                    skill_options=PlaceOptions(
                        lift_height=0.10,
                        hand_interp_steps=HAND_INTERP_STEPS,
                    ),
                ),
            ),
            picked_context,
        )
        place_success = place_compiled.plan_success
        place_trajectory = place_compiled.trajectory.positions
        trajectory = torch.cat((pick_trajectory, place_trajectory), dim=1)
        variants: dict[str, Any] = {}
        if self._variant_cfg is not None and trajectory.shape[1] > 0:
            phases = self._variant_phases(
                pick_compiled,
                place_compiled,
                hold_steps=int(pick_trajectory.shape[1]) - PICK_SAMPLE_INTERVAL,
                place_offset=int(pick_trajectory.shape[1]),
            )
            trajectory, variants = self._expand_trajectory_variants(trajectory, phases)
        trajectory = self._ensure_nonempty_trajectory(trajectory)
        return (
            pick_success & place_success,
            trajectory,
            source_pose,
            target_pose,
            variants,
        )

    def _insert_grasp_hold(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Dwell at the closed grasp pose before beginning the lift phase."""
        close_end_step = (
            round((PICK_SAMPLE_INTERVAL - HAND_INTERP_STEPS) * 0.6) + HAND_INTERP_STEPS
        )
        if trajectory.shape[1] < close_end_step:
            return trajectory
        grasp_action = trajectory[:, close_end_step - 1 : close_end_step]
        grasp_hold = grasp_action.repeat(1, GRASP_HOLD_STEPS, 1)
        return torch.cat(
            (
                trajectory[:, :close_end_step],
                grasp_hold,
                trajectory[:, close_end_step:],
            ),
            dim=1,
        )

    def _ensure_nonempty_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Return at least one hold command so planning failure is observable."""
        if trajectory.shape[1] > 0:
            return trajectory
        return self.robot.get_qpos().clone().unsqueeze(1)

    def _iter_segment_actions(self, trajectory: torch.Tensor) -> Iterable[torch.Tensor]:
        """Replay the atomic trajectory and wait for the released block to settle."""
        close_end_step = min(
            round((PICK_SAMPLE_INTERVAL - HAND_INTERP_STEPS) * 0.6)
            + HAND_INTERP_STEPS
            + GRASP_HOLD_STEPS,
            trajectory.shape[1],
        )
        for step_index, action in enumerate(trajectory.unbind(dim=1), start=1):
            yield action
            if step_index == close_end_step:
                self._stack_block.clear_dynamics()

        hold_action = trajectory[:, -1].clone()
        for _ in range(SETTLE_STEPS):
            yield hold_action

    def _validate_stack(self, plan_success: torch.Tensor) -> torch.Tensor:
        """Require both a successful atomic plan and a physically valid stack."""
        task_success = self.is_task_success()
        success = plan_success.to(device=self.device) & task_success
        if not bool(success.all().item()):
            base_pose = self._base_block.get_local_pose(to_matrix=True)
            stack_pose = self._stack_block.get_local_pose(to_matrix=True)
            base_pos = base_pose[:, :3, 3]
            stack_pos = stack_pose[:, :3, 3]
            logger.log_warning(
                "Stack validation failed: "
                f"planning_success={plan_success.detach().cpu().tolist()}, "
                f"base_position={base_pos.detach().cpu().tolist()}, "
                f"stack_position={stack_pos.detach().cpu().tolist()}, "
                f"base_fallen={self._is_fall(base_pose).detach().cpu().tolist()}, "
                f"stack_fallen={self._is_fall(stack_pose).detach().cpu().tolist()}, "
                f"stack_z_axis={stack_pose[:, :3, 2].detach().cpu().tolist()}."
            )
        return success

    def is_task_success(self, **kwargs: Any) -> torch.Tensor:
        """Return whether block 2 is upright and centered on block 1."""
        del kwargs
        block1_pose = self._base_block.get_local_pose(to_matrix=True)
        block2_pose = self._stack_block.get_local_pose(to_matrix=True)
        block1_pos = block1_pose[:, :3, 3]
        block2_pos = block2_pose[:, :3, 3]

        expected_block2_pos = block1_pos.clone()
        expected_block2_pos[:, 2] += BLOCK_HEIGHT
        tolerance = torch.tensor(
            [0.025, 0.025, 0.012], dtype=torch.float32, device=self.device
        )
        within_tolerance = torch.all(
            torch.abs(block2_pos - expected_block2_pos) < tolerance, dim=1
        )
        return (
            within_tolerance & ~self._is_fall(block1_pose) & ~self._is_fall(block2_pose)
        )

    @staticmethod
    def _is_fall(pose: torch.Tensor) -> torch.Tensor:
        """Return whether an object's local z-axis tilts by at least 45 degrees."""
        pose_rz = pose[:, :3, 2]
        world_z_axis = torch.tensor([0, 0, 1], dtype=pose.dtype, device=pose.device)
        dot_product = torch.sum(pose_rz * world_z_axis, dim=-1).clamp(-1.0, 1.0)
        return torch.arccos(dot_product) >= torch.pi / 4
