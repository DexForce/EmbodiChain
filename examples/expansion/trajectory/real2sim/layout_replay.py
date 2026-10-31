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

"""Run a saved synthetic prototype in one explicitly selected scene layout.

This is a diagnostic runner: no dataset is submitted, and collision checks are
not certified. Each requested layout is reported independently, including
planning and execution failures.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import gymnasium
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from embodichain.data_pipeline.real2sim import load_prototype
from embodichain.lab.gym.envs.managers.real2sim_sink import (
    build_real2sim_execution_trace,
    capture_real2sim_episode,
    submit_and_discard_real2sim_episode,
    Real2SimLeRobotEpisodeSink,
)
from embodichain.lab.gym.envs.managers.cfg import DatasetFunctorCfg
from embodichain.lab.gym.envs.managers.datasets import LeRobotRecorder
from embodichain.lab.gym.utils.gym_utils import build_env_cfg_from_args
from embodichain.lab.scripts.run_task import (
    _create_parser,
    discover_task_packages,
    execute_init_hooks,
)
from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.motion.expansion import (
    CandidateIdentity,
    ValidationCheck,
    ValidationResult,
)
from embodichain.lab.sim.motion.expansion.real2sim_collision import (
    BoxCollisionBody,
    PlaneCollisionBody,
    SampledRobotCollisionChecker,
)
from embodichain.lab.sim.scene.expansion import ScenePoseChange, SceneVariant
from embodichain.lab.task_program.integrations.scene_expansion import (
    execute_scene_variant,
)
from embodichain.lab.task_program.integrations.simulation.scene_expansion import (
    SimulationSceneExpansionHost,
)
from embodichain.lab.task_program.integrations.simulation.real2sim_collision import (
    TabletopPrototypeCollisionValidator,
)
from embodichain.lab.task_program.integrations.expansion import (
    TaskProgramMappedRetimeFactory,
    TaskProgramSameGridExpansionFactory,
)
from embodichain.lab.task_program.integrations.nominal import (
    TracingTaskProgramNominalSources,
)
from embodichain.lab.sim.atomic_actions import JointPositionTarget

from replay import _ReplayFactory, _TASK

__all__: list[str] = []

_LAYOUTS = {
    "nominal": (None, 0.0, 0.0, 0.0),
    "object_x_plus": ("cube", 0.03, 0.0, 0.0),
    "object_x_minus": ("cube", -0.03, 0.0, 0.0),
    "object_y_plus": ("cube", 0.0, 0.03, 0.0),
    "object_y_minus": ("cube", 0.0, -0.03, 0.0),
    "object_yaw_plus": ("cube", 0.0, 0.0, 15.0),
    "object_yaw_minus": ("cube", 0.0, 0.0, -15.0),
    "target_x_plus": ("platform", 0.03, 0.0, 0.0),
    "target_x_minus": ("platform", -0.03, 0.0, 0.0),
    "target_yaw_plus": ("platform", 0.0, 0.0, 15.0),
    "target_yaw_minus": ("platform", 0.0, 0.0, -15.0),
}


class _CollisionGuardFactory:
    """Validate each nominal plan before any selected expansion is applied."""

    def __init__(self, validator, delegate=None):
        self.validator = validator
        self.delegate = delegate
        self.nominal_plans = {}

    def create_plan_transform(self, request, *, engine):
        downstream = (
            None
            if self.delegate is None
            else self.delegate.create_plan_transform(request, engine=engine)
        )

        def guard(resolved, context, plan):
            self.nominal_plans[request.analysis_call_index] = plan.snapshot()
            report = self.validator.validate(resolved, context, plan)
            if not report.accepted:
                raise ValueError(
                    f"nominal collision at sample {report.first_failure_sample} "
                    f"in {report.first_failure_phase}: {report.first_failure_pair}: "
                    f"{report.reason}"
                )
            metadata = dict(plan.diagnostics.metadata)
            metadata.update(
                collision_evidence="host_verified",
                collision_scope="sampled_tabletop_geometry_with_gripper_contact_projection",
                geometry_compatibility="asset_identity_matched",
            )
            checked = replace(
                plan, diagnostics=replace(plan.diagnostics, metadata=metadata)
            )
            return (
                checked
                if downstream is None
                else downstream(resolved, context, checked)
            )

        return guard


def main() -> None:
    """Apply a layout using the production B=1 scene host and run the program."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layout", choices=tuple(_LAYOUTS), required=True)
    parser.add_argument(
        "--recipe",
        choices=(
            "nominal",
            "joint_7",
            "joint_11",
            "task_space",
            "time_115",
            "time_130",
        ),
        default="nominal",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-lerobot", action="store_true")
    collision_mode = parser.add_mutually_exclusive_group(required=True)
    collision_mode.add_argument("--diagnostic-no-collision-check", action="store_true")
    collision_mode.add_argument("--validate-sampled-collision", action="store_true")
    options = parser.parse_args()
    if options.save_lerobot and not options.validate_sampled_collision:
        raise ValueError("LeRobot output requires sampled plan collision checks")
    reference = options.reference_dir.resolve()
    source_report = json.loads((reference / "report.json").read_text(encoding="utf-8"))
    if (
        source_report.get("origin") != "synthetic_reference"
        or not source_report.get("baseline_verified")
        or not source_report.get("source_artifacts_complete")
    ):
        raise ValueError(
            "reference must come from a completed verified synthetic capture"
        )
    prototypes = {
        index: load_prototype(reference / "prototypes" / str(index) / "prototype.json")
        for index in range(3)
    }
    output = options.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    args = _create_parser().parse_args(
        [
            "--gym-config",
            str(_TASK),
            "--device",
            options.device,
            "--headless",
            "--renderer",
            "no-render",
            "--disable_sensor",
            "--filter_dataset_saving",
            "--seed",
            str(options.seed),
        ]
    )
    discover_task_packages()
    execute_init_hooks()
    cfg, gym_config, action_config = build_env_cfg_from_args(args)
    cfg.record_trajectory = True
    cfg.trajectory_auto_save = False
    cfg.init_rollout_buffer = options.save_lerobot
    env = gymnasium.make(id=gym_config["id"], cfg=cfg, **action_config)
    recorder = None
    report = {
        "origin": "synthetic_reference",
        "layout": options.layout,
        "recipe": options.recipe,
        "collision_check": "unverified",
        "dataset_committed": False,
        "task_verified": False,
    }
    try:
        env.reset(seed=options.seed, options={"save_data": False})
        target = env.unwrapped
        program = target.compile_task_program(target.cfg.task_program)
        entity_id, dx, dy, yaw = _LAYOUTS[options.layout]
        factory = TracingTaskProgramNominalSources(
            program,
            _ReplayFactory(
                program, prototypes, diagnostic=True, retarget_pick=entity_id == "cube"
            ),
        )
        if factory.program_fingerprint != source_report["program_fingerprint"]:
            raise ValueError("saved references belong to a different compiled program")
        cube = target.sim.get_rigid_object("cube")
        platform = target.sim.get_rigid_object("platform")
        collision_validator = None
        if options.validate_sampled_collision:
            if (
                set(target.sim.get_rigid_object_uid_list()) != {"cube", "platform"}
                or len(target.sim.get_robot_uid_list()) != 1
                or target.sim.get_articulation_uid_list()
                or target.sim.get_deformable_object_uid_list()
                or target.sim.get_rigid_object_group_uid_list()
            ):
                raise ValueError(
                    "sampled collision scene has unsupported physical bodies"
                )
            checker = SampledRobotCollisionChecker(
                target.robot.cfg.fpath,
                joint_names=target.robot.joint_names,
                bodies=(
                    BoxCollisionBody.from_rigid_object(cube),
                    BoxCollisionBody.from_rigid_object(platform),
                    PlaneCollisionBody("ground", np.eye(4)),
                ),
                world_from_base=target.robot.get_local_pose(to_matrix=True)[0]
                .detach()
                .cpu()
                .numpy(),
            )
            collision_validator = TabletopPrototypeCollisionValidator(
                robot=target.robot,
                checker=checker,
                prototypes=tuple(prototypes.values()),
                object_uid="cube",
                target_uid="platform",
                ground_uid="ground",
                finger_links=("gripper_finger1_link_1", "gripper_finger2_link_1"),
                finger_joints=("gripper_finger1_joint_1", "gripper_finger2_joint_1"),
                closing_directions=(1, 1),
                base_link="arm_base_link",
                contact_depth_m=0.002,
                base_ground_depth_m=0.0035,
            )
        poses = {
            name: target.sim.get_rigid_object(name)
            .get_local_pose(to_matrix=True)[0]
            .clone()
            for name in ("cube", "platform")
        }
        if entity_id is not None:
            pose = poses[entity_id]
            pose[0, 3] += dx
            pose[1, 3] += dy
            angle = np.deg2rad(yaw)
            rotation = torch.tensor(
                [
                    [np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=pose.dtype,
                device=pose.device,
            )
            pose[:3, :3] = rotation @ pose[:3, :3]
        changes = (
            ()
            if entity_id is None
            else (ScenePoseChange(entity_id, poses[entity_id].cpu().tolist()),)
        )
        variant = SceneVariant(
            "synthetic_ur5_reference_v1",
            options.seed,
            tuple(_LAYOUTS).index(options.layout),
            changes,
        )

        def initial_checks() -> ValidationResult:
            actual = cube.get_local_pose(to_matrix=True)[0]
            target_pose = platform.get_local_pose(to_matrix=True)[0]
            position_error = max(
                float(torch.linalg.vector_norm(actual[:3, 3] - poses["cube"][:3, 3])),
                float(
                    torch.linalg.vector_norm(
                        target_pose[:3, 3] - poses["platform"][:3, 3]
                    )
                ),
            )
            relative_rotations = torch.stack(
                (
                    actual[:3, :3].T @ poses["cube"][:3, :3],
                    target_pose[:3, :3].T @ poses["platform"][:3, :3],
                )
            )
            cosine = (
                (relative_rotations.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
            ).clamp(-1, 1)
            rotation_error = float(torch.rad2deg(torch.acos(cosine)).max())
            return ValidationResult(
                (
                    ValidationCheck(
                        "initial.layout",
                        (
                            "passed"
                            if position_error < 0.003 and rotation_error < 1.0
                            else "failed"
                        ),
                        metrics={
                            "position_error_m": position_error,
                            "rotation_error_deg": rotation_error,
                        },
                    ),
                )
            )

        def measured_checks() -> ValidationResult:
            steps = int(target._traj_steps[0].item())
            if target._traj_buffer is None or steps < 20:
                return ValidationResult(
                    (
                        ValidationCheck(
                            "placement.measured",
                            "unavailable",
                            "insufficient actual trajectory",
                        ),
                    )
                )
            values = (
                target._traj_buffer["states", "rigid_objects", "cube", "pose"][
                    0, : steps + 1
                ]
                .cpu()
                .numpy()
            )
            final = values[-1, :3]
            goal = (
                platform.get_local_pose(to_matrix=True)[0, :3, 3].detach().cpu().numpy()
            )
            xy_error = float(np.linalg.norm(final[:2] - goal[:2]))
            height_error = float(abs(final[2] - goal[2] - 0.045))
            drift = float(np.linalg.norm(values[-1, :3] - values[-20, :3]))
            lift = float(np.max(values[:, 2]) - values[0, 2])
            return ValidationResult(
                (
                    ValidationCheck(
                        "placement.measured",
                        (
                            "passed"
                            if xy_error <= 0.025 and height_error <= 0.01
                            else "failed"
                        ),
                        metrics={
                            "xy_error_m": xy_error,
                            "height_error_m": height_error,
                        },
                    ),
                    ValidationCheck(
                        "placement.stable",
                        "passed" if drift <= 0.003 else "failed",
                        metrics={"drift_m": drift},
                    ),
                    ValidationCheck(
                        "object.lifted",
                        "passed" if lift >= 0.03 else "failed",
                        metrics={"lift_m": lift},
                    ),
                )
            )

        host = SimulationSceneExpansionHost(env, initial_validator=initial_checks)
        transform = None
        if options.recipe != "nominal":
            velocities = target.robot.get_qvel_limits()[0].detach().cpu().tolist()
            common = {
                "velocity_limits": velocities,
                "acceleration_limits": [50.0] * len(velocities),
                "call_indices": (0, 1, 2),
                "diagnostic_unchecked": collision_validator is None,
            }
            if collision_validator is not None:
                common["collision_validator"] = collision_validator
            if options.recipe.startswith("time_"):
                transform = TaskProgramMappedRetimeFactory(
                    duration_scale=1.15 if options.recipe == "time_115" else 1.30,
                    **common,
                )
            else:

                def kinematics_factory(resolved, context, template):
                    endpoint = resolved.binding.endpoint(
                        "primary", "motion"
                    ).require_target(JointPositionTarget)
                    if tuple(template.controlled_joint_indices) != tuple(
                        endpoint.joint_ids
                    ):
                        raise ValueError(
                            "example task-space IK needs every arm joint to be active"
                        )
                    solver = target.robot.get_solver(endpoint.control_part)
                    base = target.robot.get_link_pose(
                        link_name=solver.root_link_name, env_ids=[0], to_matrix=True
                    )[0]
                    tcp = torch.as_tensor(
                        solver.tcp_xpos, dtype=base.dtype, device=base.device
                    )

                    def kinematics(q):
                        q_device = q.to(base.device)
                        poses = base[None] @ solver.get_fk(q_device)
                        jacobian = solver.get_jacobian(q_device, locations=tcp[:3, 3])
                        jacobian = torch.cat(
                            (
                                base[:3, :3] @ jacobian[:, :3],
                                base[:3, :3] @ jacobian[:, 3:],
                            ),
                            dim=1,
                        )
                        return poses.to(q), jacobian.to(q)

                    return kinematics

                transform = TaskProgramSameGridExpansionFactory(
                    operator=(
                        "task_space_residual"
                        if options.recipe == "task_space"
                        else "joint_residual"
                    ),
                    seed=11 if options.recipe == "joint_11" else 7,
                    geometry_family_id=variant.variant_id,
                    initial_state_id=f"initial:{variant.variant_id}",
                    kinematics_factory=(
                        kinematics_factory if options.recipe == "task_space" else None
                    ),
                    **common,
                )
        plan_factory = (
            transform
            if collision_validator is None
            else _CollisionGuardFactory(collision_validator, transform)
        )
        result = execute_scene_variant(
            env,
            variant,
            prepare_scene=host,
            measured_validator=measured_checks,
            task_program=program,
            nominal_source_factory=factory,
            plan_transform_factory=plan_factory,
            trace_actions=True,
        )
        if not result.program_success:
            report["scene"] = result.to_metadata()
            report["execution"] = (
                None if result.execution is None else result.execution.to_metadata()
            )
            if collision_validator is not None:
                report["sampled_plan_collisions"] = [
                    {"unit_id": unit, **asdict(check)}
                    for unit, check in collision_validator.reports
                ]
                if isinstance(plan_factory, _CollisionGuardFactory):
                    arrays = {}
                    for index, plan in plan_factory.nominal_plans.items():
                        arrays[f"call_{index}_positions"] = (
                            plan.joint_trajectory.positions.cpu().numpy()
                        )
                        arrays[f"call_{index}_dt"] = (
                            plan.joint_trajectory.dt.cpu().numpy()
                        )
                    if arrays:
                        np.savez_compressed(
                            output / "failed_nominal_plans.npz", **arrays
                        )
            raise RuntimeError(
                "program execution failed; see retained execution evidence"
            )
        post_checks = []
        if (
            result.program_success
            and result.execution is not None
            and target._traj_buffer is not None
        ):
            segments = {segment.name: segment for segment in result.execution.segments}
            pick, transport = segments.get("pick"), segments.get("transport")
            if (
                pick is not None
                and transport is not None
                and transport.end_step > transport.start_step
            ):
                states = target._traj_buffer["states"]
                cube_poses = states["rigid_objects", "cube", "pose"][0]
                pick_lift = float(
                    cube_poses[pick.end_step, 2] - cube_poses[pick.start_step, 2]
                )
                span = slice(transport.start_step, transport.end_step + 1)
                object_values = cube_poses[span].cpu().numpy()
                qpos = states["robot", "qpos"][0, span]
                arm_ids = target.robot.get_joint_ids(name="arm")
                tcp = (
                    target.robot.compute_batch_fk(
                        qpos=qpos[None, :, arm_ids], name="arm", to_matrix=True
                    )[0]
                    .cpu()
                    .numpy()
                )
                object_rotation = Rotation.from_quat(object_values[:, 3:7]).as_matrix()
                object_to_tcp = np.einsum(
                    "nij,nj->ni",
                    object_rotation.transpose(0, 2, 1),
                    tcp[:, :3, 3] - object_values[:, :3],
                )
                relative_rotation = object_rotation.transpose(0, 2, 1) @ tcp[:, :3, :3]
                translation_drift = float(
                    np.linalg.norm(object_to_tcp - object_to_tcp[0], axis=1).max()
                )
                angle_drift = float(
                    np.degrees(
                        Rotation.from_matrix(
                            relative_rotation[0].T @ relative_rotation
                        ).magnitude()
                    ).max()
                )
                post_checks = [
                    {
                        "check_id": "pick.measured_lift",
                        "status": "passed" if pick_lift >= 0.03 else "failed",
                        "lift_m": pick_lift,
                    },
                    {
                        "check_id": "transport.held_relation",
                        "status": (
                            "passed"
                            if translation_drift <= 0.015 and angle_drift <= 8.0
                            else "failed"
                        ),
                        "translation_drift_m": translation_drift,
                        "rotation_drift_deg": angle_drift,
                    },
                ]
        call_variants = []
        plan_arrays = {}
        if transform is not None:
            for record in transform.records:
                prefix = f"call_{record.call_index}"
                entry = {
                    "call_index": record.call_index,
                    "source_unit_id": record.source_unit_id,
                    "prototype_sha256": hashlib.sha256(
                        (
                            reference
                            / "prototypes"
                            / str(record.call_index)
                            / "prototype.json"
                        ).read_bytes()
                    ).hexdigest(),
                    "collision_evidence": record.collision_evidence,
                }
                if isinstance(transform, TaskProgramMappedRetimeFactory):
                    entry.update(
                        operator="retime",
                        duration_scale=record.duration_scale,
                        source_sample_offset=record.source_sample_offset,
                    )
                    normalized = prototypes[record.call_index].unit.report.time_map
                    plan_arrays[f"{prefix}_source_sample_offset"] = np.array(
                        record.source_sample_offset, dtype=np.int64
                    )
                    for name in (
                        "left",
                        "right",
                        "weight",
                        "source_event_indices",
                        "execution_event_indices",
                    ):
                        plan_arrays[f"{prefix}_retime_{name}"] = getattr(
                            record.result.time_map, name
                        )
                        plan_arrays[f"{prefix}_normalization_{name}"] = getattr(
                            normalized, name
                        )
                    template = record.result.template
                else:
                    entry.update(
                        operator=record.operator,
                        seed=record.seed,
                        candidate_id=record.candidate_id,
                        geometry_family_id=record.geometry_family_id,
                        variant_ordinal=record.variant_ordinal,
                        attempted=record.attempted,
                        rejected=dict(record.rejected),
                    )
                    template = record.template
                plan_arrays[f"{prefix}_positions"] = template.positions.cpu().numpy()
                plan_arrays[f"{prefix}_dt"] = template.dt.cpu().numpy()
                call_variants.append(entry)
        if plan_arrays:
            np.savez_compressed(output / "plan_variants.npz", **plan_arrays)
        bridge = target._active_task_program_bridge
        trace = bridge.execution_trace
        count = int(target._traj_steps[0].item())
        if len(trace) != count or any(
            row["step"] != index for index, row in enumerate(trace)
        ):
            raise ValueError(
                "consumed-action trace does not align with the recorded rollout"
            )
        retime_maps = (
            {}
            if not isinstance(transform, TaskProgramMappedRetimeFactory)
            else {
                record.call_index: record.result.time_map
                for record in transform.records
            }
        )
        trace_arrays = build_real2sim_execution_trace(
            trace, factory.motions, retime_maps=retime_maps
        )
        map_evidence = [
            {
                "call_index": index,
                "source_sample_offset": motion.source_sample_offset,
                "all_plan_samples_observed": True,
                "plan_samples": len(trace_arrays[f"call_{index}_left"]),
            }
            for index, motion in sorted(factory.motions.items())
        ]
        np.savez_compressed(output / "execution_trace.npz", **trace_arrays)
        report["action_to_plan_trace"] = {
            "steps": len(trace),
            "calls": map_evidence,
            "complete": len(map_evidence) == 3
            and all(item["all_plan_samples_observed"] for item in map_evidence),
        }
        source_bindings = [
            {
                "call_index": index,
                "source_id": motion.source_id,
                "source_revision": motion.source_revision,
                "unit_id": motion.unit_id,
                "origin": motion.origin,
                "authority": motion.authority,
                "collision_evidence": motion.collision_evidence,
                "geometry_compatibility": "unverified",
                "qpos_semantics": motion.qpos_semantics,
                "source_command_semantics": motion.source_command_semantics,
                "pose_origin": motion.pose_origin,
                "fk_position_error_m": motion.fk_position_error_m,
                "fk_rotation_error_rad": motion.fk_rotation_error_rad,
                "gripper_mode": motion.gripper_mode,
                "source_sample_offset": motion.source_sample_offset,
                "ik_smoothing_iterations": motion.ik_smoothing_iterations,
                "prototype_sha256": hashlib.sha256(
                    (
                        reference / "prototypes" / str(index) / "prototype.json"
                    ).read_bytes()
                ).hexdigest(),
            }
            for index, motion in sorted(factory.motions.items())
        ]
        report.update(
            {
                "program_fingerprint": factory.program_fingerprint,
                "task_verified": bool(
                    result.accepted
                    and post_checks
                    and all(check["status"] == "passed" for check in post_checks)
                    and report["action_to_plan_trace"]["complete"]
                ),
                "scene": result.to_metadata(),
                "execution": (
                    None if result.execution is None else result.execution.to_metadata()
                ),
                "post_checks": post_checks,
                "call_variants": call_variants,
                "source_bindings": source_bindings,
                "sampled_plan_collisions": (
                    []
                    if collision_validator is None
                    else [
                        {"unit_id": unit, **asdict(check)}
                        for unit, check in collision_validator.reports
                    ]
                ),
                "plan_mapping_scope": "submitted actions to plan samples; runtime holds are explicitly unassociated",
                "rollout_streams": {
                    "qpos": "simulated_measured_state",
                    "qvel": "simulated_measured_state",
                    "object_pose": "simulated_measured_state",
                    "tcp_pose": "fk_derived_from_simulated_qpos",
                    "actions": "actual_submitted_commands",
                },
            }
        )
        if collision_validator is not None and collision_validator.reports:
            expected_checks = 3 if transform is None else 6
            if len(collision_validator.reports) == expected_checks and all(
                check.accepted for _, check in collision_validator.reports
            ):
                report["collision_check"] = "sampled_plan_geometry"
                report["geometry_compatibility"] = "asset_identity_matched"
        if target._traj_buffer is not None and count > 0:
            buffer = target._traj_buffer
            qpos = buffer["states", "robot", "qpos"][0, : count + 1]
            arm_ids = target.robot.get_joint_ids(name="arm")
            tcp = target.robot.compute_batch_fk(
                qpos=qpos[None, :, arm_ids], name="arm", to_matrix=True
            )[0]
            np.savez_compressed(
                output / "actual_rollout.npz",
                object_pose=buffer["states", "rigid_objects", "cube", "pose"][
                    0, : count + 1
                ]
                .cpu()
                .numpy(),
                qpos=qpos.cpu().numpy(),
                qvel=buffer["states", "robot", "qvel"][0, : count + 1].cpu().numpy(),
                tcp_pose=tcp.cpu().numpy(),
                actions=buffer["actions"][0, :count].cpu().numpy(),
                timestamps=np.arange(count + 1) * target.step_dt,
            )
        if collision_validator is not None:
            if target._traj_buffer is None or result.execution is None:
                raise ValueError("observed collision checking needs a recorded episode")
            events = []
            names = {
                "pick": "pick_up",
                "transport": "move_held_object",
                "place": "place",
            }
            for segment in result.execution.to_metadata()["segments"]:
                if segment["name"] not in names:
                    raise ValueError(
                        "observed collision timeline contains an unknown skill"
                    )
                for event in segment["metadata"]["runtime"]["events"]:
                    if event["kind"] == "trajectory_segment_entered":
                        sample = round(float(event["timestamp"]) / target.step_dt)
                        events.append(
                            (sample, names[segment["name"]], event["segment_name"])
                        )
            events.sort()
            if (
                not events
                or events[0][0] != 0
                or any(sample >= count + 1 for sample, _, _ in events)
                or any(events[i][0] >= events[i + 1][0] for i in range(len(events) - 1))
            ):
                raise ValueError(
                    "observed collision timeline does not map to control samples"
                )
            phases = tuple(
                collision_validator.phase_policy(
                    skill,
                    phase,
                    sample,
                    events[index + 1][0] if index + 1 < len(events) else count + 1,
                )
                for index, (sample, skill, phase) in enumerate(events)
            )
            states = target._traj_buffer["states"]
            object_values = (
                states["rigid_objects", "cube", "pose"][0, : count + 1].cpu().numpy()
            )
            object_matrices = np.repeat(np.eye(4)[None], count + 1, axis=0)
            object_matrices[:, :3, :3] = Rotation.from_quat(
                object_values[:, 3:7]
            ).as_matrix()
            object_matrices[:, :3, 3] = object_values[:, :3]
            observed_collision = collision_validator.validate_observed(
                states["robot", "qpos"][0, : count + 1],
                object_poses=object_matrices,
                target_pose=platform.get_local_pose(to_matrix=True)[0]
                .detach()
                .cpu()
                .numpy(),
                phases=phases,
            )
            np.savez_compressed(
                output / "observed_geometry.npz",
                object_pose=object_values,
                target_pose=platform.get_local_pose(to_matrix=True)[0]
                .detach()
                .cpu()
                .numpy(),
                timestamps=np.arange(count + 1, dtype=np.float64) * target.step_dt,
            )
            report["sampled_observed_collision"] = asdict(observed_collision)
            if not observed_collision.accepted:
                raise ValueError(
                    f"observed collision at sample {observed_collision.first_failure_sample}: "
                    f"{observed_collision.first_failure_pair}: {observed_collision.reason}"
                )
            if report["collision_check"] == "sampled_plan_geometry":
                report["collision_check"] = "sampled_plan_and_observed_geometry"
        if options.save_lerobot:
            if (
                not report["task_verified"]
                or report["collision_check"] != "sampled_plan_and_observed_geometry"
            ):
                raise ValueError("only a checked successful episode can be submitted")
            if int(target.rollout_steps[0]) != count:
                raise ValueError("rollout and physical trajectory lengths differ")
            observations = {name: ("robot", name) for name in ("qpos", "qvel", "qf")}
            cfg_recorder = DatasetFunctorCfg(
                func=LeRobotRecorder,
                params={
                    "save_path": str(output / "dataset"),
                    "robot_meta": {"robot_type": "UR5"},
                    "extra": {
                        "scene_type": "tabletop",
                        "task_description": "real2sim_synthetic",
                        "data_type": "sim",
                    },
                    "use_videos": False,
                },
            )
            recorder = LeRobotRecorder(cfg_recorder, target)
            sink = Real2SimLeRobotEpisodeSink(
                recorder,
                observation_fields=observations,
                action_representation="joint_position",
                required_check_ids=(
                    "geometry",
                    "sampled_collision",
                    "observed_collision",
                    "task",
                ),
            )

            def artifact(path):
                return {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }

            bundles = {}
            calls = []
            selected = {entry["call_index"]: entry for entry in call_variants}
            for index, motion in sorted(factory.motions.items()):
                root = reference / "prototypes" / str(index)
                manifest = root / "prototype.json"
                bundles[str(index)] = {
                    "prototype_manifest": artifact(manifest),
                    "normalized_arrays": artifact(root / "normalized.npz"),
                    "source_manifest": artifact(root / "source" / "trajectory.json"),
                    "source_arrays": artifact(root / "source" / "trajectory.npz"),
                }
                calls.append(
                    {
                        "call_index": index,
                        "source_bundle": str(index),
                        "geometry_family_id": variant.variant_id,
                        "source_sample_offset": motion.source_sample_offset,
                        "operator": selected.get(index, {}).get("operator", "nominal"),
                        "prototype_sha256": bundles[str(index)]["prototype_manifest"][
                            "sha256"
                        ],
                        "recipe": selected.get(index),
                    }
                )
            candidate_payload = json.dumps(
                {
                    "program": factory.program_fingerprint,
                    "layout": variant.variant_id,
                    "recipe": options.recipe,
                    "seed": options.seed,
                    "prototypes": [
                        calls[i]["prototype_sha256"] for i in range(len(calls))
                    ],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            candidate_id = "candidate:" + hashlib.sha256(candidate_payload).hexdigest()
            first_source = prototypes[0].unit.source
            identity = CandidateIdentity(
                scene_case_id=variant.variant_id,
                initial_state_id=result.preparation.initial_state_id,
                candidate_id=candidate_id,
                geometry_family_id=variant.variant_id,
                source_id=first_source.source_id,
                source_revision=first_source.source_revision,
                template_id="pick_move_place_episode",
            )
            validation = ValidationResult(
                (
                    ValidationCheck(
                        "geometry",
                        "passed",
                        "source and physical asset identities match",
                    ),
                    ValidationCheck(
                        "sampled_collision",
                        "passed",
                        "sampled plan geometry with gripper contact projection",
                    ),
                    ValidationCheck(
                        "observed_collision",
                        "passed",
                        "sampled measured robot and object geometry",
                    ),
                    ValidationCheck("task", "passed", "verified measured task result"),
                )
            )
            episode = capture_real2sim_episode(
                target,
                identity=identity,
                validation=validation,
                episode_id="episode:" + candidate_id,
                commit_id="commit:" + candidate_id,
                observation_fields=observations,
                action_representation="joint_position",
                metadata={
                    "origin": "synthetic_reference",
                    "motion_limits": "diagnostic_gripper_acceleration_assumption",
                    "real2sim_observed_geometry": artifact(
                        output / "observed_geometry.npz"
                    ),
                    "real2sim_source_artifacts": bundles,
                    "real2sim_lineage": {
                        "scene_revision": variant.parent_scene_id,
                        "program_fingerprint": factory.program_fingerprint,
                        "trace": artifact(output / "execution_trace.npz"),
                        "calls": calls,
                    },
                },
            )
            receipt = submit_and_discard_real2sim_episode(
                target, sink, episode, submission_id=0
            )
            report["receipt"] = asdict(receipt)
            report["dataset_path"] = str(recorder.dataset_full_path)
            report["dataset_committed"] = receipt.confirmed
            if not receipt.confirmed:
                raise RuntimeError(f"LeRobot submission failed: {receipt.error}")
    except Exception as exc:
        report["task_verified"] = False
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if recorder is not None:
            recorder.finalize()
        (output / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        env.unwrapped.close(exit_process=False)
        SimulationManager.flush_cleanup_queue()
    if not report["task_verified"]:
        raise RuntimeError("layout did not pass all diagnostic checks; see report.json")


if __name__ == "__main__":
    main()
