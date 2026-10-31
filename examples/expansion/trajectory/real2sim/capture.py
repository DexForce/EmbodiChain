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

"""Capture a verified synthetic UR5 demonstration as Real2Sim prototypes.

This sample records a simulator-generated reference. It is not real-robot data
and does not certify spatial augmentation or downstream policy performance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from embodichain.data_pipeline.real2sim import (
    JointSpec,
    PhaseAnnotation,
    Real2SimTrajectory,
    load_prototype,
    prepare_prototype,
    save_trajectory,
)
from embodichain.lab.gym.envs.demo import execute_demo_episode
from embodichain.lab.gym.utils.gym_utils import build_env_cfg_from_args
from embodichain.lab.scripts.run_task import (
    _create_parser,
    discover_task_packages,
    execute_init_hooks,
)
from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.atomic_actions import JointPositionTarget
from embodichain.lab.sim.motion.expansion.real2sim_collision import (
    BoxCollisionBody,
    SampledRobotCollisionChecker,
)
from embodichain.lab.task_program.integrations.nominal import (
    BoundTaskProgramNominalSources,
)

__all__: list[str] = []

_TASK = Path(__file__).with_name("task.ur5.yaml")
_JOINT_KINDS = {
    **{f"joint{index}": "revolute" for index in range(1, 7)},
    "gripper_finger1_joint_1": "prismatic",
    "gripper_finger2_joint_1": "prismatic",
}
_PHASES = {
    "pick_up": {
        "approach": (
            "free",
            "object",
            ("joint_residual", "task_space_residual", "retime"),
        ),
        "close": ("contact", "object", ()),
        "lift": ("hold", "gravity", ()),
    },
    "move_held_object": {
        "transport": (
            "free",
            "target",
            ("joint_residual", "task_space_residual", "retime"),
        )
    },
    "place": {
        "approach": (
            "free",
            "target",
            ("joint_residual", "task_space_residual", "retime"),
        ),
        "release": ("contact", "target", ()),
        "retract": (
            "free",
            "target",
            ("joint_residual", "task_space_residual", "retime"),
        ),
    },
}


class _CaptureFactory:
    """Capture native plan evidence without replacing or modifying a plan."""

    def __init__(
        self,
        *,
        robot_geometry_sha256: str,
        object_geometry_sha256: str,
        target_geometry_sha256: str,
    ) -> None:
        self.records: dict[
            int, tuple[Real2SimTrajectory, tuple[JointSpec, ...], float]
        ] = {}
        self.robot_geometry_sha256 = robot_geometry_sha256
        self.object_geometry_sha256 = object_geometry_sha256
        self.target_geometry_sha256 = target_geometry_sha256

    def create_plan_transform(self, request, *, engine):
        """Build one read-only capture callback for a grounded call."""

        def capture(resolved, context, plan):
            if request.analysis_call_index in self.records:
                raise ValueError(
                    "synthetic capture encountered a repeated call occurrence"
                )
            if (
                context.batch_size != 1
                or plan.joint_trajectory is None
                or not bool(plan.plan_success.all())
            ):
                raise ValueError("synthetic capture requires one successful joint plan")
            skill = resolved.skill_id
            if skill not in _PHASES:
                raise ValueError(f"unsupported synthetic skill {skill!r}")
            trajectory = plan.joint_trajectory
            dt = trajectory.dt[0].detach().cpu().numpy().astype(np.float64)
            period = context.require_control_dt()
            if dt[0] != 0 or not np.allclose(dt[1:], period, atol=1e-6, rtol=0):
                raise ValueError("synthetic reference is not on the Gym control grid")
            names = tuple(engine.robot.joint_names)
            if set(names) != set(_JOINT_KINDS):
                raise ValueError(
                    "synthetic sample requires the declared UR5/DH joint layout"
                )
            mimic_ids = tuple(engine.robot.mimic_ids)
            mimic = dict(
                zip(
                    mimic_ids,
                    zip(
                        engine.robot.mimic_parents,
                        engine.robot.mimic_multipliers,
                        engine.robot.mimic_offsets,
                    ),
                )
            )
            joints = tuple(
                JointSpec(
                    name,
                    _JOINT_KINDS[name],
                    mimic_driver=(names[mimic[index][0]] if index in mimic else None),
                    mimic_multiplier=(
                        float(mimic[index][1]) if index in mimic else 1.0
                    ),
                    mimic_offset=(float(mimic[index][2]) if index in mimic else 0.0),
                )
                for index, name in enumerate(names)
            )
            independent = [index for index in range(len(names)) if index not in mimic]
            endpoint = resolved.binding.endpoint("primary", "motion").require_target(
                JointPositionTarget
            )
            poses = (
                engine.robot.compute_batch_fk(
                    qpos=trajectory.positions[:, :, list(endpoint.joint_ids)],
                    name=endpoint.control_part,
                    to_matrix=True,
                )[0]
                .detach()
                .cpu()
                .numpy()
            )
            xyzw = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
            eef = np.concatenate((poses[:, :3, 3], xyzw), axis=1)
            object_pose = context.scene.entities["cube"].pose.detach().cpu().numpy()
            if object_pose.ndim == 3:
                object_pose = object_pose[0]
            target_pose = None
            if skill != "pick_up":
                target_pose = (
                    context.scene.entities["platform"].pose.detach().cpu().numpy()
                )
                if target_pose.ndim == 3:
                    target_pose = target_pose[0]
            positions = trajectory.positions[0, :, independent].detach().cpu().numpy()
            initial = context.robot.qpos[0, independent].detach().cpu().numpy()
            phases = []
            for segment in plan.segments:
                if segment.name not in _PHASES[skill]:
                    raise ValueError(f"unexpected {skill} phase {segment.name!r}")
                kind, reference, permissions = _PHASES[skill][segment.name]
                phases.append(
                    PhaseAnnotation(
                        segment.name,
                        segment.start,
                        segment.stop,
                        kind,
                        reference,
                        permissions,
                    )
                )
            unit = Real2SimTrajectory(
                source_id="synthetic_ur5_pick_move_place",
                source_revision="3",
                unit_id=f"call_{request.analysis_call_index}_{skill}",
                skill_id=skill,
                origin="synthetic_reference",
                joint_names=tuple(names[index] for index in independent),
                joint_units=tuple(
                    "rad" if _JOINT_KINDS[names[index]] == "revolute" else "m"
                    for index in independent
                ),
                qpos=positions,
                initial_observed_qpos=initial,
                timestamps=np.cumsum(dt),
                time_unit="s",
                phases=tuple(phases),
                eef_poses=eef,
                pose_length_unit="m",
                quaternion_order="xyzw",
                world_from_pose_frame=np.eye(4),
                observed_link_to_tcp=np.eye(4),
                object_pose=object_pose,
                target_pose=target_pose,
                commands=positions,
                command_representation="joint_position_reference",
                command_alignment="sample_aligned_reference",
                calibration_revision="synthetic_ur5_dh_pgi_140_80_v1",
                annotation_revision="synthetic_skill_phases_v1",
                pose_origin="fk_derived",
                robot_geometry_sha256=self.robot_geometry_sha256,
                object_geometry_sha256=self.object_geometry_sha256,
                target_geometry_sha256=(
                    self.target_geometry_sha256 if target_pose is not None else None
                ),
            )
            self.records[request.analysis_call_index] = (unit, joints, period)
            return plan

        return capture


def main() -> None:
    """Run the reference episode and publish its source data only on success."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gym-config", type=Path, default=_TASK)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    options = parser.parse_args()
    output = options.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    args = _create_parser().parse_args(
        [
            "--gym-config",
            str(options.gym_config),
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
    env = gymnasium.make(id=gym_config["id"], cfg=cfg, **action_config)
    try:
        env.reset(seed=options.seed, options={"save_data": False})
        target = env.unwrapped
        program = target.compile_task_program(target.cfg.task_program)
        fingerprint = BoundTaskProgramNominalSources(program, {}).program_fingerprint
        object_geometry = BoxCollisionBody.from_rigid_object(
            target.sim.get_rigid_object("cube")
        )
        target_geometry = BoxCollisionBody.from_rigid_object(
            target.sim.get_rigid_object("platform")
        )
        collision_model = SampledRobotCollisionChecker(
            target.robot.cfg.fpath,
            joint_names=target.robot.joint_names,
            bodies=(object_geometry, target_geometry),
            world_from_base=target.robot.get_local_pose(to_matrix=True)[0]
            .detach()
            .cpu()
            .numpy(),
        )
        capture = _CaptureFactory(
            robot_geometry_sha256=collision_model.geometry_sha256,
            object_geometry_sha256=object_geometry.geometry_sha256,
            target_geometry_sha256=target_geometry.geometry_sha256,
        )
        result = execute_demo_episode(
            env, task_program=program, plan_transform_factory=capture
        )
        report = {
            "origin": "synthetic_reference",
            "baseline_verified": bool(result.completed and result.all_success),
            "source_artifacts_complete": False,
            "terminal_reason": result.terminal_reason,
            "length": result.length,
            "program_fingerprint": fingerprint,
            "captured_calls": sorted(capture.records),
        }
        (output / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        if not report["baseline_verified"] or set(capture.records) != {0, 1, 2}:
            raise RuntimeError(
                "synthetic baseline did not pass every verified call and validator"
            )
        prepared = []
        for index, (unit, joints, period) in sorted(capture.records.items()):
            save_trajectory(unit, output / "sources" / str(index))
            manifest = prepare_prototype(
                unit,
                joints=joints,
                control_dt=period,
                directory=output / "prototypes" / str(index),
            )
            prepared.append(load_prototype(manifest))
        count = int(target._traj_steps[0].item())
        buffer = target._traj_buffer
        if buffer is None or count <= 0 or buffer.shape[1] <= count:
            raise RuntimeError("actual rollout does not retain its terminal state")
        states = buffer["states"]
        measured_qpos = states["robot", "qpos"][0, : count + 1]
        arm_ids = target.robot.get_joint_ids(name="arm")
        measured_tcp = (
            target.robot.compute_batch_fk(
                qpos=measured_qpos[:, arm_ids].unsqueeze(0), name="arm", to_matrix=True
            )[0]
            .detach()
            .cpu()
            .numpy()
        )
        actual = {
            "qpos": measured_qpos.cpu().numpy(),
            "qvel": states["robot", "qvel"][0, : count + 1].cpu().numpy(),
            "object_pose": states["rigid_objects", "cube", "pose"][0, : count + 1]
            .cpu()
            .numpy(),
            "tcp_pose": measured_tcp,
            "actions": buffer["actions"][0, :count].cpu().numpy(),
            "timestamps": np.arange(count + 1) * target.step_dt,
        }
        np.savez_compressed(output / "actual_rollout.npz", **actual)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for prototype in prepared:
            poses = prototype.unit.world_tcp
            if poses is None:
                raise RuntimeError("synthetic FK pose evidence is missing")
            axes[0].plot(
                poses[:, 0, 3], poses[:, 1, 3], label=prototype.unit.source.skill_id
            )
            axes[1].plot(
                poses[:, 0, 3], poses[:, 2, 3], label=prototype.unit.source.skill_id
            )
        axes[0].plot(
            measured_tcp[:, 0, 3],
            measured_tcp[:, 1, 3],
            "k--",
            alpha=0.6,
            label="actual FK",
        )
        axes[1].plot(
            measured_tcp[:, 0, 3],
            measured_tcp[:, 2, 3],
            "k--",
            alpha=0.6,
            label="actual FK",
        )
        for axis, vertical in zip(axes, ("Y", "Z")):
            axis.set_xlabel("X (m)")
            axis.set_ylabel(f"{vertical} (m)")
            axis.grid(True, alpha=0.3)
            axis.legend()
        figure.tight_layout()
        figure.savefig(output / "trajectory_comparison.png", dpi=160)
        plt.close(figure)
        report["source_artifacts_complete"] = True
        (output / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        env.reset(options={"save_data": False})
    finally:
        env.unwrapped.close(exit_process=False)
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
