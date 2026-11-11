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

"""Shared offline fitting orchestration for CLI and annotation hosts."""

from __future__ import annotations

from collections.abc import Callable
import math
from pathlib import Path
from typing import Any

import numpy as np

from embodichain.utils import configclass

__all__ = []


@configclass
class FitInputs:
    """Explicit recording/profile inputs; importing this type needs no simulator."""

    joints: str = ""
    robot_profile: str = ""
    qpos_semantics: str = ""
    time_offset: float | None = None
    qpos_key: str = "qpos"
    timestamps_key: str = "timestamps"
    joint_dt: float | None = None
    env_index: int = 0
    device: str = "cpu"
    phase_hints: list[dict[str, Any]] | None = None
    fit_config: dict[str, Any] = {}

    def check(self) -> None:
        """Validate the closed request before reading files or creating solvers."""
        for key in ("joints", "robot_profile", "qpos_key", "timestamps_key", "device"):
            value = getattr(self, key)
            if not isinstance(value, str) or not value.strip() or len(value) > 4096:
                raise ValueError(f"{key} 需要非空字符串。")
        if self.qpos_semantics not in {
            "measured_state",
            "actual_submission",
            "design_reference",
        }:
            raise ValueError("请声明关节数据的语义。")
        for key in ("time_offset", "joint_dt"):
            value = getattr(self, key)
            if key == "joint_dt" and value is None:
                continue
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{key} 需要有限数值。")
            if key == "joint_dt" and value <= 0:
                raise ValueError("joint_dt 必须为正数。")
        if type(self.env_index) is not int or self.env_index < 0:
            raise ValueError("env_index 需要非负整数。")
        if not isinstance(self.fit_config, dict):
            raise ValueError("fit_config 需要 JSON 对象。")
        if self.phase_hints is not None:
            if not isinstance(self.phase_hints, list) or not self.phase_hints:
                raise ValueError("phase_hints 需要非空 JSON 数组，或 null。")
            for phase in self.phase_hints:
                if (
                    not isinstance(phase, dict)
                    or phase.keys() != {"name", "start", "stop"}
                    or not isinstance(phase["name"], str)
                    or type(phase["start"]) is not int
                    or type(phase["stop"]) is not int
                ):
                    raise ValueError("阶段提示需要 name、start、stop 字段。")


def fit_recording(
    inputs: FitInputs,
    *,
    skill_id: str,
    frame_times: np.ndarray,
    frame_durations: np.ndarray,
    start_frame: int,
    end_frame: int,
    provenance: dict[str, Any],
    progress: Callable[[str], None] = lambda _: None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Reuse the native fitter with a host-owned physical video clock.

    The provided clock may contain only the two selected endpoints. The actual
    inclusive frame indices are preserved separately in the result provenance.
    Native solvers, FK geometry and all numeric allocations belong to the caller's
    single fitting thread. Frame decoding remains with the annotation worker.
    """
    inputs.check()
    from embodichain.data_pipeline.skill_fitting import (
        file_digest,
        load_joint_recording,
        map_video_interval,
    )
    from embodichain.lab.sim.atomic_skills import (
        AtomicSkillEngine,
        ControlPartCommandProfile,
    )
    from embodichain.lab.sim.atomic_skills.fitting import SkillFitCfg, fit_skill_segment
    from embodichain.lab.sim.motion.motion_generator import (
        MotionGenerator,
        MotionGenCfg,
    )
    from embodichain.lab.sim.motion.planners import (
        TrapezoidalPlanner,
        TrapezoidalPlannerCfg,
    )
    from ._robot import RobotPort, RobotProfile
    import torch

    if set(inputs.fit_config) - set(SkillFitCfg.__dataclass_fields__):
        raise ValueError("Unknown fit configuration fields.")
    cfg = SkillFitCfg(**inputs.fit_config)
    cfg.validate()
    paths = [
        Path(inputs.joints).expanduser().resolve(),
        Path(inputs.robot_profile).expanduser().resolve(),
    ]
    progress("正在读取关节记录与机器人标定…")
    digests = {str(path): file_digest(path) for path in paths}
    profile = RobotProfile.load(paths[1])
    models = {
        part: file_digest(solver["urdf_path"])
        for part, solver in profile.solvers.items()
    }
    recording = load_joint_recording(
        paths[0],
        joint_names=profile.joint_names,
        semantics=inputs.qpos_semantics,
        qpos_key=inputs.qpos_key,
        timestamps_key=inputs.timestamps_key,
        sample_dt=inputs.joint_dt,
        env_index=inputs.env_index,
    )
    start, stop = map_video_interval(
        frame_times,
        frame_durations,
        recording.timestamps,
        start_frame=0,
        end_frame=len(frame_times) - 1,
        time_offset=inputs.time_offset,
    )
    if not 12 <= stop - start <= cfg.max_samples:
        raise ValueError(
            f"选区映射为 {stop - start} 个关节采样，需要 12–{cfg.max_samples} 个。"
        )
    progress("正在辨识参数、调用原生技能并校验连续回填…")
    robot = RobotPort(profile, recording.qpos[start], inputs.device)
    planner_cfg = TrapezoidalPlannerCfg(robot_uid="offline_skill_fit")
    planner = TrapezoidalPlanner(planner_cfg, robot=robot)
    motion = MotionGenerator(MotionGenCfg(planner_cfg=planner_cfg), planner=planner)
    commands = {
        part: ControlPartCommandProfile.joint_positions(
            **{
                key: torch.tensor(value, device=robot.device, dtype=torch.float32)
                for key, value in command.items()
            }
        )
        for part, command in profile.hand_commands.items()
    }
    engine = AtomicSkillEngine(motion, control_profiles=commands)
    arrays, report = fit_skill_segment(
        recording,
        engine,
        skill_id=skill_id,
        endpoints=profile.participants,
        start=start,
        stop=stop,
        acceleration_limits=np.asarray(profile.acceleration_limits),
        cfg=cfg,
        phase_hints=inputs.phase_hints,
    )
    progress("正在生成关节曲线和 FK 对比…")
    grid = np.unique(
        np.r_[
            np.linspace(
                0, len(recording.qpos) - 1, min(len(recording.qpos), 2000)
            ).astype(int),
            start,
            stop - 1,
        ]
    )
    arrays["geometry_indices"] = grid
    for name in ("original", "native", "patched", "proposed"):
        arrays[f"{name}_links"], arrays[f"{name}_tcp"] = robot.geometry(
            arrays[f"{name}_qpos"][grid]
        )
    report.update(
        **provenance,
        robot_profile=profile.metadata(),
        robot_profile_sha256=digests[str(paths[1])],
        robot_model_sha256=models,
        source_joint_sha256=digests[str(paths[0])],
        video_interval={
            "start_frame": start_frame,
            "end_frame": end_frame,
            "interval": "closed",
        },
        time_offset=inputs.time_offset,
    )
    if any(file_digest(path) != digest for path, digest in digests.items()) or any(
        file_digest(profile.solvers[part]["urdf_path"]) != digest
        for part, digest in models.items()
    ):
        raise ValueError("关节记录、机器人配置或 URDF 在拟合期间发生变化。")
    return arrays, report
