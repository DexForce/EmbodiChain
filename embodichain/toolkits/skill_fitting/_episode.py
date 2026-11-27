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

"""Compose independently fitted annotations on one immutable recording clock."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from ._pipeline import FitInputs

__all__ = []

FrameClock = Callable[[int, int], tuple[np.ndarray, np.ndarray]]


def fit_episode(
    inputs: FitInputs,
    document: dict[str, Any],
    clock: FrameClock,
    *,
    progress: Callable[[str], None] = lambda _: None,
    check_current: Callable[[], None] = lambda: None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit every supported saved annotation and retain original samples in gaps.

    Each native fit reads the original recording, never another fit's output.
    Only its core is copied into the combined result. Annotation transitions
    remain part of their Atomic Skill; uncovered samples become DemoSkill spans.
    Every sample belongs to exactly one half-open timeline span. Unsupported
    annotated skills fail explicitly rather than losing their authored meaning.
    """
    from embodichain.data_pipeline.skill_fitting import (
        file_digest,
        load_joint_recording,
        map_video_interval,
    )
    from embodichain.toolkits.trajectory_annotator._models import (
        validate_skill_intervals,
    )
    from embodichain.toolkits.trajectory_annotator._skills import (
        RECONSTRUCTION_SKILLS,
        _canonical_skill_id,
    )
    from ._pipeline import fit_recording
    from ._robot import RobotProfile

    began = time.monotonic()
    inputs.check()
    count = document["source"]["frame_count"]
    segments = validate_skill_intervals(document["segments"], count)
    skills = [_canonical_skill_id(s["label"]) for s in segments]
    for segment, skill in zip(segments, skills):
        if skill not in RECONSTRUCTION_SKILLS | {"execute_demo_skill"}:
            raise ValueError(
                f"标注 {segment['label']} [{segment['start_frame']}, "
                f"{segment['end_frame']}] 尚不支持拟合；请明确修改标注后重试。"
            )
    if inputs.phase_hints and sum(s in RECONSTRUCTION_SKILLS for s in skills) > 1:
        raise ValueError(
            "全轨迹含多个原子技能时不能共用一份相对阶段提示，请清空阶段提示。"
        )
    paths = [
        Path(inputs.joints).expanduser().resolve(),
        Path(inputs.robot_profile).expanduser().resolve(),
    ]
    digests = {str(p): file_digest(p) for p in paths}
    profile = RobotProfile.load(paths[1])
    models = {part: file_digest(s["urdf_path"]) for part, s in profile.solvers.items()}
    recording = load_joint_recording(
        paths[0],
        joint_names=profile.joint_names,
        semantics=inputs.qpos_semantics,
        qpos_key=inputs.qpos_key,
        timestamps_key=inputs.timestamps_key,
        sample_dt=inputs.joint_dt,
        env_index=inputs.env_index,
    )
    arrays = {"timestamps": recording.timestamps.copy()}
    for name in ("original", "native", "patched", "proposed"):
        arrays[name + "_qpos"] = recording.qpos.copy()
    provenance = dict(
        source_identity=document["source"]["identity"],
        source_timing=document["source"]["timing"],
        annotation_revision=document["revision"],
        annotation_sha256=hashlib.sha256(
            json.dumps(document, sort_keys=True).encode()
        ).hexdigest(),
        annotation_digest_kind="canonical_document_json",
    )
    timeline, fits, failures = [], [], []
    cursor = 0

    def gap(start: int, stop: int) -> None:
        if stop > start:
            timeline.append(
                dict(
                    kind="demo_skill",
                    skill_id="execute_demo_skill",
                    label="DemoSkill",
                    start=start,
                    stop=stop,
                    source="unannotated",
                )
            )

    for number, (segment, skill) in enumerate(zip(segments, skills), 1):
        check_current()
        ft, fd = clock(segment["start_frame"], segment["end_frame"])
        start, stop = map_video_interval(
            ft,
            fd,
            recording.timestamps,
            start_frame=0,
            end_frame=len(ft) - 1,
            time_offset=inputs.time_offset,
        )
        if start < cursor or stop <= start:
            raise ValueError(
                "标注映射到关节时钟后为空或重叠，请检查视频与关节时间对齐。"
            )
        gap(cursor, start)
        span = dict(
            kind="demo_skill" if skill == "execute_demo_skill" else "atomic_skill",
            skill_id=skill,
            label=segment["label"],
            start=start,
            stop=stop,
            source="annotation",
            annotation=segment,
        )
        if skill != "execute_demo_skill":
            core = segment.get("core_interval", segment)
            ft, fd = clock(core["start_frame"], core["end_frame"])
            progress(
                f"[{number}/{len(segments)}] {segment['label']} · 帧 {core['start_frame']}–{core['end_frame']}"
            )
            fitted, report = fit_recording(
                inputs,
                skill_id=skill,
                frame_times=ft,
                frame_durations=fd,
                start_frame=core["start_frame"],
                end_frame=core["end_frame"],
                provenance={
                    **provenance,
                    "annotation_id": segment["id"],
                    "annotation_segment": segment,
                    "annotation_video_interval": {
                        k: segment[k] for k in ("start_frame", "end_frame")
                    },
                },
                progress=lambda message, n=number: progress(
                    f"[{n}/{len(segments)}] {message}"
                ),
            )
            if not np.array_equal(
                fitted["original_qpos"], recording.qpos
            ) or not np.array_equal(fitted["timestamps"], recording.timestamps):
                raise ValueError("拟合期间原始关节记录发生变化。")
            left, right = report["start"], report["stop"]
            if not start <= left < right <= stop:
                raise ValueError("拟合核心超出了标注的关节区间。")
            for name in ("native", "patched", "proposed"):
                arrays[name + "_qpos"][left:right] = fitted[name + "_qpos"][left:right]
            span.update(
                fit_index=len(fits),
                fit_start=left,
                fit_stop=right,
                accepted=report["accepted"],
            )
            fits.append(report)
            failures.extend(
                f"{segment['id']}: {reason}" for reason in report["failures"]
            )
        timeline.append(span)
        cursor = stop
    gap(cursor, len(recording.qpos))
    check_current()
    if any(file_digest(p) != d for p, d in digests.items()) or any(
        file_digest(profile.solvers[p]["urdf_path"]) != d for p, d in models.items()
    ):
        raise ValueError("全轨迹拟合期间关节、机器人配置或 URDF 发生变化。")
    report = dict(
        **provenance,
        schema_version=1,
        scope="episode",
        skill_id="mixed_trajectory",
        accepted=not failures and all(f["accepted"] for f in fits),
        failures=failures,
        start=0,
        stop=len(recording.qpos),
        joint_names=list(profile.joint_names),
        phases=[
            dict(name=f"{s['label']}_{i}", start=s["start"], stop=s["stop"])
            for i, s in enumerate(timeline)
        ],
        parameters={},
        timeline=timeline,
        fits=fits,
        atomic_skill_count=len(fits),
        demo_skill_count=sum(s["kind"] == "demo_skill" for s in timeline),
        native_joint_rmse=float(
            np.sqrt(np.mean((arrays["native_qpos"] - recording.qpos) ** 2))
        ),
        tcp_errors={
            f"{i}.{part}": error
            for i, f in enumerate(fits)
            for part, error in f["tcp_errors"].items()
        },
        selected_participants={
            f"{i}.{role}": p
            for i, f in enumerate(fits)
            for role, p in f["selected_participants"].items()
        },
        robot_profile=profile.metadata(),
        robot_profile_sha256=digests[str(paths[1])],
        robot_model_sha256=models,
        source_joint_sha256=digests[str(paths[0])],
        time_offset=inputs.time_offset,
        qpos_semantics=inputs.qpos_semantics,
        output_semantics="design_reference",
        physical_success="unknown",
        video_interval=dict(start_frame=0, end_frame=count - 1, interval="closed"),
        elapsed_seconds=time.monotonic() - began,
    )
    return arrays, report
