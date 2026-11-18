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

"""One bounded offline fit and a decimated comparison for the annotation host."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import threading
from typing import Any, BinaryIO
from uuid import uuid4

import numpy as np

from embodichain.toolkits.skill_fitting._pipeline import FitInputs
from ._frames import FramePreview
from ._models import ConflictError
from ._skills import RECONSTRUCTION_SKILLS, _canonical_skill_id
from ._viewer import TrajectoryViewer

__all__ = []


def _comparison(
    arrays: dict[str, np.ndarray], report: dict[str, Any]
) -> dict[str, Any]:
    """Bound browser data independently of the full-resolution downloadable NPZ."""
    count, dof = arrays["original_qpos"].shape
    # Keep phase/selection boundaries even when uniform drawing points miss them.
    anchors = [report["start"], report["stop"] - 1]
    anchors.extend(report["start"] + p["start"] for p in report["phases"])
    for span in report.get("timeline", ()):
        if "fit_start" in span:
            anchors.extend((span["fit_start"], span["fit_stop"] - 1))
    grid = np.unique(
        np.r_[np.linspace(0, count - 1, min(count, 1000)).astype(int), anchors]
    )
    # At most 128 columns enter the browser; every column remains in the NPZ.
    columns = min(dof, 128)
    result = dict(
        report=report,
        indices=grid.tolist(),
        timestamps=arrays["timestamps"][grid].tolist(),
        joint_names=report["joint_names"][:columns],
        total_joints=dof,
    )
    for name in ("original", "native", "patched", "proposed"):
        result[f"{name}_qpos"] = arrays[f"{name}_qpos"][grid, :columns].tolist()
    # Fail explicitly rather than letting a large browser payload stall scrubbing.
    if len(json.dumps(result, allow_nan=False).encode()) > 16 * 1024**2:
        raise ValueError("拟合预览超过 16 MiB；请减少关节数或使用离线拟合命令。")
    return result


class FitSession:
    """Retain one job/result, reject concurrent submissions and own its lifetime."""

    def __init__(self, defaults: dict[str, Any] | None = None) -> None:
        raw = deepcopy(defaults or {})
        if set(raw) - set(FitInputs.__dataclass_fields__):
            raise ValueError("Unknown fitting input defaults.")
        self.defaults = asdict(FitInputs(**raw))
        json.dumps(self.defaults, allow_nan=False)
        self._lock = threading.RLock()
        self._state: dict[str, Any] = dict(
            phase="idle",
            job_id=None,
            message="填写关节记录与机器人配置后，一次拟合完整轨迹。",
        )
        self._selection: dict[str, Any] | None = None
        self._segments: list[dict[str, Any]] | None = None
        self._revision: int | None = None
        self._preview: dict[str, Any] | None = None
        self._bundle: Path | None = None
        self._run_task: Path | None = None
        self._directory: tempfile.TemporaryDirectory[str] | None = None
        self._thread: threading.Thread | None = None
        self._viewer: TrajectoryViewer | None = None
        self._closed = False

    def snapshot(self, document: dict[str, Any]) -> dict[str, Any]:
        """Return status with selection/revision invalidation computed live."""
        with self._lock:
            stale = self._revision is not None and (
                document["revision"] != self._revision
                or (
                    self._segments != document["segments"]
                    if self._segments is not None
                    else self._selection not in document["segments"]
                )
            )
            return {
                **deepcopy(self._state),
                "stale": stale,
                "selection": deepcopy(self._selection),
                "annotation_revision": self._revision,
            }

    def start(
        self,
        payload: dict[str, Any],
        document: dict[str, Any],
        interval: Callable[[int, int], Future[tuple[FramePreview, FramePreview]]],
        check_current: Callable[[], None],
    ) -> dict[str, Any]:
        """Reserve a job before enqueueing any decoder/solver work."""
        whole = (
            payload.keys() == {"scope", "expected_revision", "inputs"}
            and payload.get("scope") == "episode"
        )
        if not whole and payload.keys() != {
            "segment_id",
            "expected_revision",
            "skill_id",
            "inputs",
        }:
            raise ValueError(
                "拟合请求需要 segment_id、expected_revision、skill_id、inputs。"
            )
        if (
            type(payload["expected_revision"]) is not int
            or payload["expected_revision"] != document["revision"]
        ):
            raise ConflictError("标注修订已变化，请保存并刷新后重试。")
        matches = (
            [
                dict(
                    id="episode",
                    label="完整轨迹",
                    start_frame=0,
                    end_frame=document["source"]["frame_count"] - 1,
                )
            ]
            if whole
            else [s for s in document["segments"] if s["id"] == payload["segment_id"]]
        )
        if len(matches) != 1:
            raise ConflictError("请先提交并保存选中的原子技能区间。")
        skill = "mixed_trajectory" if whole else payload["skill_id"]
        if not whole and skill not in RECONSTRUCTION_SKILLS:
            raise ValueError("仅支持 pick_up、slide、twist、hand_over。")
        annotated_skill = _canonical_skill_id(matches[0]["label"])
        if not whole and annotated_skill is not None and annotated_skill != skill:
            raise ValueError(
                "拟合技能必须与选中标注一致；DemoSkill 等技能目前仅支持标注。"
            )
        raw = payload["inputs"]
        if not isinstance(raw, dict) or set(raw) - set(self.defaults):
            raise ValueError("拟合输入字段不符合格式。")
        inputs = FitInputs(**{**self.defaults, **raw})
        inputs.check()
        json.dumps(asdict(inputs), allow_nan=False)
        with self._lock:
            if self._closed or self._state["phase"] == "running":
                raise ConflictError("已有拟合正在运行，请等待当前任务完成。")
            core = matches[0].get("core_interval", matches[0])
            first, last = core["start_frame"], core["end_frame"]
            clock = interval(first, last)
            if self._viewer is not None:
                self._viewer.close()
                self._viewer = None
            if self._directory is not None:
                self._directory.cleanup()
            self._directory = tempfile.TemporaryDirectory(
                prefix="embodichain-annotation-fit-"
            )
            self._preview, self._bundle, self._run_task = None, None, None
            self._selection, self._revision = deepcopy(matches[0]), document["revision"]
            self._segments = deepcopy(document["segments"]) if whole else None
            job_id = uuid4().hex
            self._state = dict(
                phase="running",
                job_id=job_id,
                message="正在获取精确视频时钟…",
                error=None,
                inputs=asdict(inputs),
                skill_id=skill,
                scope="episode" if whole else "segment",
                run_task_available=False,
                run_task_error=None,
            )
            self._thread = threading.Thread(
                target=self._run,
                args=(
                    inputs,
                    skill,
                    deepcopy(document),
                    clock,
                    check_current,
                    interval,
                ),
                name="trajectory-fit",
                daemon=True,
            )
            self._thread.start()
            return self.snapshot(document)

    def _progress(self, text: str) -> None:
        with self._lock:
            self._state["message"] = text

    def _run(
        self,
        inputs: FitInputs,
        skill: str,
        document: dict[str, Any],
        clock: Future[tuple[FramePreview, FramePreview]],
        check_current: Callable[[], None],
        interval: Callable[[int, int], Future[tuple[FramePreview, FramePreview]]],
    ) -> None:
        try:
            from embodichain.data_pipeline.skill_fitting import save_skill_fit
            from embodichain.toolkits.skill_fitting._pipeline import fit_recording

            left, right = clock.result()
            check_current()
            from embodichain.toolkits.skill_fitting._collision import (
                prepare_scene_collision,
                resolve_scene_path,
            )

            scene_path = resolve_scene_path(inputs.scene, inputs.joints)
            if scene_path is not None:
                prepare_scene_collision(scene_path, progress=self._progress)
                check_current()
            endpoints = [left] if left.index == right.index else [left, right]
            if skill == "mixed_trajectory":
                from embodichain.toolkits.skill_fitting._episode import fit_episode

                cached = {f.index: (f.time_seconds, f.delay_seconds) for f in endpoints}

                def frame_clock(first: int, last: int) -> tuple[np.ndarray, np.ndarray]:
                    if first not in cached or last not in cached:
                        for f in interval(first, last).result():
                            cached[f.index] = (f.time_seconds, f.delay_seconds)
                    values = (
                        [cached[first]]
                        if first == last
                        else [cached[first], cached[last]]
                    )
                    return np.asarray([v[0] for v in values]), np.asarray(
                        [v[1] for v in values]
                    )

                arrays, report = fit_episode(
                    inputs,
                    document,
                    frame_clock,
                    progress=self._progress,
                    check_current=check_current,
                )
            else:
                arrays, report = fit_recording(
                    inputs,
                    skill_id=skill,
                    frame_times=np.array([v.time_seconds for v in endpoints]),
                    frame_durations=np.array([v.delay_seconds for v in endpoints]),
                    start_frame=left.index,
                    end_frame=right.index,
                    provenance=dict(
                        source_identity=document["source"]["identity"],
                        source_timing=document["source"]["timing"],
                        annotation_revision=document["revision"],
                        annotation_id=self._selection["id"],
                        annotation_sha256=hashlib.sha256(
                            json.dumps(document, sort_keys=True).encode()
                        ).hexdigest(),
                        annotation_digest_kind="canonical_document_json",
                        annotation_segment=self._selection,
                        annotation_video_interval={
                            key: self._selection[key]
                            for key in ("start_frame", "end_frame")
                        },
                    ),
                    progress=self._progress,
                )
            check_current()
            preview = _comparison(arrays, report)
            bundle = Path(self._directory.name) / f"{self._state['job_id']}.npz"
            save_skill_fit(bundle, arrays, report)
            run_task, export_error = None, None
            if skill == "mixed_trajectory" and report["accepted"]:
                from embodichain.toolkits.skill_fitting._export import export_run_task

                self._progress(
                    "正在打包 run-task 配置、模型、原生技能参数与 DemoSkill tensor…"
                )
                try:
                    run_task = export_run_task(
                        Path(self._directory.name) / "run_task", arrays, report, inputs
                    )
                except Exception as error:
                    export_error = str(error)
            check_current()
            self._progress("正在加载 Viser 双机器人模型…")
            viewer = TrajectoryViewer(arrays, report)
            viewer.start()
            with self._lock:
                self._viewer = viewer
                self._preview, self._bundle = preview, bundle
                self._run_task = run_task
                self._state.update(
                    phase="done",
                    accepted=report["accepted"],
                    run_task_available=run_task is not None,
                    run_task_error=export_error,
                    message=(
                        "已生成连续回填轨迹。"
                        if report["accepted"]
                        else (
                            "部分区间未通过校验；未通过的区间保留原始值，运行包未导出。"
                            if skill == "mixed_trajectory"
                            else "候选未通过校验；回填轨迹保留原始值。"
                        )
                    ),
                )
        except Exception as error:
            with self._lock:
                self._state.update(phase="error", message="拟合失败", error=str(error))

    def preview(self, job_id: str, document: dict[str, Any]) -> dict[str, Any]:
        """Return the immutable drawing data only for a current completed job."""
        with self._lock:
            self._require_result(job_id, document)
            return {**self._preview, "viewer": self._viewer.snapshot()}

    def update_viewer(
        self, job_id: str, document: dict[str, Any], time_seconds: float
    ) -> dict[str, Any]:
        """Synchronize the current result's two models to the displayed video time."""
        with self._lock:
            self._require_result(job_id, document)
            return self._viewer.set_time(time_seconds)

    def _require_result(self, job_id: str, document: dict[str, Any]) -> None:
        if self._state["job_id"] != job_id or self._state["phase"] != "done":
            raise ValueError("拟合结果不存在或尚未完成。")
        if self.snapshot(document)["stale"]:
            raise ConflictError("结果对应的标注修订已变化，请重新拟合。")

    def open_bundle(self, job_id: str, document: dict[str, Any]) -> BinaryIO:
        """Open under the job lock so replacing a result cannot break a download."""
        with self._lock:
            self._require_result(job_id, document)
            return self._bundle.open("rb")

    def open_run_task(self, job_id: str, document: dict[str, Any]) -> BinaryIO:
        """Open the portable deployment belonging to this exact saved revision."""
        with self._lock:
            self._require_result(job_id, document)
            if self._run_task is None:
                raise ValueError(
                    self._state.get("run_task_error")
                    or "该结果没有通过全轨迹导出校验。"
                )
            return self._run_task.open("rb")

    def close(self) -> None:
        """Join cooperative native fitting before removing the sole result bundle."""
        with self._lock:
            self._closed = True
        if self._thread is not None:
            self._thread.join()
        if self._viewer is not None:
            self._viewer.close()
        if self._directory is not None:
            self._directory.cleanup()
