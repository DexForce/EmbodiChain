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

__all__ = []


def _comparison(
    arrays: dict[str, np.ndarray], report: dict[str, Any]
) -> dict[str, Any]:
    """Bound browser data independently of the full-resolution downloadable NPZ."""
    count, dof = arrays["original_qpos"].shape
    # Keep phase/selection boundaries even when uniform drawing points miss them.
    anchors = [report["start"], report["stop"] - 1]
    anchors.extend(report["start"] + p["start"] for p in report["phases"])
    grid = np.unique(
        np.r_[np.linspace(0, count - 1, min(count, 1000)).astype(int), anchors]
    )
    # At most 128 columns enter the browser; every column remains in the NPZ.
    columns = min(dof, 128)
    geometry = arrays["geometry_indices"]
    view = np.linspace(0, len(geometry) - 1, min(len(geometry), 600)).astype(int)
    result = dict(
        report=report,
        indices=grid.tolist(),
        timestamps=arrays["timestamps"][grid].tolist(),
        joint_names=report["joint_names"][:columns],
        total_joints=dof,
        geometry_times=arrays["timestamps"][geometry[view]].tolist(),
    )
    for name in ("original", "native", "patched", "proposed"):
        result[f"{name}_qpos"] = arrays[f"{name}_qpos"][grid, :columns].tolist()
        result[f"{name}_links"] = arrays[f"{name}_links"][view].tolist()
        result[f"{name}_tcp"] = arrays[f"{name}_tcp"][view].tolist()
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
            phase="idle", job_id=None, message="选择已标注区间后开始拟合。"
        )
        self._selection: dict[str, Any] | None = None
        self._revision: int | None = None
        self._preview: dict[str, Any] | None = None
        self._bundle: Path | None = None
        self._directory: tempfile.TemporaryDirectory[str] | None = None
        self._thread: threading.Thread | None = None
        self._closed = False

    def snapshot(self, document: dict[str, Any]) -> dict[str, Any]:
        """Return status with selection/revision invalidation computed live."""
        with self._lock:
            stale = self._revision is not None and (
                document["revision"] != self._revision
                or self._selection not in document["segments"]
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
        if payload.keys() != {"segment_id", "expected_revision", "skill_id", "inputs"}:
            raise ValueError(
                "拟合请求需要 segment_id、expected_revision、skill_id、inputs。"
            )
        if (
            type(payload["expected_revision"]) is not int
            or payload["expected_revision"] != document["revision"]
        ):
            raise ConflictError("标注修订已变化，请保存并刷新后重试。")
        matches = [s for s in document["segments"] if s["id"] == payload["segment_id"]]
        if len(matches) != 1:
            raise ConflictError("请先提交并保存选中的原子技能区间。")
        if payload["skill_id"] not in ("pick_up", "slide", "twist", "hand_over"):
            raise ValueError("仅支持 pick_up、slide、twist、hand_over。")
        raw = payload["inputs"]
        if not isinstance(raw, dict) or set(raw) - set(self.defaults):
            raise ValueError("拟合输入字段不符合格式。")
        inputs = FitInputs(**{**self.defaults, **raw})
        inputs.check()
        json.dumps(asdict(inputs), allow_nan=False)
        with self._lock:
            if self._closed or self._state["phase"] == "running":
                raise ConflictError("已有拟合正在运行，请等待当前任务完成。")
            first, last = matches[0]["start_frame"], matches[0]["end_frame"]
            clock = interval(first, last)
            if self._directory is None:
                self._directory = tempfile.TemporaryDirectory(
                    prefix="embodichain-annotation-fit-"
                )
            if self._bundle is not None:
                self._bundle.unlink(missing_ok=True)
            self._preview, self._bundle = None, None
            self._selection, self._revision = deepcopy(matches[0]), document["revision"]
            job_id = uuid4().hex
            self._state = dict(
                phase="running",
                job_id=job_id,
                message="正在获取选区的精确视频时钟…",
                error=None,
                inputs=asdict(inputs),
                skill_id=payload["skill_id"],
            )
            self._thread = threading.Thread(
                target=self._run,
                args=(
                    inputs,
                    payload["skill_id"],
                    deepcopy(document),
                    clock,
                    check_current,
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
    ) -> None:
        try:
            from embodichain.data_pipeline.skill_fitting import save_skill_fit
            from embodichain.toolkits.skill_fitting._pipeline import fit_recording

            left, right = clock.result()
            check_current()
            endpoints = [left] if left.index == right.index else [left, right]
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
                ),
                progress=self._progress,
            )
            check_current()
            preview = _comparison(arrays, report)
            bundle = Path(self._directory.name) / f"{self._state['job_id']}.npz"
            save_skill_fit(bundle, arrays, report)
            with self._lock:
                self._preview, self._bundle = preview, bundle
                self._state.update(
                    phase="done",
                    accepted=report["accepted"],
                    message=(
                        "已生成连续回填轨迹。"
                        if report["accepted"]
                        else "候选未通过校验；回填轨迹保留原始值。"
                    ),
                )
        except Exception as error:
            with self._lock:
                self._state.update(phase="error", message="拟合失败", error=str(error))

    def preview(self, job_id: str, document: dict[str, Any]) -> dict[str, Any]:
        """Return the immutable drawing data only for a current completed job."""
        with self._lock:
            self._require_result(job_id, document)
            return self._preview

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

    def close(self) -> None:
        """Join cooperative native fitting before removing the sole result bundle."""
        with self._lock:
            self._closed = True
        if self._thread is not None:
            self._thread.join()
        if self._directory is not None:
            self._directory.cleanup()
