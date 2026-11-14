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

"""Compatibility provider for the in-package Scene Engine."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .scene_providers import (
    CancellationReport,
    SceneGenerationRequest,
    SceneJobRef,
    SceneJobStatus,
)
from .task_engine.orchestration.workflow_artifacts import write_run_json

__all__: list[str] = []


class LegacySceneEngineProvider:
    provider_id = "legacy_scene_engine"

    def __init__(self, profile: Mapping[str, Any]) -> None:
        values = dict(profile)
        self._services = values.pop("_services", None)
        if values:
            raise ValueError("legacy_scene_engine does not accept provider options")

    @staticmethod
    def _record(job: SceneJobRef) -> Path:
        value = job.metadata.get("record")
        if not isinstance(value, str):
            raise TypeError("Legacy scene job has no durable record")
        return Path(value)

    def submit(
        self, request: SceneGenerationRequest, *, idempotency_key: str
    ) -> SceneJobRef:
        if not idempotency_key:
            raise ValueError("Legacy scene submission requires an idempotency key")
        root = request.output / "legacy-scene"
        record = request.output / "legacy-scene-job.json"
        job = SceneJobRef(
            provider=self.provider_id,
            job_id=hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()[:32],
            request_hash=request.request_hash,
            metadata={"record": str(record), "workspace": str(root)},
        )
        if record.is_file():
            current = json.loads(record.read_text(encoding="utf-8"))
            if current.get("request_hash") != request.request_hash:
                raise ValueError("Legacy scene job key is bound to another request")
            return job
        from .scene_engine.pipeline.generate import generate_scene_from_image

        try:
            generate_scene_from_image(
                request.image_path,
                root,
                scene_z_rotation_degrees=180.0,
                services=self._services,
            )
            result = {
                "schema_version": "embodichain.legacy-scene-job/v1",
                "job_id": job.job_id,
                "request_hash": request.request_hash,
                "status": "succeeded",
                "candidate_id": job.job_id,
                "workspace": str(root),
            }
        except BaseException as error:
            result = {
                "schema_version": "embodichain.legacy-scene-job/v1",
                "job_id": job.job_id,
                "request_hash": request.request_hash,
                "status": "failed",
                "reason": f"{type(error).__name__}: {error}",
                "workspace": str(root),
            }
            write_run_json(record, result, overwrite=False)
            raise
        write_run_json(record, result, overwrite=False)
        return job

    def inspect(self, job: SceneJobRef) -> SceneJobStatus:
        value = json.loads(self._record(job).read_text(encoding="utf-8"))
        return SceneJobStatus(
            job=job,
            status=value["status"],
            candidate_id=value.get("candidate_id"),
            reason=value.get("reason"),
            metadata={"workspace": value["workspace"]},
        )

    def resume(self, job: SceneJobRef) -> SceneJobStatus:
        return self.inspect(job)

    def cancel(self, job: SceneJobRef) -> CancellationReport:
        status = self.inspect(job)
        return CancellationReport(
            job=job,
            status=status.status,
            stopped=status.status != "running",
            partial_effects=status.status in {"failed", "succeeded"},
        )

    def materialize(self, job: SceneJobRef, output: Path) -> Path:
        status = self.inspect(job)
        if status.status != "succeeded":
            raise ValueError("Legacy scene job is not successful")
        workspace = Path(status.metadata["workspace"]).resolve(strict=True)
        return workspace
