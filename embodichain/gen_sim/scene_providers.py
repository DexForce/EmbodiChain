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

"""Provider-neutral lifecycle for long-running GenSim scene generation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Protocol

__all__ = [
    "CancellationReport",
    "SceneGenerationProvider",
    "SceneGenerationRequest",
    "SceneJobRef",
    "SceneJobStatus",
    "load_scene_generation_provider",
]

SCENE_PROVIDER_ENTRY_POINT = "embodichain.gensim_scene_providers"
_JOB_STATES = frozenset(
    {"queued", "running", "succeeded", "failed", "cancelled", "unknown"}
)


def _canonical_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class SceneGenerationRequest:
    """One image-only scene request with task requirements retained by hash."""

    task_id: str
    image_path: Path
    image_sha256: str
    requirements_path: Path
    requirements_sha256: str
    output: Path
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        task_id = str(self.task_id).strip()
        if not task_id:
            raise ValueError("Scene generation task_id must be nonempty")
        image = Path(self.image_path).expanduser().resolve(strict=True)
        requirements = Path(self.requirements_path).expanduser().resolve(strict=True)
        output = Path(self.output).expanduser().absolute()
        for label, value in (
            ("image_sha256", self.image_sha256),
            ("requirements_sha256", self.requirements_sha256),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"Scene generation {label} must be a SHA-256 digest")
        if hashlib.sha256(image.read_bytes()).hexdigest() != self.image_sha256:
            raise ValueError("Scene generation image changed before submission")
        if (
            hashlib.sha256(requirements.read_bytes()).hexdigest()
            != self.requirements_sha256
        ):
            raise ValueError("Scene requirements changed before submission")
        json.dumps(dict(self.options), ensure_ascii=False, allow_nan=False)
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "image_path", image)
        object.__setattr__(self, "requirements_path", requirements)
        object.__setattr__(self, "output", output)
        object.__setattr__(self, "options", dict(self.options))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "embodichain.scene-generation-request/v1",
            "task_id": self.task_id,
            "image_path": str(self.image_path),
            "image_sha256": self.image_sha256,
            "requirements_path": str(self.requirements_path),
            "requirements_sha256": self.requirements_sha256,
            "output": str(self.output),
            "options": dict(self.options),
        }

    @property
    def request_hash(self) -> str:
        return _canonical_hash(
            {
                "schema_version": "embodichain.scene-generation-request/v1",
                "task_id": self.task_id,
                "image_sha256": self.image_sha256,
                "requirements_sha256": self.requirements_sha256,
                "options": dict(self.options),
            }
        )


@dataclass(frozen=True)
class SceneJobRef:
    provider: str
    job_id: str
    request_hash: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("provider", "job_id", "request_hash"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"SceneJobRef.{name} must be nonempty")
        json.dumps(dict(self.metadata), ensure_ascii=False, allow_nan=False)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "embodichain.scene-job-ref/v1",
            "provider": self.provider,
            "job_id": self.job_id,
            "request_hash": self.request_hash,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SceneJobStatus:
    job: SceneJobRef
    status: str
    candidate_id: str | None = None
    reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in _JOB_STATES:
            raise ValueError(f"Unsupported scene job status {self.status!r}")
        if self.status == "succeeded" and not self.candidate_id:
            raise ValueError("Successful scene jobs require candidate_id")
        json.dumps(dict(self.metadata), ensure_ascii=False, allow_nan=False)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "embodichain.scene-job-status/v1",
            "job": self.job.to_dict(),
            "status": self.status,
            "candidate_id": self.candidate_id,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CancellationReport:
    job: SceneJobRef
    status: str
    stopped: bool
    partial_effects: bool = False


class SceneGenerationProvider(Protocol):
    """Lifecycle implemented by one selected scene-generation deployment."""

    provider_id: str

    def submit(
        self, request: SceneGenerationRequest, *, idempotency_key: str
    ) -> SceneJobRef: ...

    def inspect(self, job: SceneJobRef) -> SceneJobStatus: ...

    def resume(self, job: SceneJobRef) -> SceneJobStatus: ...

    def cancel(self, job: SceneJobRef) -> CancellationReport: ...

    def materialize(self, job: SceneJobRef, output: Path) -> Path: ...


def load_scene_generation_provider(
    provider_id: str, profile: Mapping[str, Any] | None = None
) -> SceneGenerationProvider:
    """Load only the selected built-in or installed provider."""
    if not isinstance(provider_id, str) or not provider_id.strip():
        raise ValueError("Scene provider ID must be nonempty")
    selected = provider_id.strip()
    configuration = {} if profile is None else dict(profile)
    if selected == "legacy_scene_engine":
        from .scene_provider_legacy import LegacySceneEngineProvider

        return LegacySceneEngineProvider(configuration)
    matches = list(entry_points(group=SCENE_PROVIDER_ENTRY_POINT, name=selected))
    if len(matches) != 1:
        raise ValueError(
            f"Scene provider {selected!r} is not installed exactly once in "
            f"{SCENE_PROVIDER_ENTRY_POINT!r}"
        )
    factory = matches[0].load()
    provider = factory(configuration)
    if getattr(provider, "provider_id", None) != selected:
        raise ValueError("Scene provider returned a mismatched provider_id")
    for operation in ("submit", "inspect", "resume", "cancel", "materialize"):
        if not callable(getattr(provider, operation, None)):
            raise TypeError(f"Scene provider lacks {operation}()")
    return provider
