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

"""Compatibility call shapes over capability services.

Only this boundary understands historical client results and message dictionaries.
Business code may retain prompt helpers while consuming replaceable capabilities.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Protocol
import json
from .contracts import (
    ChatRequest,
    ChatMessage,
    ChatResponse,
    TextPart,
    ImagePart,
    GeometryRequest,
    GeometryResult,
    ImageGenerationRequest,
    SegmentationRequest,
    ArticulationRequest,
)
from .runtime import GenSimServices, service_scope

__all__: list[str] = []


class VisionClient(Protocol):
    """Historical prompt helper interface, independent of a provider."""

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_path: str | Path | None = None,
    ) -> str: ...


class VisionFacade:
    """Adapt existing single-image prompt helpers to a role-bound chat service."""

    def __init__(
        self, services: GenSimServices, role: str, model: str | None = None
    ) -> None:
        self.services, self.role, self.model = services, role, model

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_path: str | Path | None = None,
    ) -> str:
        parts = (TextPart(user_prompt),) + (
            (ImagePart(Path(image_path)),) if image_path is not None else ()
        )
        return (
            self.services.get(self.role)
            .complete(
                ChatRequest(
                    (
                        ChatMessage("system", (TextPart(system_prompt),)),
                        ChatMessage("user", parts),
                    ),
                    model=self.model,
                )
            )
            .text
        )


def structured_call(
    *,
    prompt: str,
    schema: Mapping[str, Any],
    model: str | None = None,
    role: str = "task_interpretation",
    services: GenSimServices | None = None,
    system_prompt: str = "Return only the requested structured JSON response. Never return reasoning, coordinates, or AtomicAction nodes.",
) -> Mapping[str, Any]:
    """Request structured output; consuming owners still validate semantics."""
    with service_scope(services) as run:
        response = run.get(role).complete(
            ChatRequest(
                (
                    ChatMessage("system", (TextPart(system_prompt),)),
                    ChatMessage("user", (TextPart(prompt),)),
                ),
                schema=schema,
                model=model,
                temperature=0,
            )
        )
        if response.structured is not None:
            return response.structured
        value = json.loads(response.text)
        if not isinstance(value, dict):
            raise ValueError("Structured output must be an object")
        return value


def reconstruct(client: Any, request: GeometryRequest) -> GeometryResult:
    """Accept new services and historical injected geometry clients."""
    if callable(getattr(client, "reconstruct", None)):
        result = client.reconstruct(request)
        expected = [uid for uid, _ in request.object_masks]
        if [obj.object_id for obj in result.objects] != expected:
            from .errors import ServiceResponseError

            raise ServiceResponseError(
                "Reconstruction object IDs do not match requested masks"
            )
        return result
    from .adapters.normalized import geometry_result

    return geometry_result(client, request)


class GeometryFacade:
    """Lazily access reconstruction while leaving run ownership unchanged."""

    def __init__(self, services: GenSimServices) -> None:
        self.services = services

    def check_health(self) -> None:
        self.services.get("geometry_reconstruction")

    def close(self) -> None:
        pass

    def reconstruct(self, request: GeometryRequest) -> GeometryResult:
        return self.services.get("geometry_reconstruction").reconstruct(request)


class ImageFacade:
    """Historical image helper shape over a replaceable image service."""

    def __init__(self, services: GenSimServices) -> None:
        self.services = services

    def check_health(self) -> None:
        self.services.get("image_generation")

    def close(self) -> None:
        pass

    def generate_image_by_prompt(self, *, prompt: str, output_path: str | Path) -> Path:
        return (
            self.services.get("image_generation")
            .generate(ImageGenerationRequest(prompt, Path(output_path)))
            .path
        )


class SegmentationFacade:
    """Historical segmentation helper shape over a replaceable service."""

    def __init__(self, services: GenSimServices) -> None:
        self.services = services

    def check_health(self) -> None:
        self.services.get("image_segmentation")

    def close(self) -> None:
        pass

    def segment_single_object(
        self, *, image_path: str | Path, prompt: str
    ) -> list[dict[str, Any]]:
        return [
            dict(mask)
            for mask in self.services.get("image_segmentation")
            .segment(SegmentationRequest(Path(image_path), prompt))
            .masks
        ]


class ArticulationFacade:
    """Historical USDC helper shape over a replaceable articulation service."""

    def __init__(self, services: GenSimServices) -> None:
        self.services = services

    def check_health(self) -> None:
        self.services.get("articulated_generation")

    def close(self) -> None:
        pass

    def generate_articulated_usdc(
        self, *, prompt: str, image_path: str | Path, output_path: str | Path
    ) -> Path:
        return (
            self.services.get("articulated_generation")
            .generate(ArticulationRequest(prompt, Path(output_path), Path(image_path)))
            .path
        )


def chat_from_legacy_messages(
    messages: list[dict[str, Any]],
    *,
    services: GenSimServices | None = None,
    role: str,
    temperature: float | None = None,
) -> str:
    """Translate old ordered multimodal dictionaries at the compatibility edge."""
    normalized = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            parts = (TextPart(content),)
        else:
            parts = tuple(
                (
                    TextPart(part["text"])
                    if part["type"] == "text"
                    else ImagePart(part["image_url"]["url"])
                )
                for part in content
            )
        normalized.append(ChatMessage(message["role"], parts))
    with service_scope(services) as run:
        return (
            run.get(role)
            .complete(ChatRequest(tuple(normalized), temperature=temperature))
            .text
        )


class ImageClient(Protocol):
    """Compatibility prompt-to-image helper contract."""

    def check_health(self) -> None: ...
    def close(self) -> None: ...
    def generate_image_by_prompt(
        self, *, prompt: str, output_path: str | Path
    ) -> Path: ...


class SegmentationClient(Protocol):
    """Compatibility segmentation helper contract."""

    def check_health(self) -> None: ...
    def close(self) -> None: ...
    def segment_single_object(
        self, *, image_path: str | Path, prompt: str
    ) -> list[dict[str, Any]]: ...


class GeometryClient(Protocol):
    """Canonical geometry helper used by generation pipelines."""

    def check_health(self) -> None: ...
    def close(self) -> None: ...
    def reconstruct(self, request: GeometryRequest) -> GeometryResult: ...


class ArticulationClient(Protocol):
    """Compatibility validated-USDC helper contract."""

    def check_health(self) -> None: ...
    def close(self) -> None: ...
    def generate_articulated_usdc(
        self, *, prompt: str, image_path: str | Path, output_path: str | Path
    ) -> Path: ...


def vision_for_role(client: VisionClient, role: str) -> VisionClient:
    """Select a role for managed clients; preserve explicit legacy injection."""
    if isinstance(client, VisionFacade):
        return VisionFacade(client.services, role, client.model)
    return client


class ArticulationJobsFacade:
    """Borrow UI job operations while owning their configured service run."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float,
        *,
        services: GenSimServices | None = None,
    ) -> None:
        from .adapters.articulation_jobs import ArticulationServerClient

        self._owns_services = services is None
        self.services = services or GenSimServices()
        self.display_name = (
            base_url if self.services.cfg is None else "configured articulation service"
        )
        try:
            if self.services.cfg is None and services is None:
                # Preserve Gradio's existing process/dotenv-resolved settings.
                self.services._injected["articulation_jobs"] = ArticulationServerClient(
                    base_url, timeout_seconds=timeout_seconds
                )
            self._client = self.services.get("articulation_jobs")
        except BaseException:
            if self._owns_services:
                self.services.close()
            raise

    def __getattr__(self, name: str) -> Any:
        from .adapters.articulation_jobs import ArticulationServerError

        if name not in {"health", "submit", "status", "cancel", "download"}:
            raise AttributeError(name)

        def invoke(*args: Any, **kwargs: Any) -> Any:
            try:
                result = getattr(self._client, name)(*args, **kwargs)
            except Exception as error:
                if name == "submit":
                    self.close()
                from .errors import ServiceError

                if isinstance(error, ServiceError):
                    raise ArticulationServerError(
                        f"Articulation service failure: {error.code}"
                    ) from None
                raise
            if name == "cancel":
                self.close()
            return result

        return invoke

    def close(self) -> None:
        if self._owns_services:
            self.services.close()
