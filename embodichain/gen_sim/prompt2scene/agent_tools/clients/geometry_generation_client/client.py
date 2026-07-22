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

"""Client for the SAM3D geometry generation server."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import requests

from embodichain.gen_sim.prompt2scene.agent_tools.clients.base import BaseHttpClient
from embodichain.gen_sim.prompt2scene.agent_tools.clients.common import (
    validate_required_strings,
)
from embodichain.gen_sim.prompt2scene.agent_tools.clients.config import (
    DEFAULT_CLIENT_CONFIG_PATH,
)
from embodichain.gen_sim.prompt2scene.agent_tools.clients.geometry_generation_client.parser import (
    parse_geometry_generation_response,
    parse_multi_object_generation_response,
)
from embodichain.gen_sim.prompt2scene.agent_tools.clients.geometry_generation_client.schemas import (
    GeometryGenerationError,
    GeometryGenerationResult,
    GeometryGenerationServerRequest,
    GeometryGenerationServerResponse,
    MultiObjectGenerationError,
    MultiObjectGenerationObject,
    MultiObjectGenerationServerRequest,
    MultiObjectGenerationServerResponse,
)

__all__ = [
    "DEFAULT_CLIENT_CONFIG_PATH",
    "GeometryGenerationClient",
    "GeometryGenerationError",
    "GeometryGenerationResult",
    "GeometryGenerationServerRequest",
    "GeometryGenerationServerResponse",
    "MultiObjectGenerationError",
    "MultiObjectGenerationObject",
    "MultiObjectGenerationServerRequest",
    "MultiObjectGenerationServerResponse",
    "clear_sam3d_generation_timings",
    "get_sam3d_generation_timings",
]

_SAM3D_GENERATION_TIMINGS: list[dict[str, Any]] = []


def clear_sam3d_generation_timings() -> None:
    """Clear SAM3D timing records for the current runner invocation."""
    _SAM3D_GENERATION_TIMINGS.clear()


def get_sam3d_generation_timings() -> list[dict[str, Any]]:
    """Return SAM3D timing records collected in this process."""
    return [dict(item) for item in _SAM3D_GENERATION_TIMINGS]


class GeometryGenerationClient(BaseHttpClient):
    """Client for making single-object SAM3D geometry generation requests."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout_s: int | None = None,
        config_path: Path | None = None,
        config_key: str = "sam3d_generation",
        session: requests.Session | None = None,
    ) -> None:
        """Initialize the geometry generation client."""
        super().__init__(
            config_key=config_key,
            server_name="Geometry generation server",
            base_url=base_url,
            timeout_s=timeout_s,
            config_path=config_path,
            session=session,
            trust_env=False,
        )
        self.generate_single_object_path = str(
            self.config.get("generate_single_object_path", "/generate_single_object")
        )
        self.generate_multiple_objects_path = str(
            self.config.get(
                "generate_multiple_objects_path", "/generate_multiple_objects"
            )
        )

    def generate(
        self,
        request: GeometryGenerationServerRequest,
        *,
        max_retries: int = 3,
    ) -> GeometryGenerationServerResponse | GeometryGenerationError:
        """Generate one GLB mesh from an object image and save it locally."""
        _validate_request(request)
        url = f"{self.base_url}{self.generate_single_object_path}"
        started_perf = time.perf_counter()
        try:
            response = self.post_with_retries(
                lambda: _post_geometry_generation_request(self, url, request),
                max_retries=max_retries,
                error_cls=GeometryGenerationError,
                request_label="geometry_generation",
            )
            if isinstance(response, GeometryGenerationError):
                _record_sam3d_generation_timing(
                    operation="single_object",
                    started_perf=started_perf,
                    status="failed",
                    image_path=request.image_path,
                    output_path=request.output_path,
                    error=response.error_message,
                )
                return response
            parsed = parse_geometry_generation_response(response, request)
            _record_sam3d_generation_timing(
                operation="single_object",
                started_perf=started_perf,
                status="ok",
                image_path=request.image_path,
                output_path=request.output_path,
            )
            return parsed
        except Exception as exc:
            _record_sam3d_generation_timing(
                operation="single_object",
                started_perf=started_perf,
                status="failed",
                image_path=request.image_path,
                output_path=request.output_path,
                error=str(exc),
            )
            raise

    def generate_multiple_objects(
        self,
        request: MultiObjectGenerationServerRequest,
        *,
        output_dir: Path | None = None,
        max_retries: int = 3,
    ) -> MultiObjectGenerationServerResponse | MultiObjectGenerationError:
        """Generate multiple GLB meshes from one image and multiple masks."""
        _validate_multi_object_request(request)
        url = f"{self.base_url}{self.generate_multiple_objects_path}"
        started_perf = time.perf_counter()
        try:
            response = self.post_with_retries(
                lambda: _post_multi_object_generation_request(self, url, request),
                max_retries=max_retries,
                error_cls=MultiObjectGenerationError,
                request_label="multi_object_geometry_generation",
            )
            if isinstance(response, MultiObjectGenerationError):
                _record_sam3d_generation_timing(
                    operation="multi_object",
                    started_perf=started_perf,
                    status="failed",
                    image_path=request.image_path,
                    output_dir=output_dir,
                    mask_count=len(request.mask_paths),
                    error=response.error_message,
                )
                return response
            parsed = parse_multi_object_generation_response(
                response,
                self.base_url,
                output_dir=output_dir,
                session=self.session,
            )
            _record_sam3d_generation_timing(
                operation="multi_object",
                started_perf=started_perf,
                status="ok",
                image_path=request.image_path,
                output_dir=output_dir,
                mask_count=len(request.mask_paths),
            )
            return parsed
        except Exception as exc:
            _record_sam3d_generation_timing(
                operation="multi_object",
                started_perf=started_perf,
                status="failed",
                image_path=request.image_path,
                output_dir=output_dir,
                mask_count=len(request.mask_paths),
                error=str(exc),
            )
            raise


def _record_sam3d_generation_timing(
    *,
    operation: str,
    started_perf: float,
    status: str,
    image_path: str | Path,
    output_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    mask_count: int | None = None,
    error: str | None = None,
) -> None:
    record: dict[str, Any] = {
        "operation": operation,
        "status": status,
        "elapsed_seconds": round(max(0.0, time.perf_counter() - started_perf), 6),
        "image_path": str(image_path),
    }
    if output_path is not None:
        record["output_path"] = str(output_path)
    if output_dir is not None:
        record["output_dir"] = str(output_dir)
    if mask_count is not None:
        record["mask_count"] = int(mask_count)
    if error:
        record["error"] = error
    _SAM3D_GENERATION_TIMINGS.append(record)


def _validate_request(request: GeometryGenerationServerRequest) -> None:
    validate_required_strings(
        {
            "Geometry generation image_path": request.image_path,
            "Geometry generation output_path": request.output_path,
        }
    )
    image_path = Path(request.image_path).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"Geometry generation input not found: {image_path}")
    if not str(request.output_path).lower().endswith(".glb"):
        raise ValueError("Geometry generation output_path must be a GLB file path.")


def _post_geometry_generation_request(
    client: GeometryGenerationClient,
    url: str,
    request: GeometryGenerationServerRequest,
) -> requests.Response:
    with _open_image_file(request.image_path) as image_file:
        return client.session.post(
            url,
            data=request.to_form_data(),
            files={
                "image": (
                    Path(request.image_path).name,
                    image_file,
                )
            },
            timeout=(10, client.timeout_s),
        )


def _open_image_file(image_path: str | Path) -> Any:
    return Path(image_path).expanduser().resolve().open("rb")


def _validate_multi_object_request(
    request: MultiObjectGenerationServerRequest,
) -> None:
    validate_required_strings(
        {"Multi-object geometry generation image_path": request.image_path}
    )
    image_path = Path(request.image_path).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Multi-object geometry generation input not found: {image_path}"
        )
    if not request.mask_paths:
        raise ValueError("mask_paths must be non-empty.")
    for mask_path in request.mask_paths:
        if not Path(mask_path).expanduser().is_file():
            raise FileNotFoundError(
                f"Multi-object geometry mask not found: {mask_path}"
            )


def _post_multi_object_generation_request(
    client: GeometryGenerationClient,
    url: str,
    request: MultiObjectGenerationServerRequest,
) -> requests.Response:
    mask_files = [
        ("masks", (Path(p).name, Path(p).expanduser().resolve().open("rb")))
        for p in request.mask_paths
    ]
    try:
        return client.session.post(
            url,
            data=request.to_form_data(),
            files=[
                (
                    "image",
                    (
                        Path(request.image_path).name,
                        _open_image_file(request.image_path),
                    ),
                )
            ]
            + mask_files,
            timeout=(10, client.timeout_s),
        )
    finally:
        for _, (_, f) in mask_files:
            f.close()
