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
import tempfile
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
        started_perf = time.perf_counter()
        try:
            # The current SAM3D server exposes only the multi-object route.
            # Adapt a single RGBA/RGB image to that route with one mask instead
            # of calling the server's removed /generate_single_object endpoint.
            with _single_object_mask(request.image_path) as mask_path:
                response = self.generate_multiple_objects(
                    MultiObjectGenerationServerRequest(
                        image_path=request.image_path,
                        mask_paths=[mask_path],
                    ),
                    output_dir=Path(request.output_path).expanduser().resolve().parent,
                    max_retries=max_retries,
                )
            if isinstance(response, MultiObjectGenerationError):
                _record_sam3d_generation_timing(
                    operation="single_object",
                    started_perf=started_perf,
                    status="failed",
                    image_path=request.image_path,
                    output_path=request.output_path,
                    error=response.error_message,
                )
                return GeometryGenerationError(
                    error_message=response.error_message,
                    status_code=response.status_code,
                    content_type=response.content_type,
                    headers=response.headers,
                    raw_response=response.raw_response,
                )
            if not response.result.objects:
                raise RuntimeError("SAM3D returned no object for single-object input")
            source_path = Path(response.result.objects[0].geometry_path)
            output_path = Path(request.output_path).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if source_path.resolve() != output_path:
                output_path.write_bytes(source_path.read_bytes())
            parsed = GeometryGenerationServerResponse(
                ok=True,
                status="ok",
                result=GeometryGenerationResult(geometry_path=str(output_path)),
                status_code=200,
            )
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
            response = _wait_for_multi_object_result(self, response)
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


class _SingleObjectMask:
    """Temporary mask context used to adapt the single-object API."""

    def __init__(self, image_path: str | Path) -> None:
        self.image_path = Path(image_path).expanduser().resolve()
        self._tmp: Any = None
        self.path: Path | None = None

    def __enter__(self) -> Path:
        from PIL import Image

        with Image.open(self.image_path) as image:
            if image.mode in {"RGBA", "LA"} or (
                image.mode == "P" and "transparency" in image.info
            ):
                mask = image.convert("RGBA").getchannel("A")
            else:
                mask = Image.new("L", image.size, 255)
            self._tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            self.path = Path(self._tmp.name)
            mask.save(self._tmp, format="PNG")
            self._tmp.close()
        return self.path

    def __exit__(self, *_: object) -> None:
        if self.path is not None:
            self.path.unlink(missing_ok=True)


def _single_object_mask(image_path: str | Path) -> _SingleObjectMask:
    return _SingleObjectMask(image_path)


def _post_multi_object_generation_request(
    client: GeometryGenerationClient,
    url: str,
    request: MultiObjectGenerationServerRequest,
) -> requests.Response:
    image_file = _open_image_file(request.image_path)
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
                        image_file,
                    ),
                )
            ]
            + mask_files,
            timeout=(10, client.timeout_s),
        )
    finally:
        image_file.close()
        for _, (_, f) in mask_files:
            f.close()


def _wait_for_multi_object_result(
    client: GeometryGenerationClient,
    response: requests.Response,
) -> requests.Response:
    """Resolve SAM3D's queued response into the completed JSON response."""
    try:
        body = response.json()
    except ValueError:
        return response
    if not isinstance(body, dict) or isinstance(body.get("result"), dict):
        return response

    status_url = body.get("status_url")
    request_id = body.get("request_id")
    if not isinstance(status_url, str) or not status_url:
        raise RuntimeError(
            "SAM3D returned a non-terminal response without status_url: "
            f"request_id={request_id!r}"
        )

    poll_url = _join_url(client.base_url, status_url)
    deadline = time.monotonic() + client.timeout_s
    interval = float(client.config.get("task_poll_interval_s", 1.0))
    while time.monotonic() < deadline:
        poll_response = client.session.get(
            poll_url,
            timeout=(10, min(client.timeout_s, 30)),
        )
        poll_response.raise_for_status()
        poll_body = poll_response.json()
        if not isinstance(poll_body, dict):
            raise RuntimeError("SAM3D task response must be a JSON object")
        status = str(poll_body.get("status", ""))
        if isinstance(poll_body.get("result"), dict):
            return poll_response
        if status in {"failed", "cancelled", "cancelling"}:
            raise RuntimeError(
                f"SAM3D task {request_id or '<unknown>'} {status}: "
                f"{poll_body.get('error', 'unknown error')}"
            )
        time.sleep(max(0.05, min(interval, 60.0)))

    raise TimeoutError(f"Timed out waiting for SAM3D task {request_id or '<unknown>'}")


def _join_url(base_url: str, path_or_url: str) -> str:
    if path_or_url.startswith(("http://", "https://")):
        return path_or_url
    if path_or_url.startswith("/"):
        return f"{base_url}{path_or_url}"
    return f"{base_url}/{path_or_url}"
