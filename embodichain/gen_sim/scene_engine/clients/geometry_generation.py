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


from __future__ import annotations

from contextlib import ExitStack
import json
import os
from pathlib import Path
import time
from typing import Any

import requests

from embodichain.gen_sim.env import load_gen_sim_env


class GeometryGenerationClient:
    """Manage the Geometry Generation Server connection."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_s: int,
        max_attempts: int,
        health_path: str,
        generate_objects_path: str,
        session: requests.Session | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._max_attempts = max_attempts
        self._health_path = health_path
        self._generate_objects_path = generate_objects_path
        self._session = session or requests.Session()

    @classmethod
    def from_env(cls) -> "GeometryGenerationClient":
        """Create a client from the shared GenSim ``.env`` configuration."""
        return cls(**_load_config())

    def check_health(self) -> None:
        last_error: Exception | None = None
        for _ in range(self._max_attempts):
            try:
                response = self._session.get(
                    self._url(self._health_path),
                    timeout=10,  # Use a shorter timeout for avoiding long waits.
                )
                response.raise_for_status()
                response_data = response.json()
                if (
                    not isinstance(response_data, dict)
                    or response_data.get("ok") is not True
                ):
                    raise RuntimeError(
                        "Geometry Generation Server health response does not contain ok=true."
                    )
                return
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                last_error = exc

        assert last_error is not None
        raise RuntimeError(
            "Geometry Generation Server health check failed after "
            f"{self._max_attempts} attempts."
        ) from last_error

    def close(self) -> None:
        self._session.close()

    def generate_objects(
        self,
        *,
        image_path: str | Path,
        object_masks: list[tuple[str, Path]],
        output_root: str | Path,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Generate objects through the geometry server's mask-list endpoint.

        The SAM3D service represents both one-object and multi-object jobs as one
        image plus a multipart ``masks`` list. The number of list items is the
        only difference, so keeping one implementation prevents the two client
        paths from drifting apart.
        """

        # Check, validate then wrap each content of the request.
        resolved_image_path = Path(image_path).expanduser().resolve()
        if not resolved_image_path.is_file():
            raise FileNotFoundError(
                f"Geometry generation input not found: {resolved_image_path}"
            )
        if not object_masks:
            raise ValueError("Geometry generation object_masks must not be empty.")
        object_ids = [object_id for object_id, _ in object_masks]
        if len(set(object_ids)) != len(object_ids):
            raise ValueError("Geometry generation object_ids must be unique.")

        resolved_object_masks: list[tuple[str, Path]] = []
        for object_id, mask_path in object_masks:
            resolved_mask_path = Path(mask_path).expanduser().resolve()
            if not resolved_mask_path.is_file():
                raise FileNotFoundError(
                    f"Geometry generation mask not found: {resolved_mask_path}"
                )
            resolved_object_masks.append((object_id, resolved_mask_path))

        # Send one multipart image + masks request, matching test_sam3d_client.py.
        response_data, response_objects = self._request_objects(
            image_path=resolved_image_path,
            object_masks=resolved_object_masks,
        )

        resolved_output_root = Path(output_root).expanduser().resolve()
        resolved_output_root.mkdir(parents=True, exist_ok=True)

        # This loop will iterate min(len(resolved_object_masks), len(response_objects)) times
        # , which is safe because we validated the lengths earlier.
        for (
            object_id,
            _,
        ), response_object in zip(  # Pair each object_id with its response_object for downloading the glb.
            resolved_object_masks,
            response_objects,
        ):
            output_path = resolved_output_root / f"{object_id}.glb"
            self._download_glb(response_object["mesh"], output_path)
        return response_data, response_objects

    def _request_objects(
        self,
        *,
        image_path: Path,
        object_masks: list[tuple[str, Path]],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        last_error: Exception | None = None
        for _ in range(self._max_attempts):
            try:
                # This stack manages the context of multiple open files, ensuring they are closed after the request.
                with ExitStack() as stack:
                    image_file = stack.enter_context(image_path.open("rb"))
                    mask_files = [
                        stack.enter_context(mask_path.open("rb"))
                        for _, mask_path in object_masks
                    ]
                    response = self._session.post(
                        self._url(self._generate_objects_path),
                        files=[
                            (
                                "image",
                                (
                                    image_path.name,
                                    image_file,
                                    _image_content_type(image_path),
                                ),
                            ),
                            *[
                                (
                                    "masks",
                                    (f"{object_id}.png", mask_file, "image/png"),
                                )
                                for (object_id, _), mask_file in zip(
                                    object_masks,
                                    mask_files,
                                )
                            ],
                        ],
                        timeout=self._timeout_s,
                    )
                response.raise_for_status()
                try:
                    response_data = response.json()
                except ValueError as exc:
                    raise RuntimeError(
                        "Geometry Generation Server response is not valid JSON."
                    ) from exc
                response_data = self._wait_for_task_if_needed(response_data)
                response_objects = _parse_objects_response(
                    response_data,
                    object_ids=[object_id for object_id, _ in object_masks],
                )
                return response_data, response_objects
            except (requests.RequestException, RuntimeError) as exc:
                last_error = exc

        assert last_error is not None
        raise RuntimeError(
            "Geometry Generation Server request failed after "
            f"{self._max_attempts} attempts."
        ) from last_error

    def _wait_for_task_if_needed(self, response_data: object) -> dict[str, Any]:
        """Poll a queued SAM3D job until it returns its final result."""
        if not isinstance(response_data, dict):
            raise RuntimeError(
                "Geometry Generation Server response must be a JSON object."
            )

        status = response_data.get("status")
        if not isinstance(status, str) or "waiting" not in status:
            return response_data

        request_id = response_data.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise RuntimeError(
                "Geometry Generation Server queued response has no request_id."
            )

        # The server test client uses one-second polling and permits ten minutes
        # for a queued job. Keep the same contract here.
        for _ in range(600):
            try:
                response = self._session.get(
                    self._url(f"/tasks/{request_id}"),
                    timeout=10,
                )
                response.raise_for_status()
                task_data = response.json()
            except (requests.RequestException, ValueError) as exc:
                raise RuntimeError(
                    f"Geometry Generation Server task polling failed: {request_id}."
                ) from exc

            if not isinstance(task_data, dict):
                raise RuntimeError(
                    "Geometry Generation Server task response must be a JSON object."
                )
            task_status = task_data.get("status")
            if task_status == "succeeded":
                return task_data
            if task_status in {"failed", "cancelled"}:
                raise RuntimeError(
                    "Geometry Generation Server task "
                    f"{task_status}: {task_data.get('error', 'unknown error')}"
                )
            if not isinstance(task_status, str) or (
                task_status != "running" and "waiting" not in task_status
            ):
                raise RuntimeError(
                    "Geometry Generation Server returned unknown task status: "
                    f"{task_status!r}."
                )

            time.sleep(1)

        raise RuntimeError(
            "Geometry Generation Server task timed out after 600 seconds: "
            f"{request_id}."
        )

    def _download_glb(self, mesh_path: str, output_path: Path) -> None:
        last_error: Exception | None = None
        for _ in range(self._max_attempts):
            try:
                response = self._session.get(
                    self._mesh_url(mesh_path),
                    timeout=self._timeout_s,
                )
                response.raise_for_status()
                glb_bytes = response.content
                if not glb_bytes.startswith(b"glTF"):
                    raise RuntimeError(
                        "Geometry Generation Server returned invalid GLB content."
                    )
                output_path.write_bytes(glb_bytes)
                return
            except (requests.RequestException, RuntimeError) as exc:
                last_error = exc

        assert last_error is not None
        raise RuntimeError(
            "Geometry Generation Server GLB download failed after "
            f"{self._max_attempts} attempts: {mesh_path}"
        ) from last_error

    def _mesh_url(self, mesh_path: str) -> str:
        if mesh_path.startswith(("http://", "https://")):
            return mesh_path
        return self._url(mesh_path)

    def _url(self, path: str) -> str:
        return f"{self._base_url}/{path.lstrip('/')}"


def _parse_objects_response(
    response_data: object,
    *,
    object_ids: list[str],
) -> list[dict[str, Any]]:
    if not isinstance(response_data, dict):
        raise RuntimeError("Geometry Generation Server response must be a JSON object.")
    if response_data.get("ok") is not True:
        raise RuntimeError(
            "Geometry Generation Server request failed: "
            f"{response_data.get('error', 'ok is not true')}"
        )
    result = response_data.get("result")
    if not isinstance(result, dict):
        raise RuntimeError(
            "Geometry Generation Server response must contain a result object."
        )
    response_objects = result.get("objects")
    if not isinstance(response_objects, list) or len(response_objects) != len(
        object_ids
    ):
        raise RuntimeError(
            "Geometry Generation Server response object count does not match masks."
        )

    parsed_objects: list[dict[str, Any]] = []
    for index, (object_id, response_object) in enumerate(
        zip(object_ids, response_objects)
    ):
        if not isinstance(response_object, dict):
            raise RuntimeError(
                f"Geometry Generation Server object {index} must be a JSON object."
            )
        if response_object.get("name") != object_id:
            raise RuntimeError(
                "Geometry Generation Server object name does not match its "
                f"requested id: {object_id!r}."
            )
        mesh_path = response_object.get("mesh")
        if not isinstance(mesh_path, str) or not mesh_path:
            raise RuntimeError(
                f"Geometry Generation Server object {index} has no mesh path."
            )
        parsed_objects.append(
            {
                "mesh": mesh_path,
                "rotation_quaternion_wxyz": _parse_numeric_list(
                    response_object.get("rotation_quaternion_wxyz"),
                    expected_length=4,
                    field_name=f"objects[{index}].rotation_quaternion_wxyz",
                ),
                "translation": _parse_numeric_list(
                    response_object.get("translation"),
                    expected_length=3,
                    field_name=f"objects[{index}].translation",
                ),
                "scale": _parse_numeric_list(
                    response_object.get("scale"),
                    expected_length=3,
                    field_name=f"objects[{index}].scale",
                ),
            }
        )
    return parsed_objects


def _parse_numeric_list(
    value: object,
    *,
    expected_length: int,
    field_name: str,
) -> list[float]:
    if not isinstance(value, list) or len(value) != expected_length:
        raise RuntimeError(
            f"Geometry Generation Server response field {field_name} is invalid."
        )
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Geometry Generation Server response field {field_name} must be numeric."
        ) from exc


def _image_content_type(image_path: Path) -> str:
    if image_path.suffix.lower() in {".jpg", ".jpeg"}:
        return "image/jpeg"
    return "image/png"


def _load_config() -> dict[str, Any]:
    load_gen_sim_env()
    prefix = "SCENE_ENGINE_GEOMETRY_GENERATION_"
    return {
        "base_url": _read_required_string(f"{prefix}BASE_URL"),
        "timeout_s": _read_positive_int(f"{prefix}TIMEOUT_S"),
        "max_attempts": _read_positive_int(f"{prefix}MAX_ATTEMPTS"),
        "health_path": _read_required_string(f"{prefix}HEALTH_PATH"),
        "generate_objects_path": _read_required_string(f"{prefix}OBJECTS_PATH"),
    }


def _read_required_string(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _read_positive_int(name: str) -> int:
    try:
        value = int(os.getenv(name, ""))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if value < 1:
        raise ValueError(f"{name} must be at least 1.")
    return value
