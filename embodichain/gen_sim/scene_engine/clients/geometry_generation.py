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
from pathlib import Path
from typing import Any

import requests


_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "scene_engine_config.json"
)


class GeometryGenerationClient:
    """Manage the Geometry Generation Server connection."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_s: int,
        max_attempts: int,
        health_path: str,
        generate_multiple_objects_path: str,
        session: requests.Session | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._max_attempts = max_attempts
        self._health_path = health_path
        self._generate_multiple_objects_path = generate_multiple_objects_path
        self._session = session or requests.Session()

    @classmethod
    def from_config(
        cls,
        config_path: str | Path | None = None,
    ) -> "GeometryGenerationClient":
        return cls(**_load_config(config_path))

    def check_health(self) -> None:
        last_error: requests.RequestException | None = None
        for _ in range(self._max_attempts):
            try:
                response = self._session.get(
                    self._url(self._health_path),
                    # timeout=self._timeout_s,
                    timeout=10, # Use a shorter timeout for avoiding long waits.
                )
                response.raise_for_status()
                return
            except requests.RequestException as exc:
                last_error = exc

        assert last_error is not None
        raise RuntimeError(
            "Geometry Generation Server health check failed after "
            f"{self._max_attempts} attempts."
        ) from last_error

    def close(self) -> None:
        self._session.close()

    def generate_multiple_objects(
        self,
        *,
        image_path: str | Path,
        object_masks: list[tuple[str, Path]],
        output_root: str | Path,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Generate multiple objects from:
        - An input image.
        - A list of object masks, each with a unique object_id and a binary mask path.
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

        # Use the wrapped data structure to send the request.
        response_data, response_objects = self._request_multiple_objects(
            image_path=resolved_image_path,
            object_masks=resolved_object_masks,
        )

        resolved_output_root = Path(output_root).expanduser().resolve()

        # This loop will iterate min(len(resolved_object_masks), len(response_objects)) times
        # , which is safe because we validated the lengths earlier.
        for (object_id, _), response_object in zip( # Pair each object_id with its response_object for downloading the glb.
            resolved_object_masks, 
            response_objects,
        ):
            output_path = resolved_output_root / f"{object_id}.glb"
            self._download_glb(response_object["mesh"], output_path)
        return response_data, response_objects

    def _request_multiple_objects(
        self,
        *,
        image_path: Path,
        object_masks: list[tuple[str, Path]],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        last_error: Exception | None = None
        for _ in range(self._max_attempts):
            try:
                with ExitStack() as stack: # This stack manages the context of multiple open files, ensuring they are closed after the request.
                    image_file = stack.enter_context(image_path.open("rb"))
                    mask_files = [
                        stack.enter_context(mask_path.open("rb"))
                        for _, mask_path in object_masks
                    ]
                    response = self._session.post(
                        self._url(self._generate_multiple_objects_path),
                        data={"json": "1"},
                        files=[
                            ("image", (image_path.name, image_file)),
                            *[
                                ("masks", (f"{object_id}.png", mask_file))
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
                response_objects = _parse_multiple_objects_response( # Parse the response.
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


def _parse_multiple_objects_response(
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
    if not isinstance(response_objects, list) or len(response_objects) != len(object_ids):
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


def _load_config(config_path: str | Path | None) -> dict[str, Any]:
    resolved_config_path = Path(config_path or _DEFAULT_CONFIG_PATH).expanduser()
    resolved_config_path = resolved_config_path.resolve()
    if not resolved_config_path.is_file():
        raise FileNotFoundError(f"Config not found: {resolved_config_path}")

    try:
        config_data = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Config is not valid JSON: {resolved_config_path}") from exc

    config = config_data.get("geometry_generation")
    if not isinstance(config, dict):
        raise ValueError("Config key geometry_generation must be an object.")

    required_keys = (
        "base_url",
        "timeout_s",
        "max_attempts",
        "health_path",
        "generate_multiple_objects_path",
    )
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ValueError(f"Missing Geometry Generation Server config keys: {missing}")

    try:
        timeout_s = int(config["timeout_s"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Geometry Generation Server config timeout_s must be an integer."
        ) from exc
    if timeout_s < 1:
        raise ValueError(
            "Geometry Generation Server config timeout_s must be at least 1."
        )

    try:
        max_attempts = int(config["max_attempts"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Geometry Generation Server config max_attempts must be an integer."
        ) from exc
    if max_attempts < 1:
        raise ValueError(
            "Geometry Generation Server config max_attempts must be at least 1."
        )

    string_keys = (
        "base_url",
        "health_path",
        "generate_multiple_objects_path",
    )
    for key in string_keys:
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(
                f"Geometry Generation Server config key {key} must be a non-empty string."
            )

    return {
        "base_url": config["base_url"].strip(),
        "timeout_s": timeout_s,
        "max_attempts": max_attempts,
        "health_path": config["health_path"].strip(),
        "generate_multiple_objects_path": config[
            "generate_multiple_objects_path"
        ].strip(),
    }
