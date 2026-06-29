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

from pathlib import Path

import requests

from embodichain.gen_sim.prompt2scene.agent_tools.clients.geometry_generation_client.schemas import (
    GeometryGenerationResult,
    GeometryGenerationServerRequest,
    GeometryGenerationServerResponse,
    MultiObjectGenerationObject,
    MultiObjectGenerationResult,
    MultiObjectGenerationServerResponse,
)
from embodichain.gen_sim.prompt2scene.utils.log import log_info

__all__ = ["parse_geometry_generation_response", "parse_multi_object_generation_response"]


def parse_geometry_generation_response(
    response: requests.Response,
    request: GeometryGenerationServerRequest,
) -> GeometryGenerationServerResponse:
    """Parse a geometry GLB response and save it to the request output path."""
    glb_bytes = response.content
    _validate_glb_response(response, glb_bytes)
    output_path = _write_glb_output(request, glb_bytes)
    result = GeometryGenerationResult(geometry_path=str(output_path))
    return GeometryGenerationServerResponse(
        ok=True,
        status="ok",
        result=result,
        status_code=response.status_code,
        content_type=response.headers.get("Content-Type"),
        headers=dict(response.headers),
    )


def _validate_glb_response(
    response: requests.Response,
    glb_bytes: bytes,
) -> None:
    if not glb_bytes.startswith(b"glTF"):
        content_type = response.headers.get("Content-Type", "")
        raise RuntimeError(
            "Geometry generation server returned invalid GLB content: "
            f"{content_type or 'unknown'}"
        )


def _write_glb_output(
    request: GeometryGenerationServerRequest,
    glb_bytes: bytes,
) -> Path:
    output_path = Path(request.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(glb_bytes)
    if not output_path.is_file():
        raise FileNotFoundError(f"Generated geometry was not written: {output_path}")
    log_info(f"Generated geometry written: {output_path}")
    return output_path


def parse_multi_object_generation_response(
    response: requests.Response,
    base_url: str,
    *,
    output_dir: Path | None = None,
    session: requests.Session | None = None,
) -> MultiObjectGenerationServerResponse:
    """Parse a multi-object geometry response, download GLBs if output_dir given."""
    body = _parse_json_body(response)
    ok = body.get("ok", False)
    if not isinstance(ok, bool) or not ok:
        error_msg = body.get("error", "ok is not true")
        raise RuntimeError(
            f"Multi-object geometry generation failed: {error_msg}"
        )

    result_data = body.get("result")
    if not isinstance(result_data, dict):
        raise RuntimeError(
            "Multi-object geometry generation response missing 'result' object"
        )
    base = base_url.rstrip("/")
    objects = _parse_multi_object_items(
        result_data,
        base,
        output_dir=output_dir,
        session=session,
    )

    return MultiObjectGenerationServerResponse(
        ok=True,
        status=str(body.get("status") or "ok"),
        result=MultiObjectGenerationResult(objects=objects),
        status_code=response.status_code,
        content_type=response.headers.get("Content-Type"),
        headers=dict(response.headers),
    )


def _parse_multi_object_items(
    body: dict[str, object],
    base_url: str,
    *,
    output_dir: Path | None,
    session: requests.Session | None,
) -> list[MultiObjectGenerationObject]:
    response_objects = body.get("objects")
    if not isinstance(response_objects, list) or not response_objects:
        raise RuntimeError(
            "Multi-object geometry generation response missing 'result.objects' list"
        )
    return [
        _parse_multi_object_item(
            item,
            index=i,
            base_url=base_url,
            output_dir=output_dir,
            session=session,
        )
        for i, item in enumerate(response_objects)
    ]


def _parse_multi_object_item(
    item: object,
    *,
    index: int,
    base_url: str,
    output_dir: Path | None,
    session: requests.Session | None,
) -> MultiObjectGenerationObject:
    if not isinstance(item, dict):
        raise RuntimeError(f"Multi-object item {index} must be a JSON object")

    mesh_rel_path = item.get("mesh")
    if not isinstance(mesh_rel_path, str) or not mesh_rel_path:
        raise RuntimeError(f"Multi-object item {index} missing 'mesh'")

    name = str(item.get("name") or Path(mesh_rel_path).stem or index)
    geometry_path = _resolve_or_download_glb(
        base_url,
        mesh_rel_path,
        name=name,
        index=index,
        output_dir=output_dir,
        session=session,
    )

    return MultiObjectGenerationObject(
        name=name,
        geometry_path=geometry_path,
        rotation_quaternion_wxyz=_float_list(
            item.get("rotation_quaternion_wxyz"),
            expected_len=4,
            field_name=f"objects[{index}].rotation_quaternion_wxyz",
        ),
        translation=_float_list(
            item.get("translation"),
            expected_len=3,
            field_name=f"objects[{index}].translation",
        ),
        scale=_float_list(
            item.get("scale"),
            expected_len=3,
            field_name=f"objects[{index}].scale",
        ),
    )


def _resolve_or_download_glb(
    base_url: str,
    mesh_rel_path: str,
    *,
    name: str,
    index: int,
    output_dir: Path | None,
    session: requests.Session | None,
) -> str:
    url = _join_url(base_url, mesh_rel_path)
    if output_dir is None:
        return url

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{name}.glb" if name else f"{index}.glb"
    dest = output_dir / filename
    _download_glb(url, dest, session=session)
    return str(dest)


def _join_url(base_url: str, path_or_url: str) -> str:
    if path_or_url.startswith(("http://", "https://")):
        return path_or_url
    if path_or_url.startswith("/"):
        return f"{base_url}{path_or_url}"
    return f"{base_url}/{path_or_url}"


def _float_list(value: object, *, expected_len: int, field_name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != expected_len:
        raise RuntimeError(f"Multi-object geometry response missing '{field_name}'")
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Multi-object geometry response field '{field_name}' must be numeric"
        ) from exc


def _parse_json_body(response: requests.Response) -> dict[str, object]:
    try:
        body = response.json()
    except ValueError as exc:
        raise RuntimeError(
            "Multi-object geometry generation server returned invalid JSON"
        ) from exc
    if not isinstance(body, dict):
        raise RuntimeError(
            "Multi-object geometry generation response must be a JSON object"
        )
    return body


def _download_glb(
    url: str,
    dest: Path,
    *,
    session: requests.Session | None,
) -> None:
    """Download a GLB from the geometry server."""
    http = session or requests.Session()
    r = http.get(url, timeout=30)
    r.raise_for_status()
    _validate_glb_response(r, r.content)
    dest.write_bytes(r.content)
    log_info(f"Generated geometry written: {dest}")
