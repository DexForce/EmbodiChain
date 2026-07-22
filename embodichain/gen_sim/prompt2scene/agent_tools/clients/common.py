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

from dataclasses import dataclass, field
from typing import Any

import requests

__all__ = [
    "ClientError",
    "build_client_error",
    "first_string",
    "format_http_error",
    "parse_error_response",
    "parse_json_object_response",
    "validate_required_strings",
    "validate_png_response",
]


@dataclass(frozen=True)
class ClientError:
    """Common HTTP client error response."""

    error_message: str
    status_code: int | None = None
    content_type: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    raw_response: dict[str, Any] | None = None


def validate_png_response(
    response: requests.Response,
    png_bytes: bytes,
) -> None:
    content_type = response.headers.get("Content-Type", "")
    if "image/png" not in content_type.lower():
        raise RuntimeError(
            "Image generation server returned non-PNG content: "
            f"{content_type or 'unknown'}"
        )
    if not png_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError("Image generation server returned invalid PNG bytes.")


def validate_required_strings(fields: dict[str, object]) -> None:
    """Validate required client request string fields."""
    for field_name, value in fields.items():
        if not str(value).strip():
            raise ValueError(f"{field_name} must be non-empty.")


def format_http_error(response: requests.Response, *, server_name: str) -> str:
    """Format an HTTP error response from an agent-tool server."""
    try:
        response_data = response.json()
    except ValueError:
        return f"{server_name} HTTP error: {response.status_code}"

    error_message = first_string(
        response_data,
        "error",
        "error_message",
        "message",
        "detail",
    )
    if error_message:
        return f"{server_name} error: {error_message}"
    return f"{server_name} HTTP error: {response.status_code}"


def parse_error_response(response: requests.Response) -> dict[str, Any] | None:
    """Parse an error response body as a JSON object if possible."""
    try:
        response_data = response.json()
    except ValueError:
        return None
    return response_data if isinstance(response_data, dict) else None


def build_client_error(
    response: requests.Response,
    *,
    server_name: str,
    error_cls: type[ClientError] = ClientError,
) -> ClientError:
    """Build a common client error dataclass from an HTTP response."""
    return error_cls(
        error_message=format_http_error(
            response,
            server_name=server_name,
        ),
        status_code=response.status_code,
        content_type=response.headers.get("Content-Type"),
        headers=dict(response.headers),
        raw_response=parse_error_response(response),
    )


def parse_json_object_response(
    response: requests.Response,
    *,
    server_name: str,
) -> dict[str, Any]:
    """Parse an HTTP response body as a JSON object."""
    try:
        response_data = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"{server_name} returned invalid JSON content: "
            f"{response.headers.get('Content-Type') or 'unknown'}"
        ) from exc
    if not isinstance(response_data, dict):
        raise RuntimeError(f"{server_name} response must be a JSON object.")
    return response_data


def first_string(data: dict[str, Any], *keys: str) -> str | None:
    """Return the first string value for the given keys."""
    for key in keys:
        value = data.get(key)
        if isinstance(value, str):
            return value
    return None
