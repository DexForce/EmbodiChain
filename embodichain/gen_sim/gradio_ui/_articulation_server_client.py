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

"""Dependency-free HTTP client for the remote articulation service."""

from __future__ import annotations

import json
import mimetypes
import shutil
import uuid
from math import isfinite
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import OpenerDirector, ProxyHandler, Request, build_opener

__all__ = ["ArticulationServerClient", "ArticulationServerError"]


class ArticulationServerError(RuntimeError):
    """Report a request or response error from the articulation service."""


class ArticulationServerClient:
    """Call the articulation-server HTTP API."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 30.0,
    ) -> None:
        """Initialize the client.

        Args:
            base_url: Absolute HTTP(S) URL for the articulation service.
            timeout_seconds: Per-request network timeout.

        Raises:
            ValueError: If the URL or timeout is invalid.
        """
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain a query or fragment")
        if not isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a finite positive number")
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout_seconds = timeout_seconds
        self.opener: OpenerDirector = build_opener(ProxyHandler({}))

    def health(self) -> dict[str, Any]:
        """Return server health details."""
        return self._json("GET", "/health")

    def submit(self, prompt: str, *, image: str | Path | None = None) -> dict[str, Any]:
        """Submit an articulation generation task."""
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt must not be empty")
        if image is None:
            body = json.dumps({"prompt": prompt}).encode("utf-8")
            return self._json(
                "POST",
                "/generate_articulation",
                body=body,
                headers={"Content-Type": "application/json"},
            )

        image_path = Path(image).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"reference image not found: {image_path}")
        body, content_type = _multipart_body(prompt, image_path)
        return self._json(
            "POST",
            "/generate_articulation",
            body=body,
            headers={"Content-Type": content_type},
        )

    def status(self, request_id: str) -> dict[str, Any]:
        """Return the current state of a generation task."""
        return self._json("GET", f"/tasks/{_request_id(request_id)}")

    def cancel(self, request_id: str) -> dict[str, Any]:
        """Request cancellation of a generation task."""
        return self._json("POST", f"/tasks/{_request_id(request_id)}/cancel", body=b"")

    def download(
        self,
        request_id: str,
        artifact: str,
        destination: str | Path,
    ) -> Path:
        """Atomically download one artifact from a completed task."""
        task = self.status(request_id)
        result = task.get("result")
        artifacts = result.get("artifacts") if isinstance(result, dict) else None
        relative_url = artifacts.get(artifact) if isinstance(artifacts, dict) else None
        if not isinstance(relative_url, str):
            raise ArticulationServerError(
                f"task {request_id} has no artifact named {artifact!r}"
            )

        target = Path(destination).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.part")
        try:
            with (
                self._open("GET", relative_url) as response,
                temporary.open("wb") as output,
            ):
                shutil.copyfileobj(response, output)
            temporary.replace(target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return target

    def _json(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        with self._open(method, path, body=body, headers=headers) as response:
            raw = response.read()
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArticulationServerError("server response is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ArticulationServerError("server response must be a JSON object")
        return payload

    def _open(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        request_headers = {"Accept": "application/json", **(headers or {})}
        request = Request(
            self._url(path),
            data=body,
            headers=request_headers,
            method=method,
        )
        try:
            return self.opener.open(request, timeout=self.timeout_seconds)
        except HTTPError as exc:
            detail = _error_detail(exc.read())
            raise ArticulationServerError(
                f"server returned HTTP {exc.code}: {detail or exc.reason}"
            ) from exc
        except URLError as exc:
            raise ArticulationServerError(
                f"could not reach articulation-server: {exc.reason}"
            ) from exc

    def _url(self, path: str) -> str:
        parsed = urlsplit(path)
        if parsed.scheme or parsed.netloc:
            raise ArticulationServerError("server artifact URL must be relative")
        return urljoin(self.base_url, path.lstrip("/"))


def _request_id(value: str) -> str:
    value = value.strip()
    if not value or any(
        character not in "0123456789abcdef-" for character in value.lower()
    ):
        raise ValueError("request_id contains unsupported characters")
    return value


def _multipart_body(prompt: str, image_path: Path) -> tuple[bytes, str]:
    boundary = f"articulation-{uuid.uuid4().hex}"
    image_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    safe_name = image_path.name.replace('"', "_").replace("\r", "_").replace("\n", "_")
    parts = [
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="prompt"\r\n\r\n',
        prompt.encode("utf-8"),
        b"\r\n",
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="image"; filename="{safe_name}"\r\n'.encode(),
        f"Content-Type: {image_type}\r\n\r\n".encode(),
        image_path.read_bytes(),
        b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ]
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _error_detail(raw: bytes) -> str:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return raw.decode("utf-8", errors="replace")[:500]
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("error") or payload.get("detail") or "")
