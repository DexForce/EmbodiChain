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

import json
import mimetypes
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

try:
    from .schemas import SelectedBox
except ImportError:
    from schemas import SelectedBox

__all__ = ["SAM3Client", "SAM3ClientError"]


class SAM3ClientError(RuntimeError):
    """Raised when the SAM3 segmentation service fails."""


class SAM3Client:
    """Self-contained HTTP client for SAM3 box segmentation."""

    def __init__(
        self,
        *,
        base_url: str,
        boxes_path: str = "/segment_boxes",
        health_path: str = "/health",
        timeout_s: float = 120.0,
        poll_interval_s: float = 2.0,
    ):
        self.base_url = base_url.strip().rstrip("/")
        if not self.base_url:
            raise ValueError("SAM3 base_url must be non-empty.")
        self.boxes_path = boxes_path
        self.health_path = health_path
        self.timeout_s = timeout_s
        self.poll_interval_s = poll_interval_s
        self._opener = build_opener(ProxyHandler({}))

    def health(self) -> dict[str, Any]:
        """Check SAM3 service health."""
        request = Request(
            self._url(self.health_path),
            headers={"Accept": "application/json"},
            method="GET",
        )
        return self._open_json_request(request)

    def segment_boxes_image(
        self,
        image_path: Path,
        *,
        selected_boxes: list[SelectedBox],
        request_id: str | None = None,
        save_visualizations: bool = False,
        progress_path: Path | None = None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Segment an image using box prompts."""
        payload: dict[str, object] = {
            "mode": "box",
            "async": True,
            "selected_boxes": [box.to_manifest() for box in selected_boxes],
            "save_visualizations": save_visualizations,
        }
        if request_id is not None:
            payload["request_id"] = request_id
        result = self._post_multipart_json(
            self.boxes_path,
            payload=payload,
            image_path=image_path,
        )
        result = self._resolve_async_result(
            result,
            progress_path=progress_path,
            verbose=verbose,
        )
        _validate_segmentation_result(result)
        return result

    def _post_multipart_json(
        self,
        path: str,
        *,
        payload: dict[str, object],
        image_path: Path,
    ) -> dict[str, Any]:
        body, content_type = _build_multipart_body(
            payload=payload,
            image_path=image_path,
        )
        request = Request(
            self._url(path),
            data=body,
            headers={
                "Accept": "application/json",
                "Content-Type": content_type,
            },
            method="POST",
        )
        return self._open_json_request(request)

    def _open_json_request(self, request: Request) -> dict[str, Any]:
        try:
            with self._opener.open(request, timeout=self.timeout_s) as response:
                response_body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise SAM3ClientError(
                f"SAM3 request to {request.full_url} failed with "
                f"HTTP {exc.code}: {detail}"
            ) from exc
        except URLError as exc:
            raise SAM3ClientError(
                f"SAM3 server is unreachable at {request.full_url}: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise SAM3ClientError(
                f"SAM3 request to {request.full_url} timed out after "
                f"{self.timeout_s}s."
            ) from exc

        try:
            decoded = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise SAM3ClientError(
                f"SAM3 server returned non-JSON: {response_body}"
            ) from exc
        if not isinstance(decoded, dict):
            raise SAM3ClientError("SAM3 response must be a JSON object.")
        return decoded

    def _resolve_async_result(
        self,
        result: dict[str, Any],
        *,
        progress_path: Path | None,
        verbose: bool,
    ) -> dict[str, Any]:
        status = str(result.get("status") or "").lower()
        status_url = result.get("status_url")
        if status not in {"queued", "running"} or not isinstance(status_url, str):
            _append_progress(progress_path, result)
            _print_progress("segmentation", result, verbose=verbose)
            return result

        _append_progress(progress_path, result)
        _print_progress("segmentation", result, verbose=verbose)
        deadline = time.monotonic() + self.timeout_s
        while True:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                raise SAM3ClientError(
                    f"SAM3 async job timed out after {self.timeout_s}s: {result}"
                )
            time.sleep(min(self.poll_interval_s, remaining_s))
            job = self._get_json(status_url)
            _append_progress(progress_path, job)
            _print_progress("segmentation", job, verbose=verbose)
            job_status = str(job.get("status") or "").lower()
            if job_status in {"queued", "running"}:
                continue
            if job_status == "succeeded":
                final_result = job.get("result")
                if not isinstance(final_result, dict):
                    raise SAM3ClientError("SAM3 async job succeeded without result.")
                return final_result
            if job_status == "failed":
                raise SAM3ClientError(f"SAM3 async job failed: {job}")
            raise SAM3ClientError(f"SAM3 async job returned unknown status: {job}")

    def _get_json(self, path: str) -> dict[str, Any]:
        request = Request(
            self._url(path),
            headers={"Accept": "application/json"},
            method="GET",
        )
        return self._open_json_request(request)

    def _url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        normalized_path = path if path.startswith("/") else f"/{path}"
        return f"{self.base_url}{normalized_path}"


def _build_multipart_body(
    *,
    payload: dict[str, object],
    image_path: Path,
) -> tuple[bytes, str]:
    image_path = image_path.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image upload path is not a file: {image_path}")

    boundary = f"----prompt2geometry-sam3-{uuid.uuid4().hex}"
    content_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    chunks = [
        f"--{boundary}\r\n".encode("utf-8"),
        b'Content-Disposition: form-data; name="payload"\r\n',
        b"Content-Type: application/json\r\n\r\n",
        json.dumps(payload).encode("utf-8"),
        b"\r\n",
        f"--{boundary}\r\n".encode("utf-8"),
        (
            'Content-Disposition: form-data; name="image"; '
            f'filename="{image_path.name}"\r\n'
        ).encode("utf-8"),
        f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
        image_path.read_bytes(),
        b"\r\n",
        f"--{boundary}--\r\n".encode("utf-8"),
    ]
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _append_progress(progress_path: Path | None, payload: dict[str, Any]) -> None:
    if progress_path is None:
        return
    progress_path = progress_path.expanduser().resolve()
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _print_progress(stage: str, payload: dict[str, Any], *, verbose: bool) -> None:
    if not verbose:
        return
    status = payload.get("status") or payload.get("ok") or "unknown"
    job_id = payload.get("job_id") or payload.get("request_id") or payload.get("id")
    progress = payload.get("progress")
    parts = [f"[{stage}] status={status}"]
    if job_id is not None:
        parts.append(f"job={job_id}")
    if progress is not None:
        parts.append(f"progress={progress}")
    print(" ".join(parts), flush=True)


def _validate_segmentation_result(result: dict[str, Any]) -> None:
    if result.get("ok") is not True:
        raise SAM3ClientError(f"SAM3 segmentation failed: {result}")
    segmentations = result.get("segmentations")
    if not isinstance(segmentations, list):
        raise SAM3ClientError("SAM3 response missing segmentations list.")
    for index, segmentation in enumerate(segmentations):
        if not isinstance(segmentation, dict):
            raise SAM3ClientError(f"SAM3 segmentation {index} must be an object.")
        target_id = segmentation.get("target_id")
        if not isinstance(target_id, str) or not target_id.strip():
            raise SAM3ClientError(f"SAM3 segmentation {index} must contain target_id.")
