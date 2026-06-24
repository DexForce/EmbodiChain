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
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

__all__ = ["ZImageClient", "ZImageClientError"]


class ZImageClientError(RuntimeError):
    """Raised when the z-image service request fails."""


class ZImageClient:
    """HTTP client for the deployed z-image PNG generation service."""

    def __init__(
        self,
        *,
        base_url: str,
        generation_path: str = "/generate.png",
        timeout_s: float = 300.0,
    ):
        """Initialize the z-image client."""
        self.base_url = base_url.strip().rstrip("/")
        if not self.base_url:
            raise ValueError("ZImage base_url must be non-empty.")
        self.generation_path = generation_path
        self.timeout_s = timeout_s
        self._opener = build_opener(ProxyHandler({}))

    def generate_png(
        self,
        *,
        prompt: str,
        output_path: Path,
        width: int = 1024,
        height: int = 1024,
        seed: int = 42,
        num_inference_steps: int = 8,
    ) -> dict[str, Any]:
        """Generate a PNG image and write it to ``output_path``."""
        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "seed": seed,
            "num_inference_steps": num_inference_steps,
        }
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            self._url(self.generation_path),
            data=body,
            headers={
                "Accept": "image/png",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        output_path = output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._opener.open(request, timeout=self.timeout_s) as response:
                content = response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ZImageClientError(
                f"z-image request to {request.full_url} failed with "
                f"HTTP {exc.code}: {detail}"
            ) from exc
        except URLError as exc:
            raise ZImageClientError(
                f"z-image server is unreachable at {request.full_url}: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise ZImageClientError(
                f"z-image request to {request.full_url} timed out after "
                f"{self.timeout_s}s."
            ) from exc

        if not content:
            raise ZImageClientError("z-image server returned an empty image response.")
        output_path.write_bytes(content)
        return {
            "provider": "z-image",
            "base_url": self.base_url,
            "generation_path": self.generation_path,
            "prompt": prompt,
            "width": width,
            "height": height,
            "seed": seed,
            "num_inference_steps": num_inference_steps,
            "output_path": str(output_path),
            "num_bytes": len(content),
        }

    def _url(self, path: str) -> str:
        normalized_path = path if path.startswith("/") else f"/{path}"
        return f"{self.base_url}{normalized_path}"
