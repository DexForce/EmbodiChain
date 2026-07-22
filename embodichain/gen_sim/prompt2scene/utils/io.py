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

import base64
import json
from pathlib import Path
from typing import Any

from embodichain.gen_sim.prompt2scene.utils.log import log_info

__all__ = ["image_to_data_url", "relative_path", "write_json"]


def relative_path(path: str | Path, root: Path) -> str:
    """Return ``path`` relative to ``root`` when it is contained by it."""
    resolved_path = Path(path)
    try:
        return str(resolved_path.relative_to(root))
    except ValueError:
        return str(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload with prompt2scene's default formatting.

    Args:
        path: Output JSON file path.
        payload: JSON-serializable dictionary payload.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if not path.is_file():
        raise FileNotFoundError(f"JSON output was not written: {path}")
    log_info(f"Wrote JSON: {path}")


def image_to_data_url(image_path: Path) -> str:
    """Return a base64 data URL for a local image file."""
    suffix_to_mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    mime_type = suffix_to_mime.get(image_path.suffix.lower(), "image/png")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
