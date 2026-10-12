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

"""Resolve a Codex model provider without putting credentials in run config."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit


def resolve_provider(
    provider: str, model: str | None, config_file: str | None
) -> tuple[str, dict[str, str] | None]:
    """Return the effective model and transient connection secrets, if needed."""
    if provider not in ("openai", "deepseek"):
        raise ValueError("provider must be 'openai' or 'deepseek'")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError("model must be None or nonempty text")
    if provider == "openai":
        return model or "gpt-6-astra", None
    if not isinstance(config_file, str) or not config_file.strip():
        raise ValueError(
            "DeepSeek requires provider_config / --provider-config pointing to a local JSON credentials file"
        )
    path = Path(config_file).expanduser()
    try:
        settings = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        # Never include the offending file contents in an exception.
        raise ValueError(
            "Could not read DeepSeek provider_config as a JSON file"
        ) from None
    if not isinstance(settings, dict) or not {"base_url", "api_key"} <= settings.keys():
        raise ValueError("DeepSeek provider_config requires base_url and api_key")
    if set(settings) - {"base_url", "api_key", "model"}:
        raise ValueError(
            "DeepSeek provider_config supports only base_url, api_key and model"
        )
    if any(
        not isinstance(value, str) or not value.strip() for value in settings.values()
    ):
        raise ValueError("DeepSeek provider_config fields must be nonempty strings")
    try:
        url = urlsplit(settings["base_url"])
        safe_url = (
            url.scheme == "https"
            and bool(url.hostname)
            and not (url.username or url.password or url.query or url.fragment)
        )
        _ = url.port
    except ValueError:
        safe_url = False
    if not safe_url:
        raise ValueError(
            "DeepSeek base_url must be an HTTPS endpoint without credentials, query or fragment"
        )
    selected = model or settings.get("model", "deepseek-flash")
    # This pipeline always sends images. Known text-only DeepSeek models cannot
    # produce a comparable result, so fail before rendering or spending tokens.
    if selected in ("deepseek-v4-pro", "deepseek-chat", "deepseek-reasoner"):
        raise ValueError("This harness requires a vision model; use deepseek-flash")
    return selected, settings


def prompt_manifest(manifest: dict) -> dict:
    """Expose only mesh evidence to the model, excluding local provider config."""
    fields = (
        "schema_version",
        "mesh_sha256",
        "geometry_sha256",
        "object_description",
        "task_description",
        "target_part",
        "vertex_count",
        "face_count",
        "vertex_index_contract",
        "coordinate_system",
        "render_normalization",
        "patch_coordinates",
        "patches",
        "render_resolution",
        "images",
        "patch_visible_views",
    )
    return {name: manifest[name] for name in fields if name in manifest}
