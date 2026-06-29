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

__all__ = ["DEFAULT_CLIENT_CONFIG_PATH", "load_client_config"]

DEFAULT_CLIENT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "client_config.json"
)


def load_client_config(
    config_key: str,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Load one agent-tool client config section."""
    resolved_config_path = (config_path or DEFAULT_CLIENT_CONFIG_PATH).resolve()
    if not resolved_config_path.is_file():
        raise FileNotFoundError(f"Client config not found: {resolved_config_path}")

    with resolved_config_path.open("r", encoding="utf-8") as f:
        raw_config = json.load(f)

    config = raw_config.get(config_key)
    if not isinstance(config, dict):
        raise ValueError(
            f"Client config section {config_key!r} not found in "
            f"{resolved_config_path}"
        )
    if not config.get("base_url"):
        raise ValueError(f"Client config section {config_key!r} requires base_url.")
    return config
