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

"""Load the shared GenSim environment configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import MutableMapping

__all__ = ["find_gen_sim_env_file", "load_gen_sim_env"]


def find_gen_sim_env_file() -> Path:
    """Return the configured shared ``.env`` file path.

    ``EMBODICHAIN_ENV_FILE`` is useful for deployments that keep secrets outside
    the source tree. Otherwise GenSim uses ``embodichain/gen_sim/.env``. The
    repository-root ``.env`` remains a backward-compatible fallback.
    """
    configured_path = os.environ.get("EMBODICHAIN_ENV_FILE")
    if configured_path:
        return Path(configured_path).expanduser().resolve()
    default_path = Path(__file__).resolve().parent / ".env"
    if default_path.is_file():
        return default_path


def load_gen_sim_env(env: MutableMapping[str, str] | None = None) -> Path | None:
    """Load missing variables from the shared GenSim ``.env`` file.

    Existing process environment variables are never overwritten so container,
    CI, and shell-provided settings retain precedence over the local file.

    Args:
        env: Environment mapping to populate. Defaults to :data:`os.environ`.

    Returns:
        The loaded path, or ``None`` when no local ``.env`` file exists.

    Raises:
        ValueError: If the file contains an invalid ``KEY=VALUE`` entry.
    """
    target_env = os.environ if env is None else env
    env_path = find_gen_sim_env_file()
    if not env_path.is_file():
        return None

    for line_number, raw_line in enumerate(
        env_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parsed = _parse_env_line(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        if not key.isidentifier():
            raise ValueError(
                f"Invalid environment variable name at {env_path}:{line_number}: {key!r}"
            )
        target_env.setdefault(key, value)
    return env_path


def _parse_env_line(line: str) -> tuple[str, str] | None:
    """Parse one conventional dotenv line without requiring a third-party package."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped.removeprefix("export ").lstrip()
    if "=" not in stripped:
        raise ValueError(f"Expected KEY=VALUE entry, got: {line!r}")

    key, value = stripped.split("=", maxsplit=1)
    key = key.strip()
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    elif " #" in value:
        value = value.split(" #", maxsplit=1)[0].rstrip()
    return key, value
