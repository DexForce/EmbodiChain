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

import re

_SNAKE_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def snake_to_camel(snake: str) -> str:
    """Convert ``pick_place`` to ``PickPlace``."""
    return "".join(part.capitalize() for part in snake.split("_") if part)


def default_env_class(snake: str) -> str:
    """Convert ``pick_place`` to ``PickPlaceEnv``."""
    return f"{snake_to_camel(snake)}Env"


def default_gym_id(snake: str, *, version: str = "v1") -> str:
    """Default gym id from task snake name."""
    return f"{snake_to_camel(snake)}-{version}"


def validate_snake(name: str) -> str:
    if not _SNAKE_RE.match(name):
        raise ValueError(
            f"Task name must be snake_case (e.g. pick_place), got: {name!r}"
        )
    return name


def validate_gym_id(gym_id: str) -> str:
    if not gym_id or " " in gym_id:
        raise ValueError(f"Invalid gym id: {gym_id!r}")
    return gym_id


def validate_package_name(name: str) -> str:
    if not re.match(r"^[a-z][a-z0-9_]*$", name):
        raise ValueError(
            f"Package name must be a valid Python module name, got: {name!r}"
        )
    return name
