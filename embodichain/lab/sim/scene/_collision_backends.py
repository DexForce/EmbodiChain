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

"""Lazy availability checks for the shared Newton query backend."""

from __future__ import annotations
from importlib import import_module

__all__: list[str] = []


def _require_newton() -> None:
    """Require the Newton runtime without constructing a simulator or query model."""
    try:
        module = import_module("newton")
    except (ImportError, OSError) as error:
        raise ImportError(
            "Newton planning queries require the Newton runtime supplied with the supported DexSim installation."
        ) from error
    if not all(
        callable(getattr(module, name, None))
        for name in ("CollisionPipeline", "ModelBuilder", "Mesh")
    ):
        raise ImportError(
            "The installed Newton runtime lacks the required collision pipeline entry points."
        )
