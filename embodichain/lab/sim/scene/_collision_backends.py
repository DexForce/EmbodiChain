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

"""Lazy availability checks for explicitly requested geometry backends."""

from __future__ import annotations

from importlib import import_module

__all__: list[str] = []

_FCL_UNAVAILABLE = (
    "FCL planning geometry is unavailable. Install the optional dependency with "
    "pip install 'embodichain[planning-geometry]' "
    "(qualified python-fcl==0.7.0.11 on Python 3.11/Linux). "
    "No collision result is available without the requested backend."
)


def _require_fcl() -> None:
    """Require usable FCL Python entry points without selecting a substitute."""
    try:
        module = import_module("fcl")
    except (ImportError, OSError) as error:
        raise ImportError(_FCL_UNAVAILABLE) from error
    if not all(
        callable(getattr(module, name, None))
        for name in (
            "DynamicAABBTreeCollisionManager",
            "CollisionObject",
            "CollisionRequest",
            "DistanceRequest",
        )
    ):
        raise ImportError(_FCL_UNAVAILABLE)


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
