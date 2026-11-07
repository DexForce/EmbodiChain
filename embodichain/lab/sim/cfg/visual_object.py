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

"""Configuration for an object with rendering and no physical behavior."""

from __future__ import annotations

from dataclasses import field

# Resolve inherited ObjectBaseCfg annotations in runtime type inspection.
import numpy as np

from embodichain.utils import configclass
from ..shapes import ShapeCfg
from .asset import ObjectBaseCfg
from .source import UsdSourceCfg

__all__ = ["VisualObjectCfg"]


@configclass
class VisualObjectCfg(ObjectBaseCfg):
    """Configure a render-only object, without a rigid body or colliders.

    Like :class:`~embodichain.lab.sim.cfg.RigidObjectCfg`, ``MeshCfg.fpath``
    supplies ordinary mesh files such as OBJ, STL, GLB and glTF.
    ``source=UsdSourceCfg(...)`` optionally selects one mesh object from a
    source USD scene. Explicit
    visual-object creation uses only its appearance; USD physics is omitted.
    Whole-scene import instead preserves each entity's authored physical type.
    """

    shape: ShapeCfg | None = None
    """Mesh, cube or sphere supplying the object's appearance."""

    source: UsdSourceCfg | None = field(default=None, kw_only=True)
    """USD source asset; mutually exclusive with :attr:`shape`."""

    body_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Scale applied to the visual geometry, independently of its root pose."""

    visible: bool = True
    """Initial visibility restored by reset."""
