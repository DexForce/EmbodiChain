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

"""Tests for atomic action core module (ObjectSemantics, ActionCfg)."""

from __future__ import annotations

import pytest
import torch

from embodichain.lab.sim.atomic_actions.affordance import Affordance
from embodichain.lab.sim.atomic_actions.core import (
    ActionCfg,
    ObjectSemantics,
)

# ---------------------------------------------------------------------------
# ObjectSemantics
# ---------------------------------------------------------------------------


class TestObjectSemantics:
    """Tests for ObjectSemantics dataclass."""

    def test_post_init_binds_label(self):
        geometry = {"bounding_box": [0.1, 0.2, 0.3]}
        aff = Affordance()
        sem = ObjectSemantics(
            affordance=aff,
            geometry=geometry,
            label="mug",
        )
        assert sem.affordance.object_label == "mug"

    def test_default_optional_fields(self):
        sem = ObjectSemantics(
            affordance=Affordance(),
            geometry={},
        )
        assert sem.label == "none"
        assert sem.properties == {}
        assert sem.entity is None


# ---------------------------------------------------------------------------
# ActionCfg
# ---------------------------------------------------------------------------


class TestActionCfg:
    """Tests for ActionCfg defaults."""

    def test_default_values(self):
        cfg = ActionCfg()
        assert cfg.name == "default"
        assert cfg.control_part == "arm"
        assert cfg.interpolation_type == "linear"
        assert cfg.velocity_limit is None
        assert cfg.acceleration_limit is None
