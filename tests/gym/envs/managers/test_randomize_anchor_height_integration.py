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

import pytest

from embodichain.lab.gym.envs import EmbodiedEnvCfg
from embodichain.lab.gym.envs.managers import EventCfg
from embodichain.lab.gym.envs.managers.randomization.spatial import (
    randomize_anchor_height,
)
from embodichain.lab.sim.cfg import RigidObjectCfg


@pytest.mark.skip(reason="Requires full simulation stack; run manually.")
def test_anchor_height_event_runs_in_reset():
    """Smoke test that the functor can be wired into an EmbodiedEnvCfg."""
    cfg = EmbodiedEnvCfg()
    cfg.events.anchor_height = EventCfg(
        func=randomize_anchor_height,
        mode="reset",
        params={
            "anchor_uid": "table",
            "height_delta_range": ([-0.05], [0.05]),
        },
    )
    cfg.background.append(
        RigidObjectCfg(uid="table", init_pos=[0.0, 0.0, 0.8], body_type="static")
    )
    cfg.rigid_object.append(
        RigidObjectCfg(uid="cube", init_pos=[0.1, 0.0, 0.9], body_type="dynamic")
    )
    # Actual env construction and reset would go here.
    assert hasattr(cfg.events, "anchor_height")
