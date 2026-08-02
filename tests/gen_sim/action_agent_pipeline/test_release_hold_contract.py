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

from types import SimpleNamespace

import torch

from embodichain.gen_sim.action_agent_pipeline.runtime.action_execution import (
    _external_post_hold_steps,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.action_parts import (
    _build_action_cfg,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.atomic_action_spec import (
    AtomicActionSpec,
)


def _place_spec() -> AtomicActionSpec:
    return AtomicActionSpec.from_mapping(
        {
            "atomic_action_class": "Place",
            "robot_name": "left_arm",
            "control": "arm",
            "target_pose": {
                "reference": "relative",
                "offset": [0.0, 0.0, 0.0],
                "frame": "world",
            },
            "cfg": {
                "sample_interval": 64,
                "hand_interp_steps": 12,
                "post_hold_steps": 12,
            },
        }
    )


def test_place_holds_at_open_gripper_pose_before_retract() -> None:
    spec = _place_spec()
    env = SimpleNamespace(
        robot=SimpleNamespace(device=torch.device("cpu")),
        open_state=torch.tensor([0.0, 0.0]),
        close_state=torch.tensor([0.7, -0.7]),
    )

    cfg = _build_action_cfg(env, spec, "arm", "hand", 2)

    assert cfg.post_hold_steps == 12
    assert _external_post_hold_steps(spec) == 0
