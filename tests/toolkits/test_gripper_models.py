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

"""Public aperture model declarations remain independent of robot geometry."""

from __future__ import annotations

import pytest


def test_public_model_resolution_preserves_named_defaults_and_closed_inline_bounds():
    from embodichain.toolkits.graspkit import resolve_parallel_jaw_gripper_model

    native = resolve_parallel_jaw_gripper_model("tianji_marvin_gripper")
    assert (native.min_opening_width, native.max_opening_width) == (0.001, 0.095)
    inline = dict(model_id="robotiq_arg2f_140", max_opening_width=0.128)
    capped = resolve_parallel_jaw_gripper_model(inline)
    assert (capped.min_opening_width, capped.max_opening_width) == (0.002, 0.128)
    capped.max_opening_width = 0.1
    assert inline["max_opening_width"] == 0.128
    with pytest.raises(ValueError, match="unsupported fields"):
        resolve_parallel_jaw_gripper_model({**inline, "collision_box": [1, 1, 1]})
    with pytest.raises(ValueError, match="unknown parallel-jaw"):
        resolve_parallel_jaw_gripper_model("unknown_hand")
