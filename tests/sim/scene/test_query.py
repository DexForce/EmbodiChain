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
from embodichain.lab.sim.scene.query import SceneView
from ._runtime_ports import Sim


@pytest.mark.parametrize("env_ids", [(1,), (0, 0), (), (False,)])
def test_explicit_unsupported_selection_rejected(env_ids):
    with pytest.raises(ValueError, match="B=1"):
        SceneView(Sim(), env_ids=env_ids)


def test_batch_world_rejected_before_entity_reads():
    sim = Sim()
    sim.num_envs = 2
    with pytest.raises(ValueError, match="B=1"):
        SceneView(sim)


def test_pose_is_owned_and_keeps_environment_dimension():
    sim = Sim()
    view = SceneView(sim)
    pose = view.pose("cube", to_matrix=False)
    assert pose.shape == (1, 7)
    pose.zero_()
    assert sim.rigid["cube"].value[0, 0] != 0
    with pytest.raises(KeyError, match="Unknown"):
        view.pose("missing")
