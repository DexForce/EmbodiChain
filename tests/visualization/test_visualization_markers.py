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


import numpy as np
import pytest

from embodichain.lab.visualization.markers import (
    MarkerGroup,
    MarkerPrototypeCfg,
    VisualizationMarkers,
    VisualizationMarkersCfg,
)


class _Sim:
    def add_marker_group(self, cfg):
        self.group = MarkerGroup(
            cfg, num_envs=3, origins=[[0, 0, 0], [10, 0, 0], [20, 0, 0]]
        )
        return self.group


def _markers(space="world"):
    sim = _Sim()
    marker = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="test",
            coordinate_space=space,
            markers={
                "box": MarkerPrototypeCfg(),
                "sphere": MarkerPrototypeCfg(shape="sphere"),
            },
        ),
        sim,
    )
    return marker, sim.group


@pytest.mark.parametrize(
    "space,expected", [("world", [1, 2, 3]), ("env", [11, 22, 13])]
)
def test_flat_interleaved_environments_and_xyzw(space, expected):
    marker, group = _markers(space)
    marker.visualize(
        [[1, 0, 0], [2, 0, 0], [3, 0, 0]],
        orientations=[[0, 0, 1, 0]] * 3,
        marker_indices=[0, 1, 0],
        environment_ids=[1, 2, 1],
    )
    assert group.counts == (0, 2, 1)
    snaps = group.snapshot()
    np.testing.assert_allclose(
        [s.position[0] for s in snaps], np.array(expected)[[0, 2, 1]]
    )
    np.testing.assert_allclose(snaps[0].wxyz, [0, 0, 0, 1])
    assert [s.env_id for s in snaps] == [1, 1, 2]
    marker.visualize(scales=[[2, 2, 2], [3, 3, 3], [4, 4, 4]])
    np.testing.assert_allclose([s.scale[0] for s in group.snapshot()], [2, 4, 3])
    marker.visualize(marker_indices=[1, 0, 1])
    np.testing.assert_allclose(
        [s.position[0] for s in group.snapshot()], np.array(expected)[[0, 2, 1]]
    )


def test_hidden_resize_clear_and_no_environment_inference():
    marker, group = _markers()
    marker.set_visibility(False)
    marker.visualize(np.zeros((6, 3)))
    assert group.counts == (6, 0, 0)
    assert not any(s.visible for s in group.snapshot())
    marker.set_visibility(True)
    assert all(s.visible for s in group.snapshot())
    marker.visualize(np.empty((0, 3)))
    assert marker.count == 0 and group.snapshot() == ()
    marker.remove()
    marker.remove()
    with pytest.raises(RuntimeError):
        marker.visualize([[0, 0, 0]])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"translations": [[float("nan"), 0, 0]]},
        {"translations": [[0, 0]]},
        {"scales": [[0, 1, 1]]},
        {"orientations": [[0, 0, 0, 0]]},
        {"marker_indices": [3]},
        {"environment_ids": [3]},
        {"environment_ids": [True]},
        {"environment_ids": [1.0]},
        {"environment_ids": [1]},
    ],
)
def test_failed_flat_update_preserves_previous_state(kwargs):
    marker, group = _markers()
    marker.visualize([[1, 2, 3]])
    with pytest.raises(ValueError):
        marker.visualize(**kwargs)
    np.testing.assert_allclose(group.snapshot()[0].position, [1, 2, 3])
    assert marker.count == 1
