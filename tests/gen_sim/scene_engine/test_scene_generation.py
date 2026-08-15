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
from scipy.spatial.transform import Rotation

from embodichain.gen_sim.scene_engine.core.scene_graph import (
    SceneGraph,
    SceneGraphNode,
)
from embodichain.gen_sim.scene_engine.pipeline.generation.scene_generation import (
    _scene_graph_based_calibration,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_generation_utils import (
    layout_object_to_transform_matrix,
    transform_matrix_to_layout_object,
)


def _y_up_layout_from_z_up_rotation(
    object_id: str,
    rotation_matrix: np.ndarray,
) -> dict[str, object]:
    y_up_to_z_up_matrix = np.eye(4)
    y_up_to_z_up_matrix[:3, :3] = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    z_up_to_y_up_matrix = np.linalg.inv(y_up_to_z_up_matrix)
    z_up_transform = np.eye(4)
    z_up_transform[:3, :3] = rotation_matrix
    return transform_matrix_to_layout_object(
        object_id,
        z_up_to_y_up_matrix @ z_up_transform @ y_up_to_z_up_matrix,
    )


def _z_up_rotation_from_y_up_layout(layout: dict[str, object]) -> np.ndarray:
    y_up_to_z_up_matrix = np.eye(4)
    y_up_to_z_up_matrix[:3, :3] = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    return (
        y_up_to_z_up_matrix
        @ layout_object_to_transform_matrix(layout)
        @ np.linalg.inv(y_up_to_z_up_matrix)
    )[:3, :3]


def test_scene_graph_calibration_makes_standing_asset_vertical() -> None:
    scene_graph = SceneGraph(
        nodes=[
            SceneGraphNode(object_id="table", parent_id=None),
            SceneGraphNode(
                object_id="bottle_001",
                parent_id="table",
                parent_relation="on",
                orientation_state="standing",
            ),
            SceneGraphNode(
                object_id="book_001",
                parent_id="table",
                parent_relation="on",
            ),
        ]
    )
    lying_rotation = Rotation.from_euler("x", 90.0, degrees=True).as_matrix()
    bottle_layout = _y_up_layout_from_z_up_rotation("bottle_001", lying_rotation)
    book_layout = _y_up_layout_from_z_up_rotation("book_001", lying_rotation)

    calibrated_layouts = _scene_graph_based_calibration(
        scene_graph=scene_graph,
        assets_layout=[bottle_layout, book_layout],
    )

    bottle_axis = _z_up_rotation_from_y_up_layout(calibrated_layouts[0])[:, 2]
    assert np.isclose(abs(bottle_axis[2]), 1.0)
    assert np.allclose(
        _z_up_rotation_from_y_up_layout(calibrated_layouts[1]),
        lying_rotation,
    )
