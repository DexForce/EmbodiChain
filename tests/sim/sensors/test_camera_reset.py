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

from collections.abc import Sequence

import numpy as np
import pytest
import torch

from embodichain.lab.sim.sensors.camera import Camera, CameraCfg

pytestmark = pytest.mark.no_sim


class _View:
    def __init__(self) -> None:
        self.pose = np.zeros((4, 4), dtype=np.float32)

    def set_local_pose(self, pose: np.ndarray) -> None:
        self.pose = pose.copy()


@pytest.mark.parametrize("extrinsics_kind", ["pose", "look_at", "look_at_default_up"])
@pytest.mark.parametrize(
    "env_ids",
    [None, [1], (1,), (2, 0), (), [], torch.tensor([2, 0])],
    ids=[
        "all",
        "list",
        "tuple_one",
        "tuple_ordered",
        "tuple_empty",
        "list_empty",
        "tensor",
    ],
)
def test_reset_preserves_batch_dimension_and_unselected_views(
    extrinsics_kind: str, env_ids: Sequence[int] | torch.Tensor | None
) -> None:
    """Sequence indices select camera rows for both extrinsics representations."""
    if extrinsics_kind == "pose":
        extrinsics = CameraCfg.ExtrinsicsCfg(pos=(1.0, 2.0, 3.0))
    else:
        extrinsics = CameraCfg.ExtrinsicsCfg(
            eye=(1.0, 2.0, 3.0),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 0.0, 1.0) if extrinsics_kind == "look_at" else None,
        )
    camera = Camera.__new__(Camera)
    camera.cfg = CameraCfg(uid="camera", extrinsics=extrinsics)
    camera.device = torch.device("cpu")
    camera._entities = [_View() for _ in range(3)]
    camera._num_instances = len(camera._entities)
    camera.reset(env_ids)

    expected = camera.cfg.extrinsics.transformation.clone()
    expected[:3, 1:3].neg_()
    selected = set(range(3) if env_ids is None else map(int, env_ids))
    for row, view in enumerate(camera._entities):
        torch.testing.assert_close(
            torch.from_numpy(view.pose),
            expected if row in selected else torch.zeros((4, 4)),
        )
