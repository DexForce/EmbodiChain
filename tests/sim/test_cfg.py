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
from scipy.spatial.transform import Rotation

from embodichain.lab.sim.cfg import ArticulationCfg, ObjectBaseCfg

_MULTI_AXIS_ROTATION_DEGREES = [31.0, -47.0, 123.0]
_TRANSLATION = [0.2, -0.3, 0.4]


@pytest.mark.parametrize("cfg_cls", [ObjectBaseCfg, ArticulationCfg])
def test_pose_cfg_from_dict_uses_intrinsic_xyz_euler(cfg_cls: type) -> None:
    cfg = cfg_cls.from_dict(
        {
            "init_pos": _TRANSLATION,
            "init_rot": _MULTI_AXIS_ROTATION_DEGREES,
        }
    )

    expected_rotation = Rotation.from_euler(
        "XYZ",
        _MULTI_AXIS_ROTATION_DEGREES,
        degrees=True,
    ).as_matrix()
    assert cfg.init_local_pose[:3, :3] == pytest.approx(expected_rotation)
    assert cfg.init_local_pose[:3, 3] == pytest.approx(_TRANSLATION)


@pytest.mark.parametrize("cfg_cls", [ObjectBaseCfg, ArticulationCfg])
def test_pose_cfg_from_dict_decodes_intrinsic_xyz_matrix(cfg_cls: type) -> None:
    init_local_pose = np.eye(4)
    init_local_pose[:3, :3] = Rotation.from_euler(
        "XYZ",
        _MULTI_AXIS_ROTATION_DEGREES,
        degrees=True,
    ).as_matrix()
    init_local_pose[:3, 3] = _TRANSLATION

    cfg = cfg_cls.from_dict({"init_local_pose": init_local_pose})

    reconstructed = Rotation.from_euler(
        "XYZ",
        cfg.init_rot,
        degrees=True,
    ).as_matrix()
    assert reconstructed == pytest.approx(init_local_pose[:3, :3])
    assert cfg.init_pos == pytest.approx(_TRANSLATION)
