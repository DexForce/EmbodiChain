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

from embodichain.lab.gym.envs.managers import FunctorCfg
from embodichain.lab.gym.envs.managers.record import record_camera_data
from embodichain.lab.sim.sensors import CameraCfg


class _CameraHandle:
    group_id = 7


class _Simulation:
    def __init__(self) -> None:
        self.camera = _CameraHandle()
        self.sensor_cfg: CameraCfg | None = None

    def add_sensor(self, sensor_cfg: CameraCfg) -> _CameraHandle:
        self.sensor_cfg = sensor_cfg
        return self.camera


class _Environment:
    def __init__(self) -> None:
        self.sim = _Simulation()
        self.camera_group_ids: list[int] = []

    def add_camera_group_id(self, group_id: int) -> None:
        self.camera_group_ids.append(group_id)


def test_record_camera_is_marked_for_visualization() -> None:
    env = _Environment()
    cfg = FunctorCfg(func=record_camera_data, params={"name": "episode_camera"})

    record_camera_data(cfg, env)

    assert env.sim.sensor_cfg is not None
    assert env.sim.sensor_cfg.visualization_role == "record"
