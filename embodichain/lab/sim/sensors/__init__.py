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

"""Sensors attached to the simulation scene.

``Camera``, ``StereoCamera``, and ``ContactSensor`` all derive from ``BaseSensor`` and expose batched ``TensorDict`` data buffers.
"""

from __future__ import annotations

from .base_sensor import BaseSensor, SensorCfg
from .context import SensorClock, SensorStateProvider, SensorContext
from .camera import Camera, CameraCfg
from .state import JointEncoder, JointEncoderCfg, FrameTransform, FrameTransformCfg
from .inertial import PVA, PVACfg, IMU, IMUCfg
from .stereo import StereoCamera, StereoCameraCfg
from .contact import (
    ContactSensor,
    ContactSensorCfg,
    ArticulationContactFilterCfg,
)
from .contact_grid import ContactGrid, ContactGridCfg
from .contact_resultant import ContactResultant, ContactResultantCfg
from .ray import (
    RayScene,
    TerrainScan,
    TerrainScanCfg,
    RayLidar,
    RayLidarCfg,
    RayDepthCamera,
    RayDepthCameraCfg,
)
from .registry import (
    SensorRegistration,
    register_sensor,
    discover_sensor_plugins,
    get_sensor_registration,
    list_sensor_types,
)

__all__ = [
    "BaseSensor",
    "RayLidar",
    "RayLidarCfg",
    "RayDepthCamera",
    "RayDepthCameraCfg",
    "RayScene",
    "TerrainScan",
    "TerrainScanCfg",
    "ContactGrid",
    "ContactGridCfg",
    "ContactResultant",
    "ContactResultantCfg",
    "PVA",
    "PVACfg",
    "IMU",
    "IMUCfg",
    "JointEncoder",
    "JointEncoderCfg",
    "FrameTransform",
    "FrameTransformCfg",
    "SensorCfg",
    "SensorClock",
    "SensorStateProvider",
    "SensorContext",
    "SensorRegistration",
    "register_sensor",
    "discover_sensor_plugins",
    "get_sensor_registration",
    "list_sensor_types",
    "Camera",
    "CameraCfg",
    "StereoCamera",
    "StereoCameraCfg",
    "ContactSensor",
    "ContactSensorCfg",
    "ArticulationContactFilterCfg",
]

# Built-ins use the same registry as plugins. Their direct constructors and
# manager-owned render/contact lifecycle remain compatible.
_builtin_sensor_factories = {}
for _name, _config, _factory in (
    ("Camera", CameraCfg, Camera),
    ("StereoCamera", StereoCameraCfg, StereoCamera),
    ("ContactSensor", ContactSensorCfg, ContactSensor),
):
    register_sensor(SensorRegistration(_name, _config, _factory, legacy_factory=True))
    _builtin_sensor_factories[_name] = _factory

register_sensor(SensorRegistration("JointEncoder", JointEncoderCfg, JointEncoder))
register_sensor(SensorRegistration("FrameTransform", FrameTransformCfg, FrameTransform))

register_sensor(SensorRegistration("PVA", PVACfg, PVA))
register_sensor(SensorRegistration("IMU", IMUCfg, IMU))

register_sensor(SensorRegistration("ContactGrid", ContactGridCfg, ContactGrid))

register_sensor(
    SensorRegistration("ContactResultant", ContactResultantCfg, ContactResultant)
)

register_sensor(SensorRegistration("TerrainScan", TerrainScanCfg, TerrainScan))

register_sensor(SensorRegistration("RayLidar", RayLidarCfg, RayLidar))
register_sensor(SensorRegistration("RayDepthCamera", RayDepthCameraCfg, RayDepthCamera))
