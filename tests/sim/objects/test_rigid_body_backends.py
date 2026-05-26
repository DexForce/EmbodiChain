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
import warp as wp

from embodichain.lab.sim.objects.backends.default import DefaultRigidBodyView
from embodichain.lab.sim.objects.backends.newton import NewtonRigidBodyView


class _Entity:
    def __init__(self, index: int) -> None:
        self.index = index

    def get_location(self) -> list[float]:
        return [float(self.index), float(self.index + 1), float(self.index + 2)]

    def get_rotation_quat(self) -> list[float]:
        return [0.0, 0.0, 0.0, 1.0]

    def get_linear_velocity(self) -> list[float]:
        return [float(self.index + 3), float(self.index + 4), float(self.index + 5)]

    def get_angular_velocity(self) -> list[float]:
        return [float(self.index + 6), float(self.index + 7), float(self.index + 8)]

    def get_linear_acceleration(self) -> list[float]:
        return [float(self.index + 9), float(self.index + 10), float(self.index + 11)]

    def get_angular_acceleration(self) -> list[float]:
        return [float(self.index + 12), float(self.index + 13), float(self.index + 14)]

    def get_native_handle(self) -> int:
        return self.index

    def get_gpu_index(self) -> int:
        return self.index


class _NewtonDataType:
    POSE = "pose"
    LINEAR_VELOCITY = "linear_velocity"
    ANGULAR_VELOCITY = "angular_velocity"
    LINEAR_ACCELERATION = "linear_acceleration"
    ANGULAR_ACCELERATION = "angular_acceleration"


class _NewtonScene:
    def __init__(self) -> None:
        self.manager = SimpleNamespace(
            lifecycle_state=SimpleNamespace(name="READY"),
            dexsim2newton_body={10: 100, 11: 101},
        )

    def gpu_fetch_rigid_body_data(self, body_ids, data_type, out) -> None:
        data = wp.to_torch(out)
        if data_type == _NewtonDataType.POSE:
            width = 7
        else:
            width = 3
        values = torch.arange(
            len(body_ids) * width, dtype=torch.float32, device=data.device
        ).reshape(len(body_ids), width)
        data.copy_(values)

    def gpu_apply_rigid_body_data(self, body_ids, data_type, payload) -> None:
        pass


def test_default_fetch_methods_fill_caller_buffer() -> None:
    view = DefaultRigidBodyView(
        entities=[_Entity(0), _Entity(10)],
        ps=object(),
        device=torch.device("cpu"),
    )

    pose = torch.empty((2, 7), dtype=torch.float32)
    lin_vel = torch.empty((2, 3), dtype=torch.float32)
    ang_vel = torch.empty((2, 3), dtype=torch.float32)
    lin_acc = torch.empty((2, 3), dtype=torch.float32)
    ang_acc = torch.empty((2, 3), dtype=torch.float32)
    ptrs = [tensor.data_ptr() for tensor in (pose, lin_vel, ang_vel, lin_acc, ang_acc)]

    assert view.fetch_pose(pose) is None
    assert view.fetch_linear_velocity(lin_vel) is None
    assert view.fetch_angular_velocity(ang_vel) is None
    assert view.fetch_linear_acceleration(lin_acc) is None
    assert view.fetch_angular_acceleration(ang_acc) is None

    assert ptrs == [
        tensor.data_ptr() for tensor in (pose, lin_vel, ang_vel, lin_acc, ang_acc)
    ]
    assert torch.allclose(
        pose,
        torch.tensor(
            [
                [0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 1.0],
                [10.0, 11.0, 12.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )
    assert torch.allclose(lin_vel, torch.tensor([[3.0, 4.0, 5.0], [13.0, 14.0, 15.0]]))
    assert torch.allclose(ang_vel, torch.tensor([[6.0, 7.0, 8.0], [16.0, 17.0, 18.0]]))
    assert torch.allclose(
        lin_acc, torch.tensor([[9.0, 10.0, 11.0], [19.0, 20.0, 21.0]])
    )
    assert torch.allclose(
        ang_acc, torch.tensor([[12.0, 13.0, 14.0], [22.0, 23.0, 24.0]])
    )


def test_newton_fetch_methods_fill_caller_buffer(monkeypatch) -> None:
    wp.init()
    monkeypatch.setattr(NewtonRigidBodyView, "_DATA_TYPE", _NewtonDataType)
    view = NewtonRigidBodyView(
        entities=[_Entity(10), _Entity(11)],
        scene=_NewtonScene(),
        device=torch.device("cpu"),
    )

    pose = torch.empty((2, 7), dtype=torch.float32)
    lin_vel = torch.empty((2, 3), dtype=torch.float32)
    ang_vel = torch.empty((2, 3), dtype=torch.float32)
    lin_acc = torch.empty((2, 3), dtype=torch.float32)
    ang_acc = torch.empty((2, 3), dtype=torch.float32)
    pose_ptr = pose.data_ptr()
    lin_vel_ptr = lin_vel.data_ptr()
    ang_vel_ptr = ang_vel.data_ptr()
    lin_acc_ptr = lin_acc.data_ptr()
    ang_acc_ptr = ang_acc.data_ptr()

    assert view.fetch_pose(pose) is None
    assert view.fetch_linear_velocity(lin_vel) is None
    assert view.fetch_angular_velocity(ang_vel) is None
    assert view.fetch_linear_acceleration(lin_acc) is None
    assert view.fetch_angular_acceleration(ang_acc) is None

    assert pose.data_ptr() == pose_ptr
    assert lin_vel.data_ptr() == lin_vel_ptr
    assert ang_vel.data_ptr() == ang_vel_ptr
    assert lin_acc.data_ptr() == lin_acc_ptr
    assert ang_acc.data_ptr() == ang_acc_ptr
    assert torch.allclose(pose, torch.arange(14, dtype=torch.float32).reshape(2, 7))
    assert torch.allclose(lin_vel, torch.arange(6, dtype=torch.float32).reshape(2, 3))
    assert torch.allclose(ang_vel, torch.arange(6, dtype=torch.float32).reshape(2, 3))
    assert torch.allclose(lin_acc, torch.arange(6, dtype=torch.float32).reshape(2, 3))
    assert torch.allclose(ang_acc, torch.arange(6, dtype=torch.float32).reshape(2, 3))
