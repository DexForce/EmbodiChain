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

import torch

from embodichain.lab.sim.agent.task_state import summarize_pour_water_state


class _FakeObject:
    def __init__(self, pose: torch.Tensor) -> None:
        self.pose = pose

    def get_local_pose(self, to_matrix: bool = True):
        return self.pose.unsqueeze(0)


class _FakeSim:
    def __init__(self, objects: dict[str, _FakeObject]) -> None:
        self.objects = objects

    def get_rigid_object(self, obj_name: str):
        return self.objects[obj_name]

    def update(self, step: int = 1) -> None:
        return None


class _FakeEnv:
    def __init__(self, poses: dict[str, torch.Tensor]) -> None:
        self.sim = _FakeSim({name: _FakeObject(pose) for name, pose in poses.items()})
        self.obj_info = {
            name: {
                "initial_pose": pose.clone(),
                "pose": pose.clone(),
                "height": pose[2, 3].clone(),
            }
            for name, pose in poses.items()
        }

    def update_obj_info(self) -> None:
        for name, obj in self.sim.objects.items():
            self.obj_info[name]["pose"] = obj.pose.clone()


def _pose(x: float, y: float, z: float, rotation: torch.Tensor | None = None):
    pose = torch.eye(4)
    pose[:3, :3] = torch.eye(3) if rotation is None else rotation
    pose[:3, 3] = torch.tensor([x, y, z])
    return pose


def test_summarize_pour_water_state_accepts_upright_objects() -> None:
    env = _FakeEnv(
        {
            "bottle": _pose(0.5, 0.0, 0.8),
            "cup": _pose(0.6, 0.1, 0.7),
        }
    )

    summary = summarize_pour_water_state(env)

    assert summary.semantic_success is True
    assert summary.failure_reasons == []


def test_summarize_pour_water_state_rejects_toppled_object() -> None:
    rotation = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    env = _FakeEnv(
        {
            "bottle": _pose(0.5, 0.0, 0.8, rotation=rotation),
            "cup": _pose(0.6, 0.1, 0.7),
        }
    )

    summary = summarize_pour_water_state(env)

    assert summary.semantic_success is False
    assert any(
        "bottle: toppled_or_tilted" in reason
        for reason in summary.failure_reasons
    )
