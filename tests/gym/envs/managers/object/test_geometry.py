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

from embodichain.lab.gym.envs.managers.object.geometry import (
    apply_svd_transfer_pcd,
    get_pcd_svd_frame,
)


def test_get_pcd_svd_frame_falls_back_to_cpu_on_svd_failure(monkeypatch):
    original_svd = torch.linalg.svd
    call_count = {"value": 0}

    def fake_svd(tensor):
        call_count["value"] += 1
        if call_count["value"] == 1:
            raise RuntimeError("cusolver error")
        return original_svd(tensor.to("cpu"))

    monkeypatch.setattr(torch.linalg, "svd", fake_svd)

    pc = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    pose = get_pcd_svd_frame(pc)

    assert pose.shape == (4, 4)
    assert call_count["value"] == 2


def test_apply_svd_transfer_pcd_does_not_call_linalg_inv(monkeypatch):
    def fail_inv(tensor):
        raise AssertionError("torch.linalg.inv should not be called")

    monkeypatch.setattr(torch.linalg, "inv", fail_inv)
    geometry = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    transformed = apply_svd_transfer_pcd(geometry, sample_points=4)

    assert transformed.shape == (1, 4, 3)
