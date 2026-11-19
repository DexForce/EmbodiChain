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

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import pytest
import torch
from embodichain.utils.nms import pose_nms, pose_nms_scope


def test_nms_memory_scope_is_owned_and_preserves_selection():
    def current():
        with pose_nms_scope() as size:
            return size

    poses = torch.eye(4).repeat(8, 1, 1)
    poses[:, 0, 3] = torch.tensor([0.0, 0.001, 0.002, 0.01, 0.012, 0.02, 0.021, 0.03])
    expected = pose_nms(poses, chunk_size=2)
    with pose_nms_scope(chunk_size=2):
        inherited = copy_context()
        with pytest.raises(RuntimeError):
            with pose_nms_scope(chunk_size=3):
                assert current() == 3
                raise RuntimeError("scope interrupted")
        assert current() == 2
        actual = pose_nms(poses)
        assert torch.equal(actual[0], expected[0]) and torch.equal(
            actual[1], expected[1]
        )
        explicit = pose_nms(poses, chunk_size=1)
        assert torch.equal(explicit[1], expected[1])
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(current).result() == 2048
    assert current() == 2048 and inherited.run(current) == 2
    with pytest.raises(ValueError):
        with pose_nms_scope(chunk_size=0):
            pass
