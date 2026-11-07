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

"""Shared runtime contract for surface and volume deformable test fixtures."""

from __future__ import annotations

import pytest
import torch


class DeformableContract:
    """Use the subtype-owned ``sim`` and ``deformable`` fixtures."""

    def test_run_simulation(self):
        for _ in range(100):
            self.sim.update(step=1)
        self.deformable.reset()
        for _ in range(100):
            self.sim.update(step=1)

    def test_remove(self):
        with pytest.raises(NotImplementedError, match="pending removal"):
            self.sim.remove_asset(self.deformable.uid)
        assert self.sim.get_deformable_object(self.deformable.uid) is self.deformable

    def test_set_local_pose_updates_selected_particle_batch(self):
        """Setting one instance pose writes only its packed simulation nodes."""
        before = self.deformable.data.nodal_pos_w
        translation = torch.tensor([0.5, 0.0, 0.0], device=self.deformable.device)
        pose = torch.eye(
            4,
            dtype=torch.float32,
            device=self.deformable.device,
        ).unsqueeze(0)
        pose[:, :3, 3] = (
            torch.as_tensor(
                self.deformable.cfg.init_pos,
                dtype=torch.float32,
                device=self.deformable.device,
            )
            + translation
        )

        self.deformable.set_local_pose(pose, env_ids=[0])
        after = self.deformable.data.nodal_pos_w

        torch.testing.assert_close(after[0], before[0] + translation)
        torch.testing.assert_close(after[1:], before[1:])
