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

"""Physically validated hand/object labels shared by native expansion tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import torch
import yaml

from embodichain.lab.sim.affordance.grasp.cache import ValidatedGraspCache

if TYPE_CHECKING:
    from embodichain.toolkits.graspkit.pg_grasp import AntipodalGraspPoseGeneratorCfg

__all__ = ["use_recorded_grasps"]


@contextmanager
def use_recorded_grasps(probe: str) -> Iterator[None]:
    """Replay physical labels for each mounted hand and contact frame."""
    path = Path(__file__).parent / "fixtures/validated_grasps.yaml"
    fixtures = yaml.safe_load(path.read_text(encoding="utf-8"))[probe]

    def lookup(
        cache: ValidatedGraspCache,
        vertices: torch.Tensor,
        triangles: torch.Tensor,
        attrs: dict | None,
        pose_adjustment: dict | None,
    ) -> tuple[dict[str, torch.Tensor], str, AntipodalGraspPoseGeneratorCfg]:
        root = cache.hand["root_link"]
        contact_frame = cache.hand["root_from_contact_frame"]
        matches = [
            entry
            for entry in fixtures
            if entry["root_link"] == root
            and entry["root_from_contact_frame"] == contact_frame
        ]
        assert len(matches) == 1, (root, contact_frame)
        entry = matches[0]
        torch.testing.assert_close(
            torch.stack((vertices.amin(0), vertices.amax(0))),
            vertices.new_tensor(entry["mesh_bounds"]),
            atol=1e-6,
            rtol=0,
        )
        key, _, _, algorithm = cache._label_job(
            vertices, triangles, attrs, pose_adjustment
        )
        labels = {
            field: torch.tensor(entry[field], dtype=torch.float32)
            for field in ("grasp_pose", "open_length")
        }
        return labels, key, algorithm

    with patch.object(ValidatedGraspCache, "_lookup", lookup):
        yield
