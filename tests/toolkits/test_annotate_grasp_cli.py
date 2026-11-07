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

"""Behavioral coverage for annotation's explicit physical-hardware contract."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import trimesh

from embodichain.toolkits.graspkit.scripts import annotate_grasp as module
from embodichain.toolkits.graspkit import pg_grasp
from embodichain.toolkits.graspkit.pg_grasp import model_geometry


def test_cli_requires_explicit_model() -> None:
    with pytest.raises(SystemExit) as exc:
        module.cli(["--mesh_path", "object.stl"])
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "contact_options, expected_bounds",
    [
        ([], (0.001, 0.08)),
        (["--min_length", "0.004", "--max_length", "0.06"], (0.004, 0.06)),
    ],
)
def test_builtin_cli_keeps_physical_apertures_and_selected_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contact_options: list[str],
    expected_bounds: tuple[float, float],
) -> None:
    path = tmp_path / "object.stl"
    trimesh.creation.icosphere(subdivisions=0, radius=0.01).export(path)
    selected = object()
    seen = {}

    def geometry(model_id, **kwargs):
        seen["identity"] = (model_id, kwargs)
        return selected

    class Generator:
        def __init__(self, model, **kwargs):
            seen["model"] = model
            seen["geometry"] = kwargs["collision_geometry"]

        def prepare_mesh(self, **kwargs):
            assert kwargs["mesh_vertices"].shape[1] == 3
            return torch.zeros(2, 2, 3)

    monkeypatch.setattr(model_geometry, "get_parallel_jaw_collision_geometry", geometry)
    monkeypatch.setattr(pg_grasp, "AntipodalGraspPoseGenerator", Generator)
    module.cli(
        [
            "--mesh_path",
            str(path),
            "--gripper-model",
            "franka_panda_hand",
            *contact_options,
        ]
    )
    assert seen["identity"] == ("franka_panda_hand", {"side": None})
    assert seen["geometry"] is selected
    assert (
        seen["model"].min_opening_width,
        seen["model"].max_opening_width,
    ) == expected_bounds
