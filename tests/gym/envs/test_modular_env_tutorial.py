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

import runpy
from pathlib import Path

import pytest

from embodichain.lab.sim.spawn.descriptors import rigid_desc_from_cfg

pytestmark = pytest.mark.no_sim


def test_tutorial_rigid_objects_compile_to_spawn_descriptors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Resolve paths without downloading assets; descriptor construction is pure.
    monkeypatch.setattr("embodichain.data.get_data_path", lambda path: path)
    namespace = runpy.run_path(
        str(
            Path(__file__).resolve().parents[3] / "scripts/tutorials/gym/modular_env.py"
        )
    )
    cfg = namespace["ExampleCfg"]()

    for object_cfg in [*cfg.background, *cfg.rigid_object]:
        descriptor, _ = rigid_desc_from_cfg(object_cfg)
        assert descriptor.physics is not None
        assert descriptor.renders
        assert descriptor.collisions
