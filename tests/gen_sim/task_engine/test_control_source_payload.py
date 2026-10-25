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

"""Inherited supported payload AABBs are not new control-scale collisions."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from embodichain.gen_sim.task_engine._task_program.press_binding import (
    prepare_press_scene,
)
from embodichain.gen_sim.task_engine._task_program import twist_support
from embodichain.gen_sim.task_engine.orchestration.source_scene import PreparedScene
from embodichain.gen_sim.task_engine.scene import (
    articulation_geometry,
    final_inspection,
)


def _setup(monkeypatch, *, new_overlap=False):
    config = {
        "uid": "control",
        "fpath": "control.usdc",
        "body_scale": [1.0] * 3,
        "init_pos": [0.0, 0.0, 0.5],
        "init_rot": [0.0] * 3,
    }
    scene = PreparedScene(
        source_config_path=Path("scene_config.json"),
        scene_dir=Path("."),
        articulations=(config,),
        rigid_objects=({"uid": "payload"},),
        background=({"uid": "table"},),
        planner_objects=({"runtime_uid": "control"},),
        uid_map={"control": "control", "payload": "payload", "table": "table"},
        table_top_z=0.0,
        z_rotation_degrees=0.0,
        body_scale_policy="preserve",
        body_scale=(1.0, 1.0, 1.0),
        asset_hashes={},
    )
    vertices = np.array([[-0.5] * 3, [0.5] * 3])
    geometry = NS(
        world_vertices=lambda cfg: vertices * cfg["body_scale"] + cfg["init_pos"]
    )
    monkeypatch.setattr(
        articulation_geometry, "read_articulation_geometry", lambda _: geometry
    )
    payload_bounds = np.array(
        [
            [-0.1, -0.1, 1.2 if new_overlap else 0.3],
            [0.1, 0.1, 1.4 if new_overlap else 0.5],
        ]
    )
    monkeypatch.setattr(
        final_inspection,
        "_measure_geometry",
        lambda cfg, **kw: {
            "bounds": (
                np.array([[-2.0] * 3, [2.0] * 3])
                if cfg["uid"] == "table"
                else payload_bounds
            )
        },
    )

    def support(cfg, *args, **kwargs):
        if new_overlap:
            cfg["init_pos"][2] += 0.5
            cfg["init_local_pose"][2][3] += 0.5
        return cfg, {"scope": "test_support_stub"}

    monkeypatch.setattr(twist_support, "adjust_twist_support", support)
    route = NS(binding=NS(object_id="control", source_sha256="source", scale=1.0))
    return scene, (route,)


def test_inherited_source_payload_is_deferred_and_explicitly_audited(monkeypatch):
    scene, routes = _setup(monkeypatch)
    result, audit = prepare_press_scene(
        scene, routes, interaction="E8", source_payloads={("control", "payload")}
    )
    record = audit[0]["source_payload_overlap_deferred"][0]
    assert record["uid"] == "payload"
    assert "not_collision_certificate" in record["scope"]
    assert "accepted" not in record
    assert result.rigid_objects == scene.rigid_objects
    assert scene.articulations[0]["init_pos"] == [0.0, 0.0, 0.5]


@pytest.mark.parametrize(
    "case", ["undeclared", "new_scale", "new_overlap", "wrong_parent"]
)
def test_unknown_or_new_overlap_is_not_exempted(monkeypatch, case):
    scene, routes = _setup(monkeypatch, new_overlap=case == "new_overlap")
    pairs = (
        set()
        if case == "undeclared"
        else {(("other" if case == "wrong_parent" else "control"), "payload")}
    )
    kwargs = (
        {"interaction_scale": 1.1, "interaction_scale_qualified": True}
        if case == "new_scale"
        else {}
    )
    with pytest.raises(ValueError, match="scaled control overlaps"):
        prepare_press_scene(
            scene, routes, interaction="E8", source_payloads=pairs, **kwargs
        )


def test_e9_cannot_acquire_e8_payload_deferral(monkeypatch):
    scene, routes = _setup(monkeypatch)
    with pytest.raises(ValueError, match="source_payloads is only valid for E8"):
        prepare_press_scene(scene, routes, source_payloads={("control", "payload")})
