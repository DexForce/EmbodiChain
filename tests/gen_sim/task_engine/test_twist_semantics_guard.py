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
"""Actual source guard rejects role changes between qualification and assembly."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from embodichain.gen_sim.task_engine._task_program.twist_binding import discover_twist
from embodichain.gen_sim.task_engine._task_program.twist_geometry_guard import (
    E8GeometryGuard,
)
from embodichain.gen_sim.task_engine._task_program import twist_semantics
from .test_twist_generic_binding import _source


def _guard(tmp_path, config, route):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        '<robot name="fixture"><link name="right_pad"><collision>'
        '<geometry><box size="0.01 0.01 0.01"/></geometry>'
        "</collision></link></robot>"
    )
    art = NS(cfg=NS(fpath=config["fpath"], body_scale=config["body_scale"]))
    robot = NS(cfg=NS(fpath=str(urdf), body_scale=(1, 1, 1)), num_instances=1)
    simulation = NS(
        is_default_backend=True,
        get_articulation=lambda uid: art,
        get_rigid_object=lambda uid: None,
    )
    sensor = NS(art=art, route=route, geometry=NS())
    return E8GeometryGuard(route, simulation, robot, sensor)


def _request(route):
    binding = route.binding
    return NS(
        goal=NS(
            semantics=NS(
                entity_id=route.link_id,
                geometry={
                    "gen_sim_twist_articulation_id": binding.object_id,
                    "gen_sim_twist_joint": binding.joint,
                    "gen_sim_twist_arm": route.arm,
                    "gen_sim_twist_target_qpos": binding.target_qpos,
                    "gen_sim_twist_axis_sign": binding.axis_sign,
                },
            )
        )
    )


def test_guard_snapshot_is_not_replaced_by_a_post_qualification_sidecar_read(
    tmp_path, monkeypatch
):
    config, _, roles = _source(tmp_path, "race")
    route = discover_twist(config, 73)
    sidecar = Path(config["fpath"]).with_suffix(".twist.json")
    qualified_digest = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    original = twist_semantics.resolve_twist_semantics

    def changed_after_qualification(path):
        qualified = original(path)
        changed = dict(roles, settings={"0": roles["settings"]["0"]})
        sidecar.write_text(json.dumps(changed))
        return qualified

    monkeypatch.setattr(
        twist_semantics, "resolve_twist_semantics", changed_after_qualification
    )
    guard = _guard(tmp_path, config, route)
    assert guard._setup_error is None
    assert guard._semantics_sidecar_digest == qualified_digest
    assert hashlib.sha256(sidecar.read_bytes()).hexdigest() != qualified_digest
    with pytest.raises(ValueError, match="semantics changed"):
        guard._fresh(_request(route), NS(batch_size=1))


@pytest.mark.parametrize("change", ("edit", "remove"))
def test_guard_rejects_sidecar_change_after_assembly(tmp_path, change):
    config, _, roles = _source(tmp_path, "changed")
    route = discover_twist(config, 73)
    guard = _guard(tmp_path, config, route)
    guard._fresh(_request(route), NS(batch_size=1))
    sidecar = Path(config["fpath"]).with_suffix(".twist.json")
    if change == "remove":
        sidecar.unlink()
    else:
        sidecar.write_text(json.dumps(roles) + "\n")
    with pytest.raises(ValueError, match="semantics changed"):
        guard._fresh(_request(route), NS(batch_size=1))


def test_guard_rejects_new_sidecar_for_an_embedded_only_binding(tmp_path):
    config, stage, roles = _source(tmp_path, "embedded")
    sidecar = Path(config["fpath"]).with_suffix(".twist.json")
    sidecar.unlink()
    embedded = dict(roles)
    embedded.pop("source_sha256")
    stage.GetDefaultPrim().SetCustomDataByKey("gen_sim:twistControl", embedded)
    stage.GetRootLayer().Save()
    route = discover_twist(config, 73)
    guard = _guard(tmp_path, config, route)
    assert guard._semantics_sidecar_digest is None
    guard._fresh(_request(route), NS(batch_size=1))
    sidecar.write_text("{}")
    with pytest.raises(ValueError, match="semantics changed"):
        guard._fresh(_request(route), NS(batch_size=1))
