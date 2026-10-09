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
"""GenSim mass migration retains binding, provenance and native readback gates."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pxr import Gf, Usd, UsdPhysics

from embodichain.gen_sim.task_engine._task_program.twist_binding import discover_twist
from embodichain.gen_sim.task_engine._task_program.twist_mass import verify_scaled_mass
from embodichain.gen_sim.task_engine._task_program.twist_mass_source import (
    create_mass_source_copy,
    qualify_mass_source_copy,
)
from embodichain.gen_sim.task_engine.task_program_bundle import (
    generate_task_program_bundle,
)
from embodichain.utils.utility import load_config
from .test_twist import graph, knob_scene
from .test_twist_mass import _ReadOnlyArticulation
from .test_twist_semantics_guard import _guard, _request


def _authored_scene(scene):
    path = Path(scene.articulations[0]["fpath"])
    stage = Usd.Stage.Open(str(path))
    for name in ("base", "knob"):
        mass = UsdPhysics.MassAPI(stage.GetPrimAtPath("/Control/" + name))
        mass.CreateCenterOfMassAttr(Gf.Vec3f(0.001, 0.002, 0.003))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.01, 0.02, 0.03))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0)))
    stage.GetRootLayer().Save()
    roles = json.loads(path.with_suffix(".twist.json").read_text())
    roles["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".twist.json").write_text(json.dumps(roles))
    return scene


def _derived(scene, tmp_path):
    from dexsim.kit.usd import parse_usd

    scene = _authored_scene(scene)
    original = scene.articulations[0]
    scale = 0.5
    copy = create_mass_source_copy(
        Path(original["fpath"]), tmp_path / "mass", (scale,) * 3
    )
    config = {**original, "fpath": str(copy.runtime), "body_scale": [scale] * 3}
    route = discover_twist(config, 90)
    descriptor = parse_usd(str(copy.source)).articulations[0]
    return copy, config, route, descriptor


def test_deployment_mass_bundle_preflights_without_lab_mass_policy(
    knob_scene, tmp_path
):
    scene = _authored_scene(knob_scene)
    _, a = generate_task_program_bundle(
        graph(scene), scene, tmp_path / "a", robot_profile="dual_franka"
    )
    _, b = generate_task_program_bundle(
        graph(scene),
        scene,
        tmp_path / "b",
        robot_profile="dual_franka",
        twist_mass_source=True,
    )
    original = load_config(a.scene)["simulation"]["articulation"][0]
    derived = load_config(b.scene)["simulation"]["articulation"][0]
    from embodichain.lab.sim.cfg import ArticulationCfg

    if "body_scale_mass_policy" in ArticulationCfg.__dataclass_fields__:
        assert original["body_scale_mass_policy"] == "fixed_mass"
    else:
        assert "body_scale_mass_policy" not in original
    assert "body_scale_mass_policy" not in derived
    old = discover_twist(original, 90)
    new = discover_twist(derived, 90)
    assert new.binding.mass_lineage_sha256
    assert (
        replace(
            new.binding,
            source_sha256=old.binding.source_sha256,
            mass_lineage_sha256=old.binding.mass_lineage_sha256,
        )
        == old.binding
    )
    qualified = qualify_mass_source_copy(
        Path(derived["fpath"]), manifest_sha256=new.binding.mass_lineage_sha256
    )
    assert (
        qualified.source.read_bytes()
        == Path(scene.articulations[0]["fpath"]).read_bytes()
    )


def test_native_mass_readback_uses_original_source_once(knob_scene, tmp_path):
    copy, _, route, descriptor = _derived(knob_scene, tmp_path)
    art = _ReadOnlyArticulation(
        copy.runtime, descriptor, scale=copy.scale, policy="native"
    )
    result = verify_scaled_mass(art, route.binding)
    assert (
        result["verified"] and result["conversion_owner"] == "gen_sim_deployment_source"
    )
    assert result["origin_source_sha256"] == copy.source_sha256
    assert art.setter_calls == 0


def test_unmodified_lab_loader_selects_deployment_mass_source_by_capability(
    knob_scene, tmp_path, monkeypatch
):
    from embodichain.lab.sim.cfg import ArticulationCfg

    scene = _authored_scene(knob_scene)
    monkeypatch.setattr(
        ArticulationCfg,
        "__dataclass_fields__",
        {
            key: value
            for key, value in ArticulationCfg.__dataclass_fields__.items()
            if key != "body_scale_mass_policy"
        },
    )
    _, paths = generate_task_program_bundle(
        graph(scene), scene, tmp_path / "baseline_loader", robot_profile="dual_franka"
    )
    art = load_config(paths.scene)["simulation"]["articulation"][0]
    assert "body_scale_mass_policy" not in art
    assert discover_twist(art, 90).binding.mass_lineage_sha256


def test_derived_mass_source_rejects_lab_conversion(knob_scene, tmp_path):
    copy, _, route, descriptor = _derived(knob_scene, tmp_path)
    art = _ReadOnlyArticulation(
        copy.runtime, descriptor, scale=copy.scale, policy="fixed_mass"
    )
    with pytest.raises(ValueError, match="conflicts"):
        verify_scaled_mass(art, route.binding)


def test_derived_mass_source_requires_bound_manifest(knob_scene, tmp_path):
    copy, _, route, descriptor = _derived(knob_scene, tmp_path)
    art = _ReadOnlyArticulation(
        copy.runtime, descriptor, scale=copy.scale, policy="native"
    )
    with pytest.raises(ValueError, match="bound lineage"):
        verify_scaled_mass(art, replace(route.binding, mass_lineage_sha256=""))


@pytest.mark.parametrize("target", ("original_usd", "original_roles", "manifest"))
def test_actual_geometry_guard_checks_mass_lineage_freshness(
    knob_scene, tmp_path, target
):
    copy, config, route, _ = _derived(knob_scene, tmp_path)
    guard = _guard(tmp_path, config, route)
    context = SimpleNamespace(batch_size=1)
    guard._fresh(_request(route), context)
    path = {
        "original_usd": copy.source,
        "original_roles": copy.source.with_suffix(".twist.json"),
        "manifest": copy.manifest,
    }[target]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed"):
        guard._fresh(_request(route), context)
