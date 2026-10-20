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

"""Asset selection and language binding across the production configuration path."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from embodichain.lab.gym.envs._asset_instructions import (
    _bind_asset_program_instructions,
    _render_asset_instruction,
)
from embodichain.lab.gym.envs.demo import resolve_demo_instruction
from embodichain.lab.gym.envs.managers import EventCfg, SceneEntityCfg
from embodichain.lab.gym.envs.managers._event_functors import asset_selection
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.sim.cfg import RigidObjectCfg
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.utils.utility import load_config

_ROOT = Path(__file__).resolve().parents[4]
_CONFIG = _ROOT / (
    "extensions/embodichain_tasks/configs/tasks/manipulation/"
    "repeated_pick_place/task.dual_ur5.assets.yaml"
)


def _term():
    original = RigidObjectCfg(uid="cube", shape=CubeCfg(size=[0.05] * 3))
    env = SimpleNamespace(
        num_envs=2,
        sim=Mock(),
        _task_program_adapter=None,
    )
    env.sim.get_rigid_object.return_value = SimpleNamespace(cfg=original)
    entity = SceneEntityCfg(uid="cube")
    cfg = EventCfg(mode="startup", params={"entity_cfg": entity})
    return asset_selection.select_rigid_object_asset(cfg, env), env, entity


def test_selection_switches_shape_and_label_without_mutating_template() -> None:
    term, env, entity = _term()
    choices = {
        "large": {
            "label": "large cube",
            "shape": {"shape_type": "Cube", "size": [0.1] * 3},
        },
        "small": {
            "label": "small cube",
            "shape": {"shape_type": "Cube", "size": [0.02] * 3},
        },
    }
    snapshot = deepcopy(choices)
    term(env, [1, 0], entity, "large", choices)
    large = env.sim.replace_rigid_object.call_args.args[0]
    assert large.uid == "cube" and large.shape.size == [0.1] * 3
    assert env._asset_selections["cube"]["label"] == "large cube"
    term(env, None, entity, "small", choices)
    assert env.sim.replace_rigid_object.call_args.args[0].shape.size == [0.02] * 3
    assert env._asset_selections["cube"]["asset"] == "small"
    assert choices == snapshot
    assert term._base_cfg["shape"]["size"] == [0.05] * 3


@pytest.mark.parametrize("rows", [[0], [0, 0], [0, 2], [True, False], [0.0, 1.0]])
def test_selection_rejects_partial_or_invalid_batches_before_mutation(rows) -> None:
    term, env, entity = _term()
    with pytest.raises(ValueError):
        term(env, rows, entity, "cube", {"cube": {"label": "cube"}})
    env.sim.replace_rigid_object.assert_not_called()


@pytest.mark.parametrize(
    "declaration",
    [
        {"label": ""},
        {"label": "cube", "uid": "other"},
        {"label": "cube", "typo": 1},
        {"label": "cube", "rigidize_usd": True},
    ],
)
def test_invalid_selection_does_not_change_asset_or_description(declaration) -> None:
    term, env, entity = _term()
    with pytest.raises(ValueError):
        term(env, None, entity, "cube", {"cube": declaration})
    env.sim.replace_rigid_object.assert_not_called()
    assert not hasattr(env, "_asset_selections")


def test_failed_replacement_preserves_previous_identity() -> None:
    term, env, entity = _term()
    env._asset_selections = {"cube": {"asset": "old", "label": "old cube"}}
    env.sim.replace_rigid_object.side_effect = ValueError("invalid shape")
    with pytest.raises(ValueError, match="invalid shape"):
        term(env, None, entity, "new", {"new": {"label": "new cube"}})
    assert env._asset_selections["cube"]["label"] == "old cube"


def test_selection_rejects_changes_after_adapter_binding() -> None:
    term, env, entity = _term()
    env._task_program_adapter = object()
    with pytest.raises(RuntimeError, match="before constructing"):
        term(env, None, entity, "cube", {"cube": {"label": "cube"}})
    env.sim.replace_rigid_object.assert_not_called()


def test_selection_requires_startup_phase() -> None:
    _, env, entity = _term()
    with pytest.raises(ValueError, match="startup"):
        asset_selection.select_rigid_object_asset(
            EventCfg(mode="reset", params={"entity_cfg": entity}), env
        )


@pytest.mark.parametrize(
    "asset,label", [("cube", "cube"), ("rubiks_cube", "Rubik's cube")]
)
def test_production_asset_program_binds_episode_and_segment_language(
    asset, label
) -> None:
    payload = load_config(_CONFIG)
    payload["id"] = f"AssetLanguage-{asset.replace('_', '-')}-v1"
    cfg = config_to_cfg(payload, source_path=_CONFIG)
    env = SimpleNamespace(
        cfg=cfg, _asset_selections={"cube": {"asset": asset, "label": label}}
    )
    original = cfg.task_program
    bound = _bind_asset_program_instructions(env, original)
    from embodichain.lab.gym.utils.registration import REGISTERED_ENVS
    from embodichain.lab.gym.envs import EmbodiedEnv

    catalog = REGISTERED_ENVS[payload["id"]].task_program_registration.catalog
    compiled = catalog.preflight(bound)
    assert resolve_demo_instruction(env)[0] == bound.instruction
    assert label in bound.instruction and "{cube}" not in bound.instruction
    segments = list(compiled.iter_segments())
    assert len(segments) == 6
    assert all(
        label in segment.instruction and "{cube}" not in segment.instruction
        for segment in segments
    )
    assert "{cube}" in original.instruction
    assert bound.program_id == original.program_id
    assert bound.targets == original.targets
    env._recording_run_uuid = "run"
    env._recording_provenance = {}
    metadata = EmbodiedEnv._new_demo_episode_metadata(env, 0)
    assert metadata["instruction"] == bound.instruction
    assert metadata["assets"]["cube"]["asset"] == asset
    env._asset_selections["cube"]["label"] = "later"
    assert metadata["assets"]["cube"]["label"] == label


def test_instruction_binding_is_literal_and_does_not_expand_labels_recursively() -> (
    None
):
    env = SimpleNamespace(
        _asset_selections={"cube": {"label": "{other}"}, "other": {"label": "changed"}}
    )
    assert (
        _render_asset_instruction(env, "Move {cube}; {cube.__class__}")
        == "Move {other}; {cube.__class__}"
    )


def test_rigidized_usd_keeps_source_and_centers_the_single_body(
    tmp_path, monkeypatch
) -> None:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from embodichain.data import constants

    monkeypatch.setattr(constants, "EMBODICHAIN_DEFAULT_DATA_ROOT", str(tmp_path))
    source = tmp_path / "source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.Xform.Define(stage, "/World")
    root = UsdGeom.Xform.Define(stage, "/World/Asset").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    body = UsdGeom.Cube.Define(stage, "/World/Asset/Body")
    body.CreateSizeAttr(0.05)
    UsdGeom.Xformable(body).AddTranslateOp().Set(Gf.Vec3d(0, 0.025, 0))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.RevoluteJoint.Define(stage, "/World/Asset/Joint")
    stage.GetRootLayer().Save()
    before = source.read_bytes()
    derived = asset_selection._rigidize_usd(source)
    assert derived == asset_selection._rigidize_usd(source)
    assert source.read_bytes() == before
    result = Usd.Stage.Open(str(derived))
    assert sum(prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in result.Traverse()) == 1
    assert not any(prim.IsA(UsdPhysics.Joint) for prim in result.Traverse())
    bounds = (
        UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
        .ComputeWorldBound(result.GetDefaultPrim())
        .ComputeAlignedRange()
    )
    assert tuple(bounds.GetMidpoint()) == pytest.approx((0, 0, 0))
    assert tuple(bounds.GetSize()) == pytest.approx((0.05, 0.05, 0.05))
