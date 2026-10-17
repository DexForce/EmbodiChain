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
"""USD source policies are exercised with real stages, without a native World."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    MassPropertiesCfg,
    RigidBodyPhysicsCfg,
    RigidObjectCfg,
    UsdSourceCfg,
)
from embodichain.lab.sim.shapes import CubeCfg, MeshCfg
from embodichain.lab.sim.spawn.usd import (
    articulation_desc_from_usd,
    rigid_desc_from_usd,
)

pytestmark = pytest.mark.no_sim


def _write_stage(tmp_path: Path, *, articulation: bool = False) -> str:
    path = tmp_path / "source.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    if articulation:
        UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    for name, pos in (("base", (2, 3, 4)), ("child", (2, 3, 5))):
        body = UsdGeom.Xform.Define(stage, f"/World/{name}")
        body.AddTranslateOp().Set(Gf.Vec3d(*pos))
        rigid = UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        if not articulation:
            rigid.CreateKinematicEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(2)
        mesh = UsdGeom.Cube.Define(stage, f"/World/{name}/mesh")
        mesh.CreateSizeAttr(0.2)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    if articulation:
        joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/hinge")
        joint.CreateBody0Rel().SetTargets(["/World/base"])
        joint.CreateBody1Rel().SetTargets(["/World/child"])
        joint.CreateLowerLimitAttr(-90)
        joint.CreateUpperLimitAttr(90)
    stage.GetRootLayer().Save()
    return str(path)


def test_usd_source_config_roundtrips_through_the_rigid_loader():
    cfg = RigidObjectCfg.from_dict(
        {
            "uid": "selected",
            "shape": {"shape_type": "Mesh", "fpath": "source.usda"},
            "usd": {
                "prim_path": "/World/Box",
                "pose_mode": "preserve",
                "variants": {"/World": {"model": "large"}},
            },
        }
    )

    assert isinstance(cfg.usd, UsdSourceCfg)
    assert cfg.usd.prim_path == "/World/Box"
    assert cfg.usd.pose_mode == "preserve"
    assert cfg.usd.variants == {"/World": {"model": "large"}}
    assert cfg.body_type is None
    assert cfg.body_scale is None
    assert cfg.init_local_pose is None


def test_selecting_one_rigid_keeps_source_pose_and_actor_type(tmp_path):
    cfg = RigidObjectCfg(
        uid="selected",
        shape=MeshCfg(fpath=_write_stage(tmp_path)),
        usd=UsdSourceCfg(prim_path="/World/base"),
        asset_physics_mode="overlay",
        attrs=RigidBodyPhysicsCfg(mass_props=MassPropertiesCfg(mass=5)),
    )

    desc, _ = rigid_desc_from_usd(cfg)

    np.testing.assert_allclose(desc.pose[:3, 3], [2, 3, 4])
    assert desc.physics.actor_type.name == "KINEMATIC"
    assert desc.physics.mass == 5
    assert cfg.body_type == "kinematic"


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("config", [1, 0, 0]),
        ("preserve", [2, 3, 4]),
        ("offset", [3, 3, 4]),
    ],
)
def test_rigid_pose_policy_is_explicit(tmp_path, mode, expected):
    cfg = RigidObjectCfg(
        uid="selected",
        shape=MeshCfg(fpath=_write_stage(tmp_path)),
        init_pos=(1, 0, 0),
        usd=UsdSourceCfg(prim_path="/World/base", pose_mode=mode),
    )
    desc, _ = rigid_desc_from_usd(cfg)
    np.testing.assert_allclose(desc.pose[:3, 3], expected)


def test_articulation_defaults_preserve_a_floating_usd_root(tmp_path):
    cfg = ArticulationCfg(
        uid="assembly", fpath=_write_stage(tmp_path, articulation=True)
    )

    desc, _ = articulation_desc_from_usd(cfg)

    assert desc.fixed_base is False
    np.testing.assert_allclose(desc.pose[:3, 3], [2, 3, 4])


def test_explicit_initial_pose_still_places_a_usd_asset(tmp_path):
    cfg = ArticulationCfg(
        uid="assembly",
        fpath=_write_stage(tmp_path, articulation=True),
        init_pos=(4, 5, 6),
    )
    desc, _ = articulation_desc_from_usd(cfg)
    np.testing.assert_allclose(desc.pose[:3, 3], [4, 5, 6])


def test_procedural_defaults_remain_dynamic_and_unit_scale():
    cfg = RigidObjectCfg(uid="cube", shape=CubeCfg())
    assert cfg.body_type == "dynamic"
    assert cfg.body_scale == (1, 1, 1)


def test_urdf_import_defaults_remain_fixed_without_self_collision():
    from embodichain.lab.sim.spawn.descriptors import articulation_desc_from_cfg

    desc = articulation_desc_from_cfg(ArticulationCfg(uid="robot", fpath="robot.urdf"))

    assert desc.fixed_base is True
    assert desc.enable_self_collision is False


def test_usd_source_config_rejects_unknown_or_invalid_options():
    with pytest.raises(ValueError, match="unknown"):
        UsdSourceCfg.from_dict({"unknown": True})
    with pytest.raises(ValueError, match="pose_mode"):
        UsdSourceCfg(pose_mode="unsupported")


def _declaration_manager():
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    import torch
    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain.lab.sim.spawn.scene import SpawnScene

    sim = object.__new__(SimulationManager)
    sim.device = torch.device("cpu")
    sim.physics = SimpleNamespace(name="default")
    sim._rigid_objects = {}
    sim._articulations = {}
    sim._robots = {}
    sim._sensors = {}
    sim.add_sensor = MagicMock()
    scene = SpawnScene.__new__(SpawnScene)
    scene._assets = {}
    scene._num_envs = 2
    scene.builder = MagicMock()
    scene.builder.is_finalized = False
    scene.builder.result = None
    scene.builder.add_scene.side_effect = lambda desc, **kw: list(desc.iter_entities())
    sim._spawn_scene = scene
    sim.notify_visualization_topology_changed = MagicMock()
    return sim


def test_whole_scene_applies_sparse_override_before_materialization(tmp_path):
    sim = _declaration_manager()
    path = _write_stage(tmp_path)

    assets = sim.add_usd(
        "room",
        path,
        usd=UsdSourceCfg(namespace="room"),
        asset_cfgs={
            "/World/base": {
                "asset_physics_mode": "overlay",
                "attrs": {"mass_props": {"mass": 5}},
            }
        },
    )

    assert set(assets) == {"/World/base", "/World/child"}
    assert assets["/World/base"].uid == "room::base"
    assert assets["/World/base"].cfg.body_type == "kinematic"
    desc = sim._spawn_scene.builder.add_scene.call_args.args[0]
    assert desc.mesh_objects[0].physics.mass == 5
    np.testing.assert_allclose(desc.mesh_objects[0].pose[:3, 3], [2, 3, 4])
    assert assets["/World/base"].num_instances == 2


def test_whole_scene_rejects_unknown_override_before_native_declaration(tmp_path):
    sim = _declaration_manager()
    with pytest.raises(ValueError, match="missing"):
        sim.add_usd("room", _write_stage(tmp_path), asset_cfgs={"/World/missing": {}})
    sim._spawn_scene.builder.add_scene.assert_not_called()
    assert sim._rigid_objects == {}


def test_whole_scene_honors_articulation_configuration_and_fk_choice(tmp_path):
    sim = _declaration_manager()
    assets = sim.add_usd(
        "room",
        _write_stage(tmp_path, articulation=True),
        asset_cfgs={
            "/World": {"build_pk_chain": True, "root_props": {"fixed_base": True}}
        },
    )

    assert assets["/World"].cfg.build_pk_chain is True
    assert assets["/World"].cfg.root_props.fixed_base is True
    desc = sim._spawn_scene.builder.add_scene.call_args.args[0]
    assert desc.articulations[0].fixed_base is True


def test_legacy_robot_mapping_preserves_physics_pose_and_mounted_sensors(tmp_path):
    from embodichain.lab.sim.cfg import RobotCfg, ArticulationRootPropertiesCfg
    from embodichain.lab.sim.sensors import CameraCfg

    sim = _declaration_manager()
    camera = CameraCfg(uid="wrist", extrinsics=CameraCfg.ExtrinsicsCfg(parent="child"))
    metadata = RobotCfg(
        uid="metadata_uid",
        init_pos=(9, 8, 7),
        build_pk_chain=False,
        root_props=ArticulationRootPropertiesCfg(fixed_base=True),
        enable_sensor=True,
        sensors=[camera],
        joint_drive_props=None,
    )

    assets = sim.add_usd(
        "room",
        _write_stage(tmp_path, articulation=True),
        robot_cfgs={"/World": metadata},
    )

    obj = assets["/World"]
    assert obj.uid == "World"
    assert obj.cfg.root_props.fixed_base is False
    np.testing.assert_allclose(obj.cfg.init_local_pose[:3, 3], [2, 3, 4])
    sim.add_sensor.assert_called_once()
    assert sim.add_sensor.call_args.args[0].extrinsics.parent == "World/child"
    assert camera.extrinsics.parent == "child"


def test_whole_scene_preflights_robot_sensor_names_before_materialization(tmp_path):
    from embodichain.lab.sim.cfg import RobotCfg
    from embodichain.lab.sim.sensors import CameraCfg

    sim = _declaration_manager()
    sim._sensors["wrist"] = object()
    metadata = RobotCfg(
        enable_sensor=True, sensors=[CameraCfg(uid="wrist")], joint_drive_props=None
    )
    with pytest.raises(ValueError, match="already exists"):
        sim.add_usd(
            "room",
            _write_stage(tmp_path, articulation=True),
            robot_cfgs={"/World": metadata},
        )
    sim._spawn_scene.builder.add_scene.assert_not_called()


def test_whole_scene_rejects_misspelled_override_fields_before_creation(tmp_path):
    sim = _declaration_manager()
    with pytest.raises(ValueError, match="misspelled"):
        sim.add_usd(
            "room",
            _write_stage(tmp_path),
            asset_cfgs={"/World/base": {"misspelled": 1}},
        )
    sim._spawn_scene.builder.add_scene.assert_not_called()


@pytest.mark.parametrize(
    "role,body_type,valid",
    [
        ("interactive", "dynamic", True),
        ("interactive", "kinematic", False),
        ("background", "kinematic", True),
        ("background", "dynamic", False),
    ],
)
def test_gym_validates_usd_actor_role_after_source_resolution(
    monkeypatch, role, body_type, valid
):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv

    cfg = RigidObjectCfg(uid="usd", shape=MeshCfg(fpath="source.usda"))
    assert cfg.body_type is None
    errors = Mock()
    monkeypatch.setattr(
        "embodichain.lab.gym.envs.embodied_env.logger.log_error", errors
    )

    def declare(*, cfg):
        cfg.body_type = body_type
        return SimpleNamespace(cfg=cfg)

    env = SimpleNamespace(
        cfg=SimpleNamespace(
            background=[cfg], rigid_object=[cfg], articulation=[], rigid_object_group=[]
        ),
        sim=SimpleNamespace(add_rigid_object=declare),
    )
    if role == "interactive":
        EmbodiedEnv._setup_interactive_objects(env)
    else:
        EmbodiedEnv._setup_background(env)

    assert errors.called is not valid


def test_source_path_override_infers_articulation_uid(tmp_path):
    path = _write_stage(tmp_path, articulation=True)
    cfg = ArticulationCfg(build_pk_chain=False)
    desc, _ = articulation_desc_from_usd(cfg, source_path=path)
    assert desc.name == Path(path).stem


def test_whole_scene_keeps_constraints_and_remaps_overridden_body_uid(tmp_path):
    path = _write_stage(tmp_path)
    stage = Usd.Stage.Open(path)
    # The standalone pair needs a solver-controlled endpoint.
    UsdPhysics.RigidBodyAPI(
        stage.GetPrimAtPath("/World/child")
    ).GetKinematicEnabledAttr().Set(False)
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/joint")
    joint.CreateBody0Rel().SetTargets(["/World/base"])
    joint.CreateBody1Rel().SetTargets(["/World/child"])
    stage.GetRootLayer().Save()
    sim = _declaration_manager()
    assets = sim.add_usd("room", path, asset_cfgs={"/World/child": {"uid": "door"}})
    scene = sim._spawn_scene.builder.add_scene.call_args.args[0]
    assert scene.constraints[0].body0 == "base"
    assert scene.constraints[0].body1 == "door"
    assert set(assets) == {"/World/base", "/World/child"}


@pytest.mark.parametrize(
    "overrides",
    [
        {"shape": {"shape_type": "Mesh", "fpath": "different.usd"}},
        {"usd": {"prim_path": "/ignored"}},
    ],
)
def test_whole_scene_rejects_ignored_per_asset_source_selection(tmp_path, overrides):
    sim = _declaration_manager()
    with pytest.raises(ValueError, match="source|selection"):
        sim.add_usd(
            "room", _write_stage(tmp_path), asset_cfgs={"/World/base": overrides}
        )
    sim._spawn_scene.builder.add_scene.assert_not_called()


def test_usdz_routes_through_the_usd_import_contract():
    from embodichain.lab.sim.sim_manager import _is_usd_path as manager_is_usd
    from embodichain.lab.sim.spawn.descriptors import _is_usd_path as descriptor_is_usd

    assert manager_is_usd("asset.USDZ")
    assert descriptor_is_usd("asset.USDZ")


def test_older_dexsim_usd_contract_fails_before_reading_source(monkeypatch):
    import dexsim.kit.usd

    monkeypatch.setattr(dexsim.kit.usd, "parse_stage", lambda stage: None)
    with pytest.raises(RuntimeError, match="compatible DexSim"):
        rigid_desc_from_usd(
            RigidObjectCfg(uid="old", shape=MeshCfg(fpath="absent.usd"))
        )
