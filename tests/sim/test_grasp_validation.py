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

"""Grasp metric contracts for fixed attempt budgets and physical retention."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from embodichain.lab.sim.grasp import validation as grasp_validation
from embodichain.lab.sim.cfg import RigidObjectCfg
from embodichain.lab.sim.grasp.validation import GraspValidator, _result
from embodichain.robot_registry import resolve_gripper_defaults


def _trace(length=421):
    row = np.zeros((length, 19))
    row[:, 3:12] = np.eye(3).reshape(9)
    row[:, 12:15] = [0.09, 1, 1]
    return row


def test_disturbance_precedes_hold_and_loss_is_final():
    row = _trace()
    row[65:, 3:12] = np.diag([-1, -1, 1]).reshape(9)
    kept = _result(row, True, False)
    assert kept["success"] and kept["disturbance"]["rotation_deg"] == 180
    assert kept["hold"]["rotation_deg"] == 0

    # Briefly lose the object in the Y leg, then regain both pad contacts.
    row[95, 13:15] = 0
    lost = _result(row, True, False)
    assert not lost["success"] and lost["failure_stage"] == "disturbance"
    assert lost["disturbance"]["drop_s"] == pytest.approx(0.95)
    assert lost["hold"] is None and not any(lost["retention"].values())
    assert _result(row, True, True)["status"] == "missing"
    assert _result(row[:-1], True, False)["status"] == "missing"

    # Loss after the completed return belongs to the following hold window.
    row = _trace()
    row[120 + 70 :, 13:15] = 0
    lost = _result(row, True, False)
    assert lost["disturbance"]["success"] and lost["failure_stage"] == "hold"
    assert lost["hold"]["drop_s"] == pytest.approx(0.70)
    assert lost["retention"] == {"0.5": True, "1.0": False, "3.0": False}


def test_dexuni_contact_evidence_uses_distance_only_for_unmeasured_rows():
    validator = GraspValidator.__new__(GraspValidator)
    validator._geometry_contacts = True
    validator.target_id = torch.tensor([[1]])
    validator.pad_ids = torch.tensor([[2, 3]])
    validator.support_id = torch.tensor([[4]])
    data = dict(
        user_ids=torch.tensor([[[1, 2], [3, 1], [1, 4]]]),
        is_valid=torch.ones(1, 3, dtype=torch.bool),
        impulse_valid=torch.tensor([[True, False, False]]),
        impulse=torch.tensor([[0.0, 0.0, 0.0]]),
        distance=torch.tensor([[-0.001, 0.0, 0.001]]),
    )
    validator.sensor = SimpleNamespace(
        update=lambda: None,
        get_data=lambda: data,
        get_latest_contact_loss_mask=lambda: torch.tensor([False]),
    )
    fingers, support, lost = validator._contacts()
    assert fingers.tolist() == [[False, True]]
    assert not support.item() and not lost.item()
    data["distance"][0, 1] = float("nan")
    assert validator._contacts()[2].item()


@pytest.mark.parametrize("keeps_tipping", [False, True])
def test_targets_settle_in_the_same_half_second(keeps_tipping: bool) -> None:
    validator = GraspValidator.__new__(GraspValidator)
    elapsed = 0

    def advance(step: int) -> None:
        nonlocal elapsed
        elapsed += step

    def poses(*, to_matrix: bool) -> torch.Tensor:
        result = torch.eye(4).repeat(2, 1, 1)
        # Each arena moves and returns within a different observation window.
        if elapsed == 120:
            result[0, 0, 3] = 0.01
        if elapsed == 170 or (keeps_tipping and elapsed % 50 == 20):
            result[1, 1:3, 1:3] = torch.tensor([[0, -1], [1, 0]])
        return result

    validator.sim = SimpleNamespace(num_envs=2, device="cpu", update=advance)
    validator.obj = SimpleNamespace(get_local_pose=poses)
    if keeps_tipping:
        with pytest.raises(ValueError, match="did not settle"):
            validator._settled_pose()
        assert elapsed == 500
    else:
        torch.testing.assert_close(validator._settled_pose(), torch.eye(4))
        assert 220 <= elapsed <= 250


def test_rotation_shift_composes_in_arena_axes_and_clears_table() -> None:
    validator = GraspValidator.__new__(GraspValidator)
    validator.sim = SimpleNamespace(device="cpu", num_envs=2, update=Mock())
    poses = torch.eye(4).repeat(2, 1, 1)
    # Arena 0 starts at Rz(90°) and arena 1 at Ry(90°).
    poses[0, :3, :3] = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    poses[1, :3, :3] = torch.tensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    poses[:, :3, 3] = torch.tensor([[0.3, 0.1, 0.4], [-0.2, 0.2, 0.6]])
    validator.obj = Mock()
    validator.obj.get_local_pose.return_value = poses.clone()
    shape_pose = torch.eye(4)
    shape_pose[:2, 3] = torch.tensor([0.01, -0.03])
    validator.shapes = [
        SimpleNamespace(
            shape_type=SimpleNamespace(name="BOX"),
            local_pose=shape_pose,
            half_extents=torch.tensor([0.02, 0.04, 0.07]),
        )
    ]
    rotation = grasp_validation._rotation_shift_matrix([90, 0, 90])
    validator._place_rotated_target(rotation)
    placed = validator.obj.set_local_pose.call_args.args[0]
    torch.testing.assert_close(placed[:, :2, 3], poses[:, :2, 3])
    # After turns about arena X then Z, local +X and +Y point up, respectively.
    expected_rot = torch.tensor(
        [[[0, 0, 1], [0, -1, 0], [1, 0, 0]], [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(placed[:, :3, :3], expected_rot)
    torch.testing.assert_close(placed[:, 2, 3], torch.tensor([0.01, 0.07]))
    separated = validator.obj.set_local_pose.call_args_list[0].args[0]
    assert (validator._clearance(separated) > 0).all()
    validator.sim.update.assert_called_once()
    linear, angular = validator.obj.set_velocity.call_args.args
    assert not linear.any() and not angular.any()


def test_rotation_shift_requires_three_finite_angles() -> None:
    for invalid in ([0, 90], [0, float("nan"), 0]):
        with pytest.raises(ValueError, match="rotation_shift_deg"):
            grasp_validation._rotation_shift_matrix(invalid)


@pytest.fixture
def trial_validator() -> GraspValidator:
    validator = GraspValidator.__new__(GraspValidator)
    validator._show_trials = False
    validator.enable_disturbance = True
    validator.close_wait_steps = 20
    validator.initial = validator.root_tcp = validator.root_contact = torch.eye(4)
    validator.axes = [0, 1, 2]
    validator.opened = torch.tensor([0.0, 0.0, 0.0, 0.2])
    validator.closed = torch.tensor([0.0, 0.0, 0.0, 1.0])
    validator.hand, validator.obj, validator.sensor = Mock(), Mock(), Mock()
    validator.cfg = SimpleNamespace(root_link="hand")
    validator.com = torch.zeros(3)
    validator.hand.get_link_pose.return_value = validator.initial[None]
    validator.obj.get_local_pose.return_value = validator.initial[None]
    validator._clearance = lambda pose: torch.ones(1)
    no_contact = torch.zeros(1, dtype=torch.bool)
    validator._contacts = lambda: (
        torch.ones(1, 2, dtype=torch.bool),
        no_contact,
        no_contact,
    )
    validator.sim = Mock(
        num_envs=1,
        device="cpu",
        viewer=None,
        arena_offsets=torch.zeros(1, 3),
        is_viewer_open=False,
    )
    return validator


def test_fixed_mount_offset_places_calibrated_root_at_pregrasp(
    trial_validator: GraspValidator,
) -> None:
    mount_from_root = torch.tensor(
        [
            [0.0, -1.0, 0.0, 0.03],
            [1.0, 0.0, 0.0, -0.01],
            [0.0, 0.0, 1.0, 0.02],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    trial_validator._root_from_mount = torch.linalg.inv(mount_from_root)
    tcp = torch.eye(4)
    tcp[:3, 3] = torch.tensor([0.1, 0.2, 0.3])
    trial_validator.validate(tcp[None].numpy())
    (mount_pose,) = trial_validator.hand.set_local_pose.call_args.args
    expected = tcp.clone()
    expected[2, 3] -= 0.08
    torch.testing.assert_close(mount_pose[0] @ mount_from_root, expected)


@pytest.mark.parametrize("enabled", [True, False])
def test_close_wait_then_lift_and_optional_world_xy_round_trips(
    trial_validator: GraspValidator,
    enabled: bool,
) -> None:
    validator = trial_validator
    commands = []
    validator.sim.update.side_effect = lambda step: commands.append(
        validator.hand.set_qpos.call_args.args[0][0]
    )

    validator.enable_disturbance = enabled
    # The hand is rotated 90° around world Z. Its local stage axes differ from
    # the world X and Y translation directions requested by the protocol.
    pose = np.eye(4)
    pose[:2, :2] = [[0, -1], [1, 0]]
    (trial,) = validator.validate(pose[None])
    # Each row is the three local stage positions followed by the finger command.
    assert commands[80 - 1].tolist() == pytest.approx([0, 0, 0.08, 0.6])
    np.testing.assert_allclose(
        torch.stack(commands[100 - 1 : 120]), np.tile([0, 0, 0.08, 1], (21, 1))
    )
    assert commands[160 - 1].tolist() == pytest.approx([0, 0, 0.13, 1])
    world_xy = torch.stack(commands).numpy()[:, :2] @ pose[:2, :2].T
    # Lift ends at step 200; each leg takes 30 steps, followed by the 3 s hold.
    if enabled:
        np.testing.assert_allclose(
            world_xy[[199, 229, 259, 289, 319]],
            [[0, 0], [0.03, 0], [0, 0], [0, 0.03], [0, 0]],
            atol=1e-7,
        )
        assert len(commands) == 620
        assert trial["disturbance"]["duration_s"] == pytest.approx(1.2)
    else:
        np.testing.assert_allclose(world_xy, 0)
        assert len(commands) == 500 and trial["disturbance"] is None
        assert trial["peak_acceleration_m_s2"] is None
    np.testing.assert_allclose(world_xy[-300:], 0)
    assert trial["hold"]["duration_s"] == pytest.approx(3.0)
    validator.sim.open_viewer.assert_not_called()


def test_window_opens_at_first_pregrasp_and_is_reused(
    trial_validator: GraspValidator, monkeypatch: pytest.MonkeyPatch
) -> None:
    validator = trial_validator
    validator._show_trials = True
    viewer = Mock()

    def open_viewer() -> bool:
        # The first visible frame has the hand open, 8 cm before the candidate.
        (hand_pose,) = validator.hand.set_local_pose.call_args.args
        torch.testing.assert_close(hand_pose[0, :3, 3], torch.tensor([0, 0, -0.08]))
        (joints,) = validator.hand.set_qpos.call_args.args
        torch.testing.assert_close(joints[0], validator.opened)
        (object_pose,) = validator.obj.set_local_pose.call_args.args
        torch.testing.assert_close(object_pose[0], validator.initial)
        validator.sim.viewer = viewer
        validator.sim.is_viewer_open = True
        return True

    validator.sim.open_viewer.side_effect = open_viewer
    monkeypatch.setattr(grasp_validation.time, "sleep", lambda seconds: None)
    for _ in range(2):
        validator.validate(np.eye(4)[None])
    validator.sim.open_viewer.assert_called_once()
    viewer.set_look_at.assert_called_once()


@pytest.mark.parametrize(
    ("options", "renderer"), [({}, "no-render"), ({"headless": False}, "fast-rt")]
)
def test_validator_starts_without_a_window_and_selects_the_requested_renderer(
    monkeypatch: pytest.MonkeyPatch, options: dict[str, bool], renderer: str
) -> None:
    manager = Mock(side_effect=RuntimeError("stop before native startup"))
    monkeypatch.setattr(grasp_validation, "SimulationManager", manager)
    with pytest.raises(RuntimeError, match="stop before native startup"):
        GraspValidator(RigidObjectCfg(uid="target"), "FrankaPandaHand", **options)
    (cfg,) = manager.call_args.args
    assert cfg.headless
    assert cfg.render_cfg.renderer == renderer


@pytest.mark.parametrize(
    ("field", "targets", "message"),
    [
        ("open_qpos", (0.01, 0.02), "mimic"),
        ("close_qpos", (0.04, 0.03), "mimic"),
        ("close_qpos", (0.041, 0.041), "limits"),
    ],
)
def test_dh_qpos_respects_mimic_and_travel(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    targets: tuple[float, float],
    message: str,
) -> None:
    hand = "DHPGI14080"
    names = resolve_gripper_defaults(hand)["control_parts"]["eef"]
    monkeypatch.setattr(grasp_validation, "SimulationManager", Mock())
    with pytest.raises(ValueError, match=message):
        GraspValidator(
            RigidObjectCfg(uid="target"), hand, **{field: dict(zip(names, targets))}
        )


def test_stage_preserves_relative_meshes_native_limits_and_rounded_pika_opening(
    tmp_path, monkeypatch
):
    import os
    import xml.etree.ElementTree as ET
    from embodichain.data import get_data_path
    from embodichain.robot_registry import resolve_robot_cfg

    hand = resolve_robot_cfg("PikaGripper")
    source = Path(get_data_path(resolve_gripper_defaults("PikaGripper")["fpath"]))
    monkeypatch.chdir(tmp_path)
    Path("assets").symlink_to(source.parent, target_is_directory=True)
    derived = Path("derived/hand.urdf")
    derived.parent.mkdir()
    xml = ET.parse(source)
    for mesh in xml.iter("mesh"):
        mesh.set("filename", str(Path("../assets") / mesh.attrib["filename"]))
    xml.write(derived)
    stage = tmp_path / "stage"
    stage.mkdir()
    hand.fpath = str(derived)
    joints = [
        j.get("name")
        for j in ET.parse(source).findall("joint")
        if j.get("type") != "fixed"
    ]
    hand.qpos_limits = np.array([[0.0, 0.05], [-0.001, 0.05]])
    hand.joint_drive_props.stiffness = {"eef": 1000.0}
    rounded = {name: float(np.float32(0.05)) for name in joints}
    grasp_validation._stage(hand, stage, "default", open_qpos=rounded)
    staged = ET.parse(hand.fpath)
    assert staged.find("joint[@name='stage_mount']/child").get("link") == hand.root_link
    meshes = staged.findall(".//mesh")
    # Native importers normalize parent components before opening a mesh.
    assert meshes and all(
        Path(os.path.normpath(stage / mesh.attrib["filename"])).is_file()
        for mesh in meshes
    )
    assert hand.qpos_limits == dict(zip(joints, [[0.0, 0.05], [-0.001, 0.05]]))
    assert hand.joint_commands["open"] == dict.fromkeys(joints, 0.05)
    assert all(hand.joint_drive_props.stiffness[name] == 1000.0 for name in joints)


def test_native_physx_capsule_preserves_long_axis_and_offset() -> None:
    from dexsim.engine import CapsuleGeometry
    from embodichain.lab.sim.objects.backends.collision import (
        collision_shapes_from_entity,
    )

    capsule = CapsuleGeometry()
    local = np.eye(4, dtype=np.float32)
    local[:3, 3] = [0.05, 0.04, 0.01]
    capsule.local_pose = local
    capsule.radius, capsule.half_height = 0.01, 0.05
    body = Mock()
    body.get_shape_count.return_value = 1
    body.get_shape_geometry.return_value = capsule
    validator = GraspValidator.__new__(GraspValidator)
    validator.shapes = collision_shapes_from_entity(
        SimpleNamespace(get_physical_body=lambda: body), object_uid="target"
    )
    axes = validator.shapes[0].local_pose
    torch.testing.assert_close(axes[:3, 2], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(axes[:3, 3], torch.from_numpy(local[:3, 3]))
    torch.testing.assert_close(validator._clearance(torch.eye(4)[None]), torch.zeros(1))


@pytest.mark.parametrize("axis", ["X", "Y", "Z"])
def test_capsule_clearance_uses_authored_usd_axis(tmp_path: Path, axis: str) -> None:
    from dexsim.spawn import objects_from_usd
    from pxr import Usd, UsdGeom, UsdPhysics
    from embodichain.lab.sim.objects.backends.collision import (
        collision_shapes_from_entity,
    )

    path = tmp_path / "compound.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    body = UsdGeom.Xform.Define(stage, "/Body")
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    capsule = UsdGeom.Capsule.Define(stage, "/Body/Capsule")
    capsule.CreateAxisAttr(axis)
    capsule.CreateRadiusAttr(0.01)
    capsule.CreateHeightAttr(0.1)
    UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())
    cylinder = UsdGeom.Cylinder.Define(stage, "/Body/Cylinder")
    cylinder.CreateRadiusAttr(0.001)
    cylinder.CreateHeightAttr(0.002)
    UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
    stage.GetRootLayer().Save()
    (descriptor,) = objects_from_usd("test", str(path))
    validator = GraspValidator.__new__(GraspValidator)
    validator.shapes = collision_shapes_from_entity(
        SimpleNamespace(object_desc=descriptor), object_uid="target"
    )
    assert len(validator.shapes) == 2
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[:, 2, 3] = 0.06 if axis == "Z" else 0.01
    poses[1, 2, 3] += 0.021
    torch.testing.assert_close(validator._clearance(poses), torch.tensor([0.0, 0.021]))


@pytest.mark.gpu
@pytest.mark.subprocess_sim
@pytest.mark.parametrize("physics", ["default", "newton"])
def test_missed_and_centered_grasps_in_independent_arenas(physics: str) -> None:
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), physics],
        check=True,
        timeout=180,
    )


def _check_native_trials(physics: str) -> None:
    target = RigidObjectCfg.from_dict(
        {
            "uid": "block",
            "shape": {"shape_type": "Cube", "size": [0.065, 0.065, 0.12]},
            "init_pos": [0.0, 0.0, 0.061],
            "attrs": {
                "mass_props": {"mass": 0.015},
                "material_props": {"static_friction": 0.99, "dynamic_friction": 0.97},
            },
        }
    )
    poses = np.tile(np.diag([1.0, -1.0, -1.0, 1.0]), (3, 1, 1))
    poses[:, 0, 3] = [0.15, -0.15, 0.0]
    poses[:, 2, 3] = 0.035
    validator = GraspValidator(target, "FrankaPandaHand", num_envs=3, physics=physics)
    try:
        placements = validator.initial.repeat(3, 1, 1)
        settled, centers = validator.drop_objects(
            np.eye(3)[None], 0.02, return_support_centers=True
        )
        assert settled.shape == (1, 4, 4)
        contact = settled[0, :3, :3] @ centers[0] + settled[0, :3, 3]
        assert abs(contact[2].item()) < 0.01
        assert abs(centers[0, 2].item() + 0.06) < 0.01
        placements[:, 1, 3] += torch.tensor(
            [-0.06, 0.03, 0.06], device=placements.device
        )
        validator.restore_placement(placements)
        torch.testing.assert_close(validator.initial, placements)
        first, second, centered = validator.validate(poses)
        assert first["failure_stage"] == second["failure_stage"] == "lift"
        assert not first["success"] and not second["success"]
        assert centered["disturbance"]["entered"]
        assert centered["disturbance"]["duration_s"] == pytest.approx(1.2)
        if physics == "default":
            assert centered["success"]
    finally:
        validator.close()


if __name__ == "__main__":
    _check_native_trials(sys.argv[1])
