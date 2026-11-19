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

"""Scene fitting and participant contracts independent of names/local assets."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from embodichain.toolkits.skill_fitting._participants import select_participants
from embodichain.toolkits.skill_fitting._robot import RobotProfile
from embodichain.toolkits.skill_fitting._scene import (
    _load_pickup_defaults,
    _hand_geometry,
    _tcp_z_half_turn,
    load_scene,
    prepare_scene_grasps,
    rank_grasps,
)


def test_scene_pickup_defaults_are_explicit_validated_and_digest_bound(
    tmp_path,
) -> None:
    import hashlib

    source = tmp_path / "scene_assets.yaml"
    source.write_text("objects: {}\n")
    cfg, files = _load_pickup_defaults(tmp_path)
    assert cfg.grasp_world_z_offset_m == 0 and cfg.replay_interpolation_factor == 1
    source.write_text(
        "objects: {}\nskill_fitting:\n  pick_up:\n"
        "    grasp_world_z_offset_m: -0.01\n    replay_interpolation_factor: 2\n"
    )
    cfg, files = _load_pickup_defaults(source)
    assert cfg.grasp_world_z_offset_m == -0.01 and cfg.replay_interpolation_factor == 2
    assert files == {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
    for bad in [
        "replay_interpolation_factor: 0",
        "grasp_world_z_offset_m: .nan",
        "replay_factor_typo: 2",
    ]:
        source.write_text("skill_fitting:\n  pick_up:\n    " + bad + "\n")
        with pytest.raises(ValueError):
            _load_pickup_defaults(source)


@pytest.fixture
def dual_profile(tmp_path: Path) -> RobotProfile:
    declarations = ['<link name="base"/>']
    control_parts = {}
    solvers = {}
    commands = {}
    names = []
    for arm, hand in (("boom", "clamp"), ("wrist_drive", "pincer")):
        declarations += [
            f'<link name="{arm}"/>',
            f'<joint name="{arm}_joint" type="revolute"><parent link="base"/><child link="{arm}"/></joint>',
        ]
        control_parts[arm] = [f"{arm}_joint"]
        control_parts[hand] = [f"{hand}_a", f"{hand}_b"]
        names.extend(control_parts[arm] + control_parts[hand])
        solvers[arm] = {"urdf_path": "robot.urdf", "end_link_name": arm}
        commands[hand] = {"open": [0.0, 0.0], "grasp": [0.03, 0.03]}
        for joint in control_parts[hand]:
            declarations += [
                f'<link name="{joint}"/>',
                f'<joint name="{joint}" type="prismatic"><parent link="{arm}"/><child link="{joint}"/></joint>',
            ]
    (tmp_path / "robot.urdf").write_text(
        '<robot name="fixture">' + "".join(declarations) + "</robot>"
    )
    (tmp_path / "profile.json").write_text(
        json.dumps(
            dict(
                joint_names=names,
                control_parts=control_parts,
                solvers=solvers,
                hand_commands=commands,
                participants={"primary": {"motion": "boom", "grasp": "clamp"}},
                lower=[-1.0] * len(names),
                upper=[1.0] * len(names),
                velocity_limits=[1.0] * len(names),
                acceleration_limits=[10.0] * len(names),
                calibration_revision="independent_fixture",
            )
        )
    )
    return RobotProfile.load(tmp_path / "profile.json")


@pytest.mark.parametrize(
    "columns,expected", [((1, 2), "boom"), ((4, 5), "wrist_drive")]
)
def test_auto_participant_follows_configured_hand_event_not_default_role(
    dual_profile, columns, expected
) -> None:
    qpos = np.zeros((30, 6))
    event = np.r_[np.zeros(8), np.linspace(0, 0.03, 12), np.full(10, 0.03)]
    qpos[:, list(columns)] = event[:, None]
    selected, report = select_participants(
        dual_profile, qpos, skill_id="pick_up", selection="auto", phase_hints=None
    )
    assert selected["primary"]["motion"] == expected
    assert report["mode"] == "auto"


def test_ambiguous_participants_require_explicit_selection(dual_profile) -> None:
    qpos = np.zeros((30, 6))
    qpos[:, [1, 2, 4, 5]] = np.linspace(0, 0.03, 30)[:, None]
    with pytest.raises(ValueError, match="Cannot automatically identify"):
        select_participants(
            dual_profile, qpos, skill_id="pick_up", selection="auto", phase_hints=None
        )
    selected, _ = select_participants(
        dual_profile,
        qpos,
        skill_id="pick_up",
        selection="wrist_drive",
        phase_hints=None,
    )
    assert selected == {"primary": {"motion": "wrist_drive", "grasp": "pincer"}}


def test_runtime_transition_connects_postures_with_bounded_joint_derivatives(
    dual_profile,
) -> None:
    from embodichain.toolkits.skill_fitting._execution import _connection

    start = torch.zeros(6, dtype=torch.float64)
    stop = torch.tensor([0.6, 0.02, 0.02, -0.4, 0.01, 0.01], dtype=torch.float64)
    dt = 0.01
    path = _connection(start, stop, dual_profile, dt, 0.25)
    torch.testing.assert_close(path[0], start)
    torch.testing.assert_close(path[-1], stop)
    velocity = torch.diff(path, dim=0) / dt
    acceleration = torch.diff(velocity, dim=0) / dt
    assert bool(
        (
            velocity.abs().max(0).values <= torch.tensor(dual_profile.velocity_limits)
        ).all()
    )
    assert bool(
        (
            acceleration.abs().max(0).values
            <= torch.tensor(dual_profile.acceleration_limits)
        ).all()
    )
    assert velocity[[0, -1]].abs().max() < 0.002


def test_native_pickup_grounds_relative_grasp_against_current_object_pose() -> None:
    from embodichain.toolkits.skill_fitting._execution import _invocation

    pose = torch.eye(4)
    pose[:3, :3] = torch.tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    pose[:3, 3] = torch.tensor([0.5, -0.1, 0.72])
    relative = torch.eye(4)
    relative[:3, 3] = torch.tensor([0.03, 0.02, -0.01])
    obj = SimpleNamespace(
        uid="target",
        get_local_pose=lambda **_: pose[None],
        get_vertices=lambda **_: torch.tensor([[[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]]),
        get_triangles=lambda **_: torch.tensor([[[0, 1, 2]]]),
    )
    engine = SimpleNamespace(
        device=torch.device("cpu"),
        make_invocation=lambda skill, goal, **kwargs: (goal, kwargs),
    )
    goal, kwargs = _invocation(
        engine,
        dict(
            skill_id="pick_up",
            goal=dict(
                kind="object_grasp", object_id="target", object_to_tcp=relative.tolist()
            ),
            control_parts={"primary": {"motion": "custom_arm", "grasp": "custom_hand"}},
            options=dict(lift_height=0.12),
            motion_policy=dict(strategy="motion_gen", sample_count=80, control_dt=0.01),
        ),
        {"target": obj},
    )
    torch.testing.assert_close(goal.grasp_xpos, pose @ relative)
    torch.testing.assert_close(goal.object_pose, pose)
    assert goal.semantics.entity_id == "target"
    assert kwargs["control_parts"]["primary"]["grasp"] == "custom_hand"
    assert kwargs["skill_options"].lift_height == 0.12


@pytest.mark.parametrize("both_participants", [False, True])
def test_native_sequence_holds_loaded_gripper_command_during_other_arm_motion(
    dual_profile, both_participants
) -> None:
    from embodichain.toolkits.skill_fitting._execution import _bound_command_path

    # The first hand is loaded: its actual opening is 0.02, while the commanded
    # full closure is 0.03. The second manipulator must not release that preload.
    previous = torch.tensor([0.4, 0.03, 0.03, 0.0, 0.0, 0.0])
    planned = torch.tensor(
        [[0.39, 0.02, 0.02, 0.0, 0.0, 0.0], [0.39, 0.02, 0.02, 0.2, 0.03, 0.03]]
    )
    participants = {"primary": {"motion": "wrist_drive", "grasp": "pincer"}}
    if both_participants:
        participants["secondary"] = {"motion": "boom", "grasp": "clamp"}
    actual = _bound_command_path(planned, previous, dual_profile, participants)
    expected = (
        planned
        if both_participants
        else torch.tensor(
            [[0.4, 0.03, 0.03, 0.0, 0.0, 0.0], [0.4, 0.03, 0.03, 0.2, 0.03, 0.03]]
        )
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_profile_physics_does_not_override_kinematics(dual_profile, tmp_path) -> None:
    profile_path = tmp_path / "profile.json"
    raw = json.loads(profile_path.read_text())
    physics = dict(
        root_props=dict(fixed_base=True, min_position_iters=32, min_velocity_iters=8),
        joint_drive_props=dict(stiffness={"clamp": 500.0}, damping={"clamp": 50.0}),
        link_attrs={
            "pad": dict(
                link_names_expr=["clamp_a"], attrs=dict(mass_props=dict(mass=0.064))
            )
        },
    )
    raw["simulation"] = physics
    profile_path.write_text(json.dumps(raw))
    loaded = RobotProfile.load(profile_path)
    assert loaded.metadata()["simulation"] == physics
    assert loaded.solvers == dual_profile.solvers
    assert loaded.control_parts == dual_profile.control_parts
    raw["simulation"]["init_pos"] = [0, 0, 1]
    profile_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="simulation permits only"):
        RobotProfile.load(profile_path)


def test_scene_keeps_local_mesh_and_exported_world_pose_separate(
    tmp_path: Path,
) -> None:
    import trimesh

    mesh = trimesh.creation.box(extents=(0.02, 0.03, 0.04))
    mesh.export(tmp_path / "asset.obj")
    pose = [[0, -1, 0, 2], [1, 0, 0, 3], [0, 0, 1, 4], [0, 0, 0, 1]]
    (tmp_path / "scene_assets.yaml").write_text(
        "objects:\n  target:\n    shape:\n      shape_type: Mesh\n      fpath: asset.obj\n"
        "  support:\n    graspable: false\n    shape:\n      shape_type: Mesh\n      fpath: asset.obj\n"
    )
    (tmp_path / "export_report.json").write_text(
        json.dumps(
            {"target": {"anchor": pose}, "support": {"anchor": np.eye(4).tolist()}}
        )
    )
    objects, files = load_scene(tmp_path)
    assert len(objects) == 1
    np.testing.assert_allclose(objects[0].vertices.min(0), [-0.01, -0.015, -0.02])
    np.testing.assert_allclose(objects[0].pose, pose, atol=1e-12)
    assert set(Path(p).name for p in files) == {
        "asset.obj",
        "export_report.json",
        "scene_assets.yaml",
    }
    exported, _ = load_scene(tmp_path, include_ungraspable=True)
    assert [obj.identifier for obj in exported] == ["target", "support"]
    np.testing.assert_allclose(exported[1].pose, np.eye(4))
    (tmp_path / "export_report.json").unlink()
    with pytest.raises(ValueError, match="world-from-object"):
        load_scene(tmp_path)


def test_grasp_ranking_excludes_invalid_and_compares_full_pose() -> None:
    from embodichain.lab.sim.atomic_skills.affordance_sampling import (
        AffordancePoseCandidates,
    )

    poses = torch.eye(4).repeat(3, 1, 1)
    poses[1, 0, 3] = 0.01
    poses[1, :3, :3] = torch.diag(torch.tensor([-1.0, -1.0, 1.0]))
    poses[2, 0, 3] = 0.02
    candidates = AffordancePoseCandidates.from_rows(
        [(poses, torch.tensor([float("inf"), 0, 0]))], device=torch.device("cpu")
    )
    # The exact pose is invalid. The 2 cm candidate beats a 1 cm candidate
    # rotated by pi, under the declared 5 cm/radian orientation weight.
    assert rank_grasps(candidates, np.eye(4), 0.05) == [2, 1]


def test_half_turn_preserves_runtime_tcp_origin_and_qualifies_its_actual_surfaces() -> (
    None
):
    """A tilted, offset contact calibration must not change the symmetry axis."""
    from scipy.spatial.transform import Rotation
    from embodichain.lab.sim.atomic_skills.affordance_sampling import (
        AffordancePoseCandidates,
    )
    from embodichain.toolkits.graspkit.pg_grasp.fixed_geometry import (
        FixedGripperCollisionGeometry,
    )

    calibration = np.eye(4)
    calibration[:3, :3] = Rotation.from_euler("xyz", [0.2, 0.3, 0.4]).as_matrix()
    calibration[:3, 3] = [0.01, -0.02, -0.05]
    points = torch.tensor([[0.01, 0.02, 0.03], [-0.04, 0.05, -0.06]])
    geometry = FixedGripperCollisionGeometry(
        points, {"grasp_from_tcp": calibration.tolist()}
    )
    variant = _tcp_z_half_turn(geometry)
    contact = np.eye(4)
    contact[:3, :3] = Rotation.from_euler("xyz", [-0.4, 0.5, -0.6]).as_matrix()
    contact[:3, 3] = [1.0, 2.0, 3.0]
    original_tcp = contact @ calibration
    half_turn = np.diag([-1.0, -1.0, 1.0, 1.0])
    expected_tcp = original_tcp @ half_turn
    actual_tcp = variant.canonical_to_tcp_poses(torch.tensor(contact)).numpy()
    np.testing.assert_allclose(actual_tcp, expected_tcp, atol=1e-12)
    np.testing.assert_allclose(actual_tcp[:3, 3], original_tcp[:3, 3])
    np.testing.assert_allclose(actual_tcp[:3, 2], original_tcp[:3, 2])
    # Expected physical surface points: take the original points into TCP,
    # rotate the whole hand about that local TCP Z, then place it in world.
    tcp_from_contact = np.linalg.inv(calibration)
    tcp_points = points.numpy() @ tcp_from_contact[:3, :3].T + tcp_from_contact[:3, 3]
    expected_points = tcp_points @ expected_tcp[:3, :3].T + expected_tcp[:3, 3]
    actual_points = variant.points.numpy() @ contact[:3, :3].T + contact[:3, 3]
    np.testing.assert_allclose(actual_points, expected_points, atol=2e-8)
    np.testing.assert_array_equal(geometry.points, points)
    np.testing.assert_allclose(geometry.provenance["grasp_from_tcp"], calibration)
    # An observed pose ten degrees from the half turn must choose it rather
    # than the physically equivalent pose 170 degrees away, in a rotated world.
    reference = expected_tcp.copy()
    reference[:3, :3] = (
        expected_tcp[:3, :3] @ Rotation.from_euler("z", 10, degrees=True).as_matrix()
    )
    candidates = AffordancePoseCandidates.from_rows(
        [
            (
                torch.tensor(np.stack((original_tcp, actual_tcp)), dtype=torch.float32),
                torch.zeros(2),
            )
        ],
        device=torch.device("cpu"),
    )
    assert rank_grasps(candidates, reference, 0.05) == [1, 0]


def test_contact_grounding_preserves_a_virtual_runtime_tcp(tmp_path: Path) -> None:
    """The configured TCP is 56 mm behind the geometry's contact centre."""
    parts = [
        '<link name="base"/><link name="hand"/>',
        '<joint name="pivot" type="revolute"><parent link="base"/><child link="hand"/><axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>',
    ]
    for name, x, axis in (("one", -0.04, 1), ("two", 0.04, -1)):
        parts.append(
            f'<link name="{name}"><collision><origin xyz="0 0 0.05"/><geometry><box size="0.01 0.02 0.10"/></geometry></collision></link>'
        )
        parts.append(
            f'<joint name="{name}" type="prismatic"><parent link="hand"/><child link="{name}"/><origin xyz="{x} 0 0.06"/><axis xyz="{axis} 0 0"/><limit lower="0" upper="0.03" effort="1" velocity="1"/></joint>'
        )
    path = tmp_path / "hand.urdf"
    path.write_text('<robot name="calibrated_hand">' + "".join(parts) + "</robot>")
    tcp = np.eye(4)
    tcp[2, 3] = 0.10
    profile = RobotProfile(
        joint_names=("pivot", "one", "two"),
        control_parts={"drive": ["pivot"], "clamp": ["one", "two"]},
        solvers={
            "drive": {
                "urdf_path": str(path),
                "end_link_name": "hand",
                "tcp": tcp.tolist(),
            }
        },
        hand_commands={"clamp": {"open": [0.0, 0.0], "grasp": [0.03, 0.03]}},
    )
    geometry, model = _hand_geometry(
        profile, {"motion": "drive", "grasp": "clamp"}, np.zeros(3), 0.004
    )
    contact = torch.eye(4)
    contact[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
    tcp_pose = geometry.canonical_to_tcp_poses(contact)
    np.testing.assert_allclose(tcp_pose[:3, 3], [1.0, 2.0, 2.944], atol=1e-6)
    assert geometry.points[:, 2].max() == pytest.approx(0.004, abs=1e-6)
    assert model["max_opening_width"] == pytest.approx(0.06)


@pytest.mark.parametrize(
    "overrides,expected_angle,offset",
    [
        ({}, 10.0, 0.0),
        ({"grasp_filter_ground_collision": True}, 170.0, 0.0),
        ({"grasp_approach_deviation_rad": np.pi / 6}, 170.0, 0.0),
        ({}, 10.0, -0.01),
    ],
)
def test_scene_selects_near_half_turn_only_when_its_query_accepts_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overrides, expected_angle, offset
) -> None:
    """Exercise host configuration, local symmetry and validity masks together."""
    from scipy.spatial.transform import Rotation
    import trimesh
    from embodichain.lab.sim.atomic_skills.affordance import AntipodalAffordance
    from embodichain.lab.sim.atomic_skills.affordance_sampling import (
        AffordancePoseCandidates,
    )
    from embodichain.lab.sim.atomic_skills.fitting import SkillFitCfg
    from embodichain.toolkits.graspkit.pg_grasp.fixed_geometry import (
        FixedGripperCollisionGeometry,
    )
    from embodichain.toolkits.skill_fitting import _scene

    mesh = trimesh.creation.box(extents=(0.02, 0.02, 0.02))
    mesh.export(tmp_path / "object.obj")
    (tmp_path / "scene_assets.yaml").write_text(
        json.dumps(
            {
                "objects": {
                    "target": {
                        "shape": {"shape_type": "Mesh", "fpath": "object.obj"},
                        "pose": np.eye(4).tolist(),
                    }
                }
            }
        )
    )
    calibration = np.eye(4)
    calibration[:3, :3] = Rotation.from_euler("y", 0.2).as_matrix()
    calibration[:3, 3] = [0.01, 0.02, -0.05]
    geometry = FixedGripperCollisionGeometry(
        torch.tensor([[0.01, 0.02, 0.03]]),
        {
            "grasp_from_tcp": calibration.tolist(),
            "files_sha256": {},
        },
    )
    model = dict(model_id="test_clamp", min_opening_width=0.001, max_opening_width=0.1)
    monkeypatch.setattr(_scene, "_hand_geometry", lambda *args: (geometry, model))

    source_tcp = np.eye(4)
    source_tcp[2, 3] = 0.1
    far_tcp = source_tcp.copy()
    far_tcp[:3, :3] = Rotation.from_euler("z", 170, degrees=True).as_matrix()
    canonical_pose = torch.tensor(
        far_tcp @ np.linalg.inv(calibration), dtype=torch.float32
    )

    def qualified_pool(self, generator, object_poses, approach_direction):
        # Independent geometric-query oracle: the near roll needs a wider
        # approach cone and is blocked by the optional inferred support plane.
        half_turn = (
            generator.collision_geometry.provenance.get("tcp_local_z_rotation_rad")
            is not None
        )
        valid = not half_turn or (
            not generator.collision_cfg.filter_ground_collision
            and generator.algorithm_cfg.approach_deviation_angle > np.pi / 4
        )
        pose = generator.collision_geometry.canonical_to_tcp_poses(canonical_pose)
        return AffordancePoseCandidates.from_rows(
            [(pose[None], torch.tensor([0.0 if valid else float("inf")]))],
            device="cpu",
        )

    monkeypatch.setattr(AntipodalAffordance, "get_grasp_candidates", qualified_pool)

    class RecordedRobot:
        device = torch.device("cpu")

        def get_joint_ids(self, name):
            return [0] if name == "drive" else [1]

        def compute_fk(self, qpos, name):
            poses = torch.eye(4).repeat(len(qpos), 1, 1)
            poses[:, 2, 3] = qpos[:, 0]
            return poses

    qpos = np.column_stack(
        (
            np.r_[np.linspace(0.25, 0.1, 8), np.full(22, 0.1)],
            np.r_[np.zeros(10), np.linspace(0, 0.03, 10), np.full(10, 0.03)],
        )
    )
    profile = SimpleNamespace(hand_commands={"clamp": {"open": [0.0], "grasp": [0.03]}})
    goals, metadata, _ = prepare_scene_grasps(
        tmp_path,
        "target",
        RecordedRobot(),
        profile,
        {"motion": "drive", "grasp": "clamp"},
        qpos,
        SkillFitCfg(**overrides),
        None,
        grasp_world_z_offset_m=offset,
    )
    actual = Rotation.from_matrix(goals[0].grasp_xpos.numpy()[:3, :3]).magnitude()
    assert np.rad2deg(actual) == pytest.approx(expected_angle, abs=1e-5)
    np.testing.assert_allclose(
        goals[0].grasp_xpos.numpy()[:3, 3],
        source_tcp[:3, 3] + [0, 0, offset],
        atol=1e-7,
    )
    np.testing.assert_allclose(goals[0].object_pose.numpy(), np.eye(4))
    np.testing.assert_allclose(
        np.array(metadata["nominal_grasp_xpos"])[0, :3, 3], source_tcp[:3, 3], atol=1e-7
    )
    assert metadata["grasp_world_z_offset_m"] == offset
    assert metadata["ground_collision_filter"] == overrides.get(
        "grasp_filter_ground_collision", False
    )


@pytest.mark.parametrize("completed", [True, False])
def test_sequence_trace_keeps_submitted_commands_and_measured_states_causal(
    tmp_path, completed
) -> None:
    import h5py
    from embodichain.toolkits.skill_fitting._sequence_trace import _SequenceTrace

    robot = SimpleNamespace(joint_names=["a", "b"], step=0)
    robot.get_qpos = lambda: torch.tensor(
        [[robot.step, -robot.step]], dtype=torch.float32
    )
    robot.get_qvel = lambda: torch.tensor([[1.0, -1.0]])
    robot.get_local_pose = lambda: torch.tensor([[0.0, 0, 0, 0, 0, 0, 1.0]])
    obj = SimpleNamespace(
        get_local_pose=lambda: torch.tensor(
            [[0.0, 0, 0.7 + robot.step * 0.001, 0, 0, 0, 1.0]]
        )
    )
    sim = SimpleNamespace(
        get_rigid_object_uid_list=lambda: ["cup"], get_rigid_object=lambda uid: obj
    )
    env = SimpleNamespace(
        robot=robot, sim=sim, expert_action_spec=SimpleNamespace(joint_names=["a"])
    )
    trace = _SequenceTrace(
        tmp_path / "trace.h5", env, SimpleNamespace(joint_names=["b", "a"]), 0.01
    )
    for index in range(1, 301):
        robot.step = index
        trace.append(torch.tensor([index + 1000.0]))
    trace.close(completed=completed)
    with h5py.File(tmp_path / "trace.h5") as data:
        np.testing.assert_array_equal(
            data["commands/qpos"][:, 0], np.arange(1001, 1301)
        )
        np.testing.assert_array_equal(data["states/qpos"][:, 0], -np.arange(301))
        np.testing.assert_array_equal(data["states/qpos"][:, 1], np.arange(301))
        np.testing.assert_allclose(
            data["states/objects/cup/pose"][:, 2],
            0.7 + np.arange(301) * 0.001,
            atol=1e-7,
        )
        assert data.attrs["completed"] == completed
        assert data.attrs["command_count"] == 300
