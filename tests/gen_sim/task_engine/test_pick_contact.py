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

"""Native-rest closing order is candidate screening, not grasp evidence."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import trimesh
from types import SimpleNamespace
from unittest.mock import Mock

from embodichain.gen_sim.task_engine._task_program.pick_contact import (
    ClosingContactScreen,
    ContactPart,
)


def _poses(axis: int, values: list[float]) -> np.ndarray:
    poses = np.tile(np.eye(4), (len(values), 1, 1))
    poses[:, axis, 3] = values
    return poses


def _screen(
    body_z: list[float], target_rest: float = 0.001, *, revision: int = 1
) -> ClosingContactScreen:
    body = trimesh.creation.box((0.01, 0.01, 0.01))
    pad = trimesh.creation.box((0.004, 0.02, 0.02))
    parts = (
        ContactPart(
            "knuckle",
            (body,),
            _poses(2, body_z),
            0.001,
            False,
            **({"contact_offset": 0.006} if revision >= 2 else {}),
        ),
        ContactPart(
            "negative_pad",
            (pad,),
            _poses(0, [-0.02, -0.013, -0.01]),
            0.001,
            True,
            **({"contact_offset": 0.006} if revision >= 2 else {}),
        ),
        ContactPart(
            "positive_pad",
            (pad,),
            _poses(0, [0.02, 0.013, 0.01]),
            0.001,
            True,
            **({"contact_offset": 0.006} if revision >= 2 else {}),
        ),
    )
    return ClosingContactScreen(
        parts,
        (trimesh.creation.box((0.02, 0.02, 0.02)),),
        target_rest,
        revision=revision,
        **({"target_contact_offset": 0.003} if revision >= 2 else {}),
    )


def test_positive_stops_at_pad_contact_not_hypothetical_command_end() -> None:
    result = _screen([0.1, 0.1, 0.014]).inspect(np.eye(4))
    assert result["valid"] is True
    assert result["checked_samples"] == 2
    assert result["nominal_closing_fraction"] == 0.5


def test_revision_eight_preserves_revision_seven_native_closing_gates() -> None:
    legacy = _screen([0.1, 0.1, 0.014], revision=7)
    native = _screen([0.1, 0.1, 0.014], revision=8)
    for x in (0.0, 0.01, 0.1):
        pose = np.eye(4)
        pose[0, 3] = x
        assert native.inspect(pose) == legacy.inspect(pose)
    assert native.report()["proposal_source_proxy_deferred"] is False
    assert native.report()["deferred_source_proposal_calls"] == 0
    assert "proposal_source_proxy_deferred" not in legacy.report()


def test_far_closing_parts_skip_exact_distances_without_changing_the_result() -> None:
    screen = _screen([0.1] * 3, revision=7)
    calls = []
    for manager in screen.managers:
        manager.min_distance_other = Mock(wraps=manager.min_distance_other)
        calls.append(manager.min_distance_other)
    pose = np.eye(4)
    pose[0, 3] = 1.0
    result = screen.inspect(pose)
    assert result["reason"] == "bilateral_pad_rest_not_reached"
    assert result["checked_samples"] == 3
    assert sum(call.call_count for call in calls) == 0
    assert screen.inspect(pose) == result


@pytest.mark.parametrize("revision", range(1, 9))
def test_closing_broad_phase_matches_exact_mesh_queries(revision, monkeypatch) -> None:
    base = _screen([0.02, 0.018, 0.014], revision=revision)
    extra = trimesh.creation.box((0.01, 0.03, 0.02))
    extra.apply_translation((0.04, 0.02, 0.025))
    meshes = (trimesh.creation.box((0.02, 0.02, 0.02)), extra)
    kwargs = {"revision": revision}
    if revision >= 2:
        kwargs["target_contact_offset"] = 0.003
    fast = ClosingContactScreen(base.parts, meshes, 0.001, **kwargs)
    exact = ClosingContactScreen(base.parts, meshes, 0.001, **kwargs)
    monkeypatch.setattr(
        exact,
        "_separated_parts",
        lambda pose: np.zeros((exact.steps, len(exact.parts)), dtype=bool),
        raising=False,
    )
    rng = np.random.default_rng(102018)
    poses = [np.eye(4)]
    for _ in range(64):
        pose = trimesh.transformations.rotation_matrix(
            rng.uniform(-np.pi, np.pi), rng.normal(size=3)
        )
        pose[:3, 3] = rng.uniform(-0.06, 0.06, 3)
        poses.append(pose)
    for pose in poses:
        assert fast.inspect(pose) == exact.inspect(pose)


def test_closing_broad_phase_preserves_native_halo_distances() -> None:
    screen = _screen([0.02] * 3, revision=2)
    body = screen.managers[0]
    body.min_distance_other = Mock(wraps=body.min_distance_other)
    result = screen.inspect(np.eye(4))
    assert body.min_distance_other.call_count == 1
    assert result["reason"] == "nonpad_contact_envelope_before_bilateral_pad_rest"
    assert result["minimum_distance_m"] == pytest.approx(0.005)


def test_closing_broad_phase_keeps_exact_queries_near_rest_boundary() -> None:
    screen = _screen([0.0170001] * 3)
    body = screen.managers[0]
    body.min_distance_other = Mock(wraps=body.min_distance_other)
    assert screen.inspect(np.eye(4))["valid"] is True
    assert body.min_distance_other.call_count == 2


def test_native_opening_allows_local_grasp_without_enclosing_remote_geometry() -> None:
    from embodichain.gen_sim.task_engine._task_program.grasp_filter import (
        opening_envelope_mask,
    )

    base = _screen([0.1] * 3, revision=4)
    remote = trimesh.creation.box((0.02, 0.02, 0.02))
    remote.apply_translation((0.20, 0.20, 0.0))
    target = (trimesh.creation.box((0.02, 0.02, 0.02)), remote)
    screen = ClosingContactScreen(
        base.parts, target, 0.001, revision=5, target_contact_offset=0.003
    )
    vertices = torch.tensor(np.concatenate([m.vertices for m in target]))
    assert not opening_envelope_mask(
        vertices, torch.eye(4)[None], torch.eye(4), 0.04
    ).any()
    assert screen.inspect(np.eye(4))["valid"] is True
    assert screen.report()["native_opening_rest_screen"] is True


def test_native_opening_rejects_pad_rest_overlap_before_closing() -> None:
    base = _screen([0.1] * 3, revision=4)
    target = (trimesh.creation.box((0.044, 0.02, 0.02)),)
    old = ClosingContactScreen(
        base.parts, target, 0.001, revision=4, target_contact_offset=0.003
    )
    assert old.inspect(np.eye(4))["valid"] is True
    screen = ClosingContactScreen(
        base.parts, target, 0.001, revision=5, target_contact_offset=0.003
    )
    result = screen.inspect(np.eye(4))
    assert result["valid"] is False
    assert result["reason"] == "pad_rest_at_opening"
    assert result["checked_samples"] == 1


def test_native_depth_offsets_are_measured_from_pad_span_and_pair_rest() -> None:
    base = _screen([0.1] * 3, revision=5)
    parts = tuple(
        ContactPart(
            p.name,
            (trimesh.creation.box((0.004, 0.02, 0.065)),) if p.pad else p.meshes,
            p.tcp_poses,
            p.rest_offset,
            p.pad,
            p.contact_offset,
        )
        for p in base.parts
    )
    legacy = ClosingContactScreen(
        parts,
        (trimesh.creation.box((0.02, 0.02, 0.02)),),
        0.001,
        revision=5,
        target_contact_offset=0.003,
    )
    assert legacy.depth_offsets() == (0.0,)
    screen = ClosingContactScreen(
        parts,
        (trimesh.creation.box((0.02, 0.02, 0.02)),),
        0.001,
        revision=6,
        target_contact_offset=0.003,
    )
    assert screen.depth_offsets() == pytest.approx(
        (0.0, 0.01525, 0.0305, 0.04575, 0.061)
    )


def test_nonpenetrating_body_inside_native_rest_envelope_is_rejected() -> None:
    screen = _screen([0.01696] * 3)
    result = screen.inspect(np.eye(4))
    assert result["valid"] is False
    assert result["reason"] == "nonpad_rest_before_bilateral_pad"
    assert result["minimum_distance_m"] == pytest.approx(0.00196)
    assert result["native_pair_rest_offset_m"] == 0.002
    assert result["checked_samples"] == 1


def test_threshold_is_actual_pair_offset_not_fixed_two_millimetres() -> None:
    assert _screen([0.01696] * 3, target_rest=0.0005).inspect(np.eye(4))["valid"]


def test_contact_envelope_revision_rejects_body_before_bilateral_pad_rest() -> None:
    assert _screen([0.02] * 3).inspect(np.eye(4))["valid"]
    result = _screen([0.02] * 3, revision=2).inspect(np.eye(4))
    assert result["valid"] is False
    assert result["reason"] == "nonpad_contact_envelope_before_bilateral_pad_rest"
    assert result["checked_samples"] == 1
    assert result["native_pair_contact_offset_m"] == pytest.approx(0.009)
    assert result["minimum_distance_m"] == pytest.approx(0.005)


def test_contact_halo_at_first_bilateral_rest_is_not_actual_body_contact() -> None:
    result = _screen([0.1, 0.02, 0.014], revision=2).inspect(np.eye(4))
    assert result["valid"] is True
    assert result["checked_samples"] == 2
    assert result["potential_nonpad_halo_at_pad_rest_endpoint"] is True


def test_actual_body_rest_still_rejects_at_bilateral_endpoint() -> None:
    result = _screen([0.1, 0.016, 0.014], revision=2).inspect(np.eye(4))
    assert result["valid"] is False
    assert result["reason"] == "nonpad_and_pad_order_unresolved_at_sample"


def test_revision_four_records_only_first_halo_without_rejecting_source() -> None:
    screen = _screen([0.02, 0.018, 0.014], revision=4)
    result = screen.inspect(np.eye(4))
    assert result["valid"] is True
    assert result["reason"] == "nominal_bilateral_pad_rest_first"
    assert result["checked_samples"] == 2
    assert result["first_nonpad_contact_halo_link"] == "knuckle"
    assert result["first_nonpad_contact_halo_minimum_distance_m"] == pytest.approx(
        0.005
    )
    assert result["first_nonpad_contact_halo_sample_index"] == 0
    assert result["first_nonpad_contact_halo_native_pair_contact_offset_m"] == (
        pytest.approx(0.009)
    )
    assert screen.report()["nonpad_contact_halo_is_veto"] is False
    result["first_nonpad_contact_halo_link"] = "changed"
    assert screen.inspect(np.eye(4))["first_nonpad_contact_halo_link"] == "knuckle"


@pytest.mark.parametrize(
    ("body_z", "reason", "step"),
    [
        ([0.016] * 3, "nonpad_rest_before_bilateral_pad", 0),
        ([0.1, 0.016, 0.014], "nonpad_and_pad_order_unresolved_at_sample", 1),
    ],
)
def test_revision_four_keeps_native_rest_order_veto(body_z, reason, step) -> None:
    screen = _screen(body_z, revision=4)
    screen.opening_screen = Mock()
    assert screen.mask(torch.eye(4)[None, None], torch.eye(4)[None]).tolist() == [
        [False]
    ]
    result = screen.inspect(np.eye(4))
    assert result["reason"] == reason
    assert result["first_nonpad_contact_halo_sample_index"] == step
    screen.opening_screen.inspect.assert_not_called()


@pytest.mark.parametrize("height, valid", [(0.025, True), (0.012, False)])
def test_revision_four_retains_real_opening_screen_after_source_halo(height, valid):
    from embodichain.gen_sim.task_engine._task_program.release_contact import (
        OpeningContactScreen,
    )

    screen = _screen([0.02] * 3, revision=4)
    target = np.eye(4)
    target[2, 3] = height
    screen.opening_screen = OpeningContactScreen(
        screen,
        (trimesh.creation.box((0.1, 0.1, 0.02)),),
        np.eye(4),
        target,
        0.001,
        0.003,
        identity={"reference_entity_id": "support"},
    )
    poses = torch.eye(4, dtype=torch.float64)[None, None]
    before, rng = poses.clone(), torch.random.get_rng_state().clone()
    assert screen.mask(poses, torch.eye(4)[None]).tolist() == [[valid]]
    assert torch.equal(poses, before)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert screen.inspect(np.eye(4))["valid"] is True
    opening = screen.report()["opening"]
    assert opening["physical_evidence"] is False
    assert opening["accepted_queries"] == int(valid)
    assert opening["rejected_queries"] == int(not valid)
    if not valid:
        assert opening["reasons"][0]["reason"] == "hand_support_rest_envelope_overlap"


@pytest.mark.parametrize("revision, valid", [(1, True), (2, False), (3, False)])
def test_legacy_source_halo_decisions_and_report_keys_stay_unchanged(revision, valid):
    screen = _screen([0.02] * 3, revision=revision)
    result = screen.inspect(np.eye(4))
    assert result["valid"] is valid
    assert not any(key.startswith("first_nonpad_contact_halo") for key in result)
    assert "nonpad_contact_halo_is_veto" not in screen.report()
    if revision == 3:
        assert screen.report()["opening"] is None
    else:
        assert "opening" not in screen.report()


@pytest.mark.parametrize("offset", [None, -0.001, 0.0005, float("nan")])
@pytest.mark.parametrize("revision", [2, 3, 4])
def test_contact_revisions_require_native_contact_offsets(
    offset: float | None, revision: int
) -> None:
    legacy = _screen([0.1] * 3)
    with pytest.raises(ValueError, match="contact offsets"):
        ClosingContactScreen(
            legacy.parts,
            (trimesh.creation.box((0.02, 0.02, 0.02)),),
            0.001,
            revision=revision,
            target_contact_offset=offset,
        )


def test_rigid_world_transform_and_pose_batch_do_not_change_result() -> None:
    screen = _screen([0.1, 0.1, 0.014])
    world = torch.eye(4, dtype=torch.float64)
    world[:3, :3] = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    world[:3, 3] = torch.tensor([2.0, -3.0, 0.7])
    tcp = world[None, None].expand(1, 2, -1, -1).clone()
    assert screen.mask(tcp, world[None]).tolist() == [[True, True]]
    assert screen.report()["unique_tcp_queries"] == 1


def test_cache_is_call_local_and_does_not_modify_inputs_or_rng() -> None:
    screen = _screen([0.1, 0.1, 0.014])
    pose = np.eye(4)
    before = pose.copy()
    rng = torch.random.get_rng_state().clone()
    first = screen.inspect(pose)
    first["valid"] = False
    assert screen.inspect(pose)["valid"] is True
    assert np.array_equal(pose, before)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert _screen([0.1, 0.1, 0.014]).report()["unique_tcp_queries"] == 0


def test_declared_upright_quality_rejects_up_down_closing_not_only_approach():
    from scipy.spatial.transform import Rotation

    screen = _screen([0.1] * 3)
    screen.closure_quality = SimpleNamespace(
        max_tilt_degrees=30.0, up_world=(0.0, 0.0, 1.0)
    )
    pose = torch.eye(4, dtype=torch.float64)
    pose[:3, :3] = torch.from_numpy(
        Rotation.from_euler("y", 59.5, degrees=True).as_matrix()
    )
    # Identical object/TCP rotation preserves contact geometry, not world quality.
    assert screen.mask(pose[None, None], pose[None]).tolist() == [[False]]


@pytest.mark.parametrize("value", [-0.001, float("nan"), float("inf")])
def test_invalid_native_offsets_are_not_defaulted(value: float) -> None:
    with pytest.raises(ValueError, match="offsets"):
        _screen([0.1, 0.1, 0.014], target_rest=value)


@pytest.mark.parametrize("revision", [1, 2, 3, 4])
def test_missing_bilateral_pad_contact_is_not_accepted(revision) -> None:
    screen = _screen([0.1] * 3, revision=revision)
    distant = np.eye(4)
    distant[2, 3] = 1.0
    assert screen.inspect(distant)["reason"] == "bilateral_pad_rest_not_reached"


def test_nonfinite_or_multi_environment_query_is_rejected() -> None:
    screen = _screen([0.1] * 3)
    with pytest.raises(ValueError, match="B1"):
        screen.mask(torch.eye(4).repeat(2, 1, 1), torch.eye(4).repeat(2, 1, 1))
    pose = np.eye(4)
    pose[0, 0] = np.nan
    with pytest.raises(ValueError, match="TCP pose"):
        screen.inspect(pose)


def test_closing_mask_preserves_original_ik_population_and_rng(monkeypatch) -> None:
    from embodichain.gen_sim.task_engine._task_program.actions import (
        GenSimPickUp,
        _PickSelection,
        _PICK_SELECTION,
    )
    from embodichain.lab.sim.atomic_actions.primitives.pick_up import (
        PickUp,
        PickUpOptions,
    )

    action = GenSimPickUp()
    action._planning_services = SimpleNamespace(
        device=torch.device("cpu"), robot=SimpleNamespace(device=torch.device("cpu"))
    )
    grasps = torch.eye(4).repeat(1, 2, 1, 1)
    grasps[:, :, :3, :3] = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
    grasps[0, 1, 0, 3] = 1.0
    original = grasps.clone()
    queried = []

    def mask(poses, reference):
        queried.append(poses.clone())
        assert reference.shape == (1, 4, 4)
        return poses[..., 0, 3] > 0.5

    def shared(self, poses, seed, reference, target, options, direction):
        assert torch.equal(poses, original)
        torch.rand(3)
        variants = torch.stack((poses, poses), dim=2)
        return poses, self._approach_alignment_mask(variants, options, direction).any(
            -1
        )

    monkeypatch.setattr(PickUp, "_select_feasible_grasp_variants", shared)
    before = torch.random.get_rng_state().clone()
    torch.rand(3)
    expected = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(before)
    token = _PICK_SELECTION.set(
        _PickSelection(closing_geometry=SimpleNamespace(mask=mask))
    )
    try:
        selected, valid = action._select_feasible_grasp_variants(
            grasps,
            torch.zeros(1, 1),
            torch.eye(4)[None],
            object(),
            PickUpOptions(approach_alignment_max_angle=0.1),
            torch.tensor([0.0, 0.0, -1.0]),
        )
    finally:
        _PICK_SELECTION.reset(token)
    assert torch.equal(selected, original)
    assert valid.tolist() == [[False, True]]
    assert len(queried) == 1
    assert torch.equal(torch.random.get_rng_state(), expected)
    assert torch.equal(grasps, original)


@pytest.mark.parametrize("changed_hand", [False, True])
@pytest.mark.parametrize("bad_quality", [False, True])
@pytest.mark.parametrize("native_scene", [False, True])
def test_emitted_close_checks_actual_hand_commands_before_nominal_geometry(
    monkeypatch, changed_hand, bad_quality, native_scene
) -> None:
    from unittest.mock import Mock
    from embodichain.gen_sim.task_engine._task_program.actions import GenSimPickUp
    from embodichain.lab.sim.atomic_actions import goals
    from embodichain.lab.sim.atomic_actions.primitives.pick_up import PickUpOptions

    options = PickUpOptions(hand_interp_steps=3)
    positions = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.35, -0.35], [0.0, 0.7, -0.7]]])
    if changed_hand:
        positions[0, -1, 2] = -0.2
    arm = SimpleNamespace(control_part="arm", joint_ids=(0,))
    hand = SimpleNamespace(control_part="hand", joint_ids=(1, 2))
    endpoint = SimpleNamespace(
        require_target=lambda _: hand,
        joint_positions=lambda command, **kwargs: (
            torch.tensor([[0.0, 0.0]])
            if command == "open"
            else torch.tensor([[0.7, -0.7]])
        ),
    )
    request = SimpleNamespace(
        invocation_id="pick",
        goal=SimpleNamespace(object_pose=None, semantics=object()),
        skill_options=options,
        binding=SimpleNamespace(
            endpoint=lambda _, name: (
                endpoint
                if name == "grasp"
                else SimpleNamespace(require_target=lambda _: arm)
            )
        ),
    )
    plan = SimpleNamespace(
        segments=(SimpleNamespace(name="close", start=0, stop=3),),
        joint_trajectory=SimpleNamespace(positions=positions),
    )
    fk = Mock(return_value=torch.eye(4).repeat(1, 3, 1, 1))
    action = GenSimPickUp()
    action.pick_purposes = {"pick": "ordinary"}
    scene = Mock(return_value={"valid": False, "kind": "native_rigid_pair_rest"})
    action._inspect_closing_scene = scene
    action._planning_services = SimpleNamespace(
        robot=SimpleNamespace(compute_batch_fk=fk)
    )
    screen = SimpleNamespace(
        revision=8 if native_scene else 1,
        mask=Mock(return_value=torch.ones(1, 3, dtype=torch.bool)),
        report=lambda: {"revision": 1},
    )
    object_pose = torch.eye(4)[None]
    if bad_quality:
        from scipy.spatial.transform import Rotation
        from embodichain.gen_sim.task_engine._task_program.pick_closure_quality import (
            UprightClosurePolicy,
        )

        native = _screen([0.1] * 3)
        native.closure_quality = UprightClosurePolicy(30)
        object_pose[:, :3, :3] = torch.tensor(
            Rotation.from_euler("y", 59.5, degrees=True).as_matrix(),
            dtype=torch.float32,
        )
        fk.return_value = object_pose[:, None].expand(1, 3, 4, 4).clone()
        screen.mask.side_effect = native.mask
        screen.report = native.report
    monkeypatch.setattr(
        goals, "_resolve_object_pose", lambda *args, **kwargs: object_pose
    )
    result = action.inspect_closing_plan(
        request, SimpleNamespace(batch_size=1), plan, screen=screen
    )
    assert result["valid"] is (
        not changed_hand and not bad_quality and not native_scene
    )
    if native_scene and not changed_hand and not bad_quality:
        scene.assert_called_once()
        assert result["closing_scene"] is scene.return_value
        assert scene.call_args.args[3] is screen
    else:
        scene.assert_not_called()
    if changed_hand:
        assert result["reason"] == "emitted_hand_closing_commands_changed"
        screen.mask.assert_not_called()
        fk.assert_not_called()
    else:
        assert torch.equal(fk.call_args.kwargs["qpos"], positions[:, :, :1])


def test_native_builder_rejects_missing_mimic_command_coverage(
    monkeypatch, tmp_path
) -> None:
    from embodichain.gen_sim.task_engine._task_program import pick_contact

    path = tmp_path / "hand.urdf"
    path.write_text(
        '<robot name="hand"><joint name="leader" type="revolute"><parent link="base"/><child link="finger"/></joint><joint name="follower" type="revolute"><parent link="base"/><child link="pad"/></joint></robot>'
    )
    native_robot = SimpleNamespace(
        get_name=lambda: "robot", render_articulation=object(), arena_name="episode"
    )
    native_object = SimpleNamespace(
        get_name=lambda: "physical_payload",
        arena_name="episode",
        native=lambda: SimpleNamespace(get_physical_body=lambda: object()),
    )
    handles = {"r": native_robot, "o": native_object}
    simulation = SimpleNamespace(
        num_envs=1,
        is_default_backend=True,
        spawn_result=SimpleNamespace(get_paths=lambda: tuple(handles), get=handles.get),
    )
    endpoint = SimpleNamespace(
        task_state_key="resource",
        require_target=lambda _: SimpleNamespace(joint_ids=(0,), control_part="hand"),
    )
    request = SimpleNamespace(
        goal=SimpleNamespace(semantics=SimpleNamespace(entity_id="payload")),
        binding=SimpleNamespace(endpoint=lambda *args: endpoint),
    )
    robot = SimpleNamespace(
        uid="robot", joint_names=("leader", "follower"), cfg=SimpleNamespace(fpath=path)
    )
    bindings = SimpleNamespace(
        rigid_objects=(
            SimpleNamespace(entity_id="payload", simulation_uid="physical_payload"),
        )
    )
    monkeypatch.setattr(
        pick_contact, "_native_meshes", lambda _: (trimesh.creation.box(),)
    )
    with pytest.raises(ValueError, match="every moving hand joint"):
        pick_contact.build_closing_contact_screen(robot, simulation, bindings, request)


@pytest.mark.parametrize("revision", [1, 2, 3, 4])
def test_native_builder_keeps_float32_relative_poses_exactly_homogeneous(
    monkeypatch, tmp_path, revision
) -> None:
    from embodichain.gen_sim.task_engine._task_program import pick_contact
    from embodichain.utils.math import pose_inv

    path = tmp_path / "hand.urdf"
    path.write_text(
        '<robot name="hand"><joint name="leader" type="revolute"><parent link="base"/><child link="negative_finger_pad"/></joint><joint name="follower" type="revolute"><parent link="base"/><child link="positive_finger_pad"/></joint></robot>'
    )
    tool = torch.eye(4)
    angle = torch.tensor(0.713)
    tool[:3, :3] = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    tool[:3, 3] = torch.tensor([0.271, -0.139, 0.813])
    links = ("base", "negative_finger_pad", "positive_finger_pad")

    def fk(*, qpos, link_names, qpos_joint_names):
        assert tuple(qpos_joint_names) == ("leader", "follower")
        result = tool.repeat(len(qpos), len(link_names), 1, 1)
        for index, name in enumerate(link_names):
            if name != "base":
                local = torch.eye(4).repeat(len(qpos), 1, 1)
                local[:, 0, 3] = (-1 if name.startswith("negative") else 1) * 0.02
                result[:, index] = tool @ local
        return result

    contact_reads = []

    def read_contact():
        contact_reads.append(True)
        return 0.006

    body = SimpleNamespace(
        get_rest_offset=lambda: 0.001, get_contact_offset=read_contact
    )
    native_robot = SimpleNamespace(
        get_name=lambda: "robot",
        arena_name="episode",
        path="episode/robot",
        render_articulation=SimpleNamespace(get_physical_body=lambda _: body),
        get_link_names=lambda: links,
        get_link_pose=lambda name: fk(
            qpos=torch.zeros(1, 2),
            link_names=[name],
            qpos_joint_names=("leader", "follower"),
        )[0, 0].numpy(),
    )
    native_object = SimpleNamespace(
        get_name=lambda: "physical_payload",
        arena_name="episode",
        path="episode/physical_payload",
        native=lambda: SimpleNamespace(get_physical_body=lambda: body),
    )
    handles = {"r": native_robot, "o": native_object}
    simulation = SimpleNamespace(
        num_envs=1,
        is_default_backend=True,
        spawn_result=SimpleNamespace(get_paths=lambda: tuple(handles), get=handles.get),
    )
    endpoint = SimpleNamespace(
        task_state_key="resource",
        require_target=lambda _: SimpleNamespace(joint_ids=(0, 1), control_part="hand"),
        joint_positions=lambda command, **kwargs: (
            torch.zeros(1, 2) if command == "open" else torch.tensor([[0.7, -0.7]])
        ),
    )
    request = SimpleNamespace(
        goal=SimpleNamespace(semantics=SimpleNamespace(entity_id="payload")),
        binding=SimpleNamespace(endpoint=lambda *args: endpoint),
    )
    robot = SimpleNamespace(
        uid="robot",
        joint_names=("leader", "follower"),
        cfg=SimpleNamespace(
            fpath=path,
            solver_cfg={
                "hand": SimpleNamespace(end_link_name="base", tcp=torch.eye(4))
            },
        ),
        get_qpos=lambda: torch.zeros(1, 2),
        get_local_pose=lambda **kwargs: torch.eye(4)[None],
        compute_fk=fk,
    )
    bindings = SimpleNamespace(
        rigid_objects=(
            SimpleNamespace(entity_id="payload", simulation_uid="physical_payload"),
        )
    )
    monkeypatch.setattr(
        pick_contact, "_native_meshes", lambda _: (trimesh.creation.box(),)
    )
    from embodichain.gen_sim.task_engine._task_program import release_contact

    opening = Mock()
    build_opening = Mock(return_value=opening)
    monkeypatch.setattr(release_contact, "build_opening_screen", build_opening)
    context, preview = object(), object()
    screen = pick_contact.build_closing_contact_screen(
        robot,
        simulation,
        bindings,
        request,
        revision=revision,
        context=context,
        preview=preview,
        purpose="ordinary",
    )
    assert screen.revision == revision
    assert len(contact_reads) == (len(screen.parts) + 1 if revision >= 2 else 0)
    if revision in (3, 4):
        assert screen.opening_screen is opening
        build_opening.assert_called_once_with(
            screen,
            robot,
            simulation,
            bindings,
            request,
            context,
            preview,
            purpose="ordinary",
        )
    else:
        assert screen.opening_screen is None
        build_opening.assert_not_called()
    assert screen.identity["native_object_path"] == "episode/physical_payload"
    for part in screen.parts:
        assert np.array_equal(
            part.tcp_poses[:, 3], np.tile([0.0, 0.0, 0.0, 1.0], (65, 1))
        )
    expected = (
        pose_inv(tool)
        @ fk(
            qpos=torch.zeros(1, 2),
            link_names=["positive_finger_pad"],
            qpos_joint_names=("leader", "follower"),
        )[0, 0]
    )
    assert np.allclose(
        next(p for p in screen.parts if p.name == "positive_finger_pad").tcp_poses[0],
        expected.numpy(),
    )


@pytest.mark.parametrize("revision", [None, 1, 2, 3, 4])
def test_engine_factory_passes_live_opening_context_only_for_revisions_three_four(
    monkeypatch, revision
) -> None:
    from embodichain.gen_sim.task_engine._task_program import (
        assembly,
        pick_contact,
        place_recovery,
    )
    from embodichain.gen_sim.task_engine._task_program.actions import GenSimPickUp
    from embodichain.gen_sim.task_engine._task_program.state_handoff import (
        StagingContract,
    )
    from embodichain.lab.task_program.semantics import RobotSkillProfile

    profile = Mock(spec=RobotSkillProfile)
    profile.profile_id = "test.robot"
    profile.action_control_profiles.return_value = {}
    robot, simulation, binding = object(), object(), object()
    factory = object.__new__(assembly._TaskFactory)
    factory._robot, factory._simulation = robot, simulation
    factory._robot_profile_binding = profile
    factory._registration = SimpleNamespace(
        scene_binding=binding, validate_engine=Mock()
    )
    factory._create_motion_generator = Mock(return_value=SimpleNamespace(robot=robot))
    for name in (
        "_drawers",
        "_motion_samples",
        "_cartesian_calls",
        "_articulation_calls",
        "_pour_return_calls",
        "_press_routes",
        "_twist_routes",
        "_clearance_routes",
        "_alignment_continuation_routes",
        "_pick_continuations",
        "_pour_continuations",
    ):
        setattr(factory, name, ())
    factory._grasp_pose_generators = factory._pour_receivers = {}
    factory._adaptive_pick = factory._handover_scene_clearance = False
    factory._coordinated_motion = False
    factory._staging_contract = StagingContract.empty()
    factory._pick_contact_revision = revision
    factory._pick_purposes = (("pick", "ordinary"),)
    factory._upright_container_picks = {}
    engine, screen = Mock(), object()
    monkeypatch.setattr(assembly, "GenSimActionEngine", Mock(return_value=engine))
    monkeypatch.setattr(place_recovery, "scene_vertices", lambda *args: ())
    build = Mock(return_value=screen)
    monkeypatch.setattr(pick_contact, "build_closing_contact_screen", build)
    assert factory.create_atomic_action_engine(profile) is engine
    pick = next(
        call.args[0]
        for call in engine.register.call_args_list
        if isinstance(call.args[0], GenSimPickUp)
    )
    if revision is None:
        assert pick.closing_geometry_factory is None
        build.assert_not_called()
        return
    request = SimpleNamespace(invocation_id="pick")
    context, preview = object(), object()
    pick.continuation_previews = {"pick": preview}
    assert pick.closing_geometry_factory(request, context) is screen
    kwargs = {"revision": revision}
    if revision in (3, 4):
        kwargs.update(context=context, preview=preview, purpose="ordinary")
    build.assert_called_once_with(robot, simulation, binding, request, **kwargs)
