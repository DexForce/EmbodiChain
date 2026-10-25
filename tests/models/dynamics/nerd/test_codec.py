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
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.autograd import gradcheck
from embodichain_models.dynamics.nerd.codec import (
    NeRDStateCodec,
    quat_rotate,
    plane_contact_features,
    append_history,
    reset_history,
)


def ant_state(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    q = torch.randn(batch, 15, dtype=torch.double) * 0.1
    rotation = q[:, 3:7] + q.new_tensor((0.1, 0.2, 0.3, 1.0))
    q[:, 3:7] = rotation / rotation.norm(dim=-1, keepdim=True)
    return q, torch.randn(batch, 14, dtype=torch.double)


def test_ant_real_official_frame_features_match_reference_rounding() -> None:
    """Frozen actual Warp1.5.1/Torch2.2.2 windows catch FP32 formula drift."""
    with np.load(Path(__file__).parent / "data/ant-official-frame.npz") as data:
        values = {key: torch.from_numpy(data[key].copy()) for key in data.files}
    contacts = {
        key: value for key, value in values.items() if key.startswith("contact_")
    }
    thickness = contacts.pop("contact_thicknesses")
    contacts["contact_thicknesses_0"] = thickness
    contacts["contact_thicknesses_1"] = torch.zeros_like(thickness)
    codec = NeRDStateCodec("ant")
    actual = codec.encode(
        values["q"], values["qd"], values["joint_f"], contacts, values["gravity_dir"]
    )
    for name, value in actual.items():
        torch.testing.assert_close(value, values["expected_" + name], atol=0, rtol=0)
    target = codec.targets(
        values["q"], values["qd"], values["next_q"], values["next_qd"]
    )
    torch.testing.assert_close(target, values["expected_target"], atol=1e-5, rtol=1e-4)


def test_com_reference_shift_and_inverse() -> None:
    codec = NeRDStateCodec("ant", base_com=(0.1, -0.2, 0.3))
    q, qd = ant_state()
    legacy = codec.native_to_legacy(q, qd)
    center = q[:, :3] + quat_rotate(q[:, 3:7], q.new_tensor(codec.base_com))
    # A spatial twist evaluated at the body's COM reproduces its linear velocity.
    torch.testing.assert_close(
        legacy[:, 3:6] + torch.linalg.cross(legacy[:, :3], center), qd[:, :3]
    )
    torch.testing.assert_close(codec.legacy_to_native(q, legacy), qd)


def test_original_velocity_override_retains_values_and_input_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Imported twists need not pass through the lossy COM representation."""
    codec = NeRDStateCodec("ant")
    q = torch.zeros(2, 3, 15, dtype=torch.double)
    q[..., 6] = 1.0
    qd = torch.zeros(2, 3, 14, dtype=torch.double, requires_grad=True)
    original = torch.arange(2 * 6 * 14, dtype=torch.double).reshape(2, 6, 14) / 100
    original = original[:, ::2].requires_grad_()
    assert not original.is_contiguous()
    following = (original.detach() + 0.25).requires_grad_()
    before = original.detach().clone()
    contacts = {
        "contact_depths": torch.zeros(2, 3, 2, dtype=torch.double),
        "contact_normals": torch.ones(2, 3, 6, dtype=torch.double),
        "contact_points_1": torch.ones(2, 3, 6, dtype=torch.double),
        "contact_masks": torch.tensor([True, False]).expand(2, 3, 2),
    }

    def reject_native_conversion(*args: object) -> None:
        raise AssertionError("Original velocity must bypass COM conversion")

    monkeypatch.setattr(codec, "native_to_legacy", reject_native_conversion)
    inputs = codec.encode(
        q,
        qd,
        torch.zeros(2, 3, 8, dtype=torch.double),
        contacts,
        legacy_velocity=original,
    )
    target = codec.targets(
        q,
        qd,
        q,
        qd,
        current_embedding=inputs["states_embedding"],
        legacy_velocity=original,
        next_legacy_velocity=following,
    )
    torch.testing.assert_close(
        inputs["states_embedding"][..., 15:], original, atol=0, rtol=0
    )
    torch.testing.assert_close(target[..., 15:], following - original, atol=0, rtol=0)
    assert torch.count_nonzero(inputs["contact_points_1"][..., 3:]) == 0
    torch.testing.assert_close(original, before, atol=0, rtol=0)
    (3 * inputs["states_embedding"][..., 15:].sum() + target[..., 15:].sum()).backward()
    torch.testing.assert_close(original.grad, torch.full_like(original, 2))
    torch.testing.assert_close(following.grad, torch.ones_like(following))
    assert qd.grad is None


@pytest.mark.parametrize("which", ["encode", "current_target", "next_target"])
@pytest.mark.parametrize("invalid", ["shape", "dtype", "device"])
def test_legacy_velocity_override_rejects_mismatched_metadata(
    which: str, invalid: str
) -> None:
    codec = NeRDStateCodec("cartpole")
    q = torch.zeros(2, 3, 2)
    qd = torch.zeros_like(q)
    value = {
        "shape": torch.zeros(2, 3, 1),
        "dtype": torch.zeros(2, 3, 2, dtype=torch.float64),
        "device": torch.empty(2, 3, 2, device="meta"),
    }[invalid]
    with pytest.raises(ValueError, match="Legacy velocity override"):
        if which == "encode":
            codec.encode(q, qd, qd, legacy_velocity=value)
        else:
            kwargs = {
                (
                    "legacy_velocity"
                    if which == "current_target"
                    else "next_legacy_velocity"
                ): value
            }
            codec.targets(q, qd, q, qd, **kwargs)


@pytest.mark.parametrize("robot", ["ant", "cartpole"])
def test_explicit_absent_legacy_velocity_preserves_native_contract(robot: str) -> None:
    codec = NeRDStateCodec(robot)
    q = torch.zeros(2, codec.q_dim)
    qd = torch.ones(2, codec.qd_dim)
    if robot == "ant":
        q[:, 6] = 1.0
    contacts = {
        "contact_depths": torch.zeros(2, 1),
        "contact_normals": torch.zeros(2, 3),
        "contact_points_1": torch.zeros(2, 3),
    }
    default = codec.encode(q, qd, torch.zeros_like(qd), contacts)
    optional = codec.encode(q, qd, torch.zeros_like(qd), contacts, legacy_velocity=None)
    for name in default:
        torch.testing.assert_close(default[name], optional[name], atol=0, rtol=0)
    torch.testing.assert_close(
        codec.targets(q, qd, q, qd + 0.1),
        codec.targets(
            q, qd, q, qd + 0.1, legacy_velocity=None, next_legacy_velocity=None
        ),
        atol=0,
        rtol=0,
    )


def test_ant_targets_decode_in_frame_at_t() -> None:
    codec = NeRDStateCodec("ant", base_com=(0.1, -0.2, 0.3))
    q, qd = ant_state()
    next_q, next_qd = ant_state()
    next_q = next_q.clone()
    next_q[:, :3] += q.new_tensor((0.03, -0.07, 0.02))
    next_q[:, 3:7] = (
        q.new_tensor((0.2, -0.1, 0.3, 0.9)) / q.new_tensor((0.2, -0.1, 0.3, 0.9)).norm()
    )
    next_q[:, 7:] += 0.01
    next_qd = next_qd + 0.13
    prediction = codec.targets(q, qd, next_q, next_qd)
    actual_q, actual_qd = codec.decode(q, qd, prediction)
    torch.testing.assert_close(actual_q, next_q)
    torch.testing.assert_close(actual_qd, next_qd)
    # Rotation target is relative to orientation at t, not identity at t+1.
    assert prediction[:, 3:6].abs().sum() > 0.1


def test_ant_targets_reuse_the_current_encoded_frame() -> None:
    codec = NeRDStateCodec("ant", base_com=(0.1, -0.2, 0.3))
    q, qd = ant_state()
    next_q, next_qd = q.clone(), qd + 0.13
    next_q[:, :3] += 0.02
    contacts = {
        "contact_depths": q.new_full((len(q), 1), 1e4),
        "contact_normals": q.new_zeros(len(q), 3),
        "contact_points_1": q.new_zeros(len(q), 3),
    }
    inputs = codec.encode(q, qd, torch.zeros_like(qd), contacts)
    target = codec.targets(
        q, qd, next_q, next_qd, current_embedding=inputs["states_embedding"]
    )
    actual_q, actual_qd = codec.decode(
        q, qd, target, current_embedding=inputs["states_embedding"]
    )
    torch.testing.assert_close(actual_q, next_q)
    torch.testing.assert_close(actual_qd, next_qd)
    with pytest.raises(ValueError, match="embedding shape"):
        codec.targets(q, qd, next_q, next_qd, current_embedding=q)


def test_cartpole_large_angle_wrapping_matches_official_preprocessing() -> None:
    codec = NeRDStateCodec("cartpole")
    q = torch.tensor([[0.25, 123.4], [0.5, -321.7]])
    next_q = q + torch.tensor([[0.01, 0.007], [-0.02, 0.009]])
    velocity = torch.zeros_like(q)

    def official_wrap(value):
        return value - torch.floor((value + torch.pi) / (2 * torch.pi)) * (2 * torch.pi)

    target = codec.targets(q, velocity, next_q, velocity)
    expected_angle = official_wrap(official_wrap(next_q[:, 1]) - official_wrap(q[:, 1]))
    torch.testing.assert_close(target[:, 1], expected_angle, atol=0, rtol=0)
    actual_q, _ = codec.decode(q, velocity, target)
    expected_decoded = official_wrap(official_wrap(q[:, 1]) + expected_angle)
    torch.testing.assert_close(actual_q[:, 1], expected_decoded, atol=0, rtol=0)


def test_cartpole_wrap_and_decode() -> None:
    codec = NeRDStateCodec()
    q = torch.tensor([[0.4, torch.pi - 0.01]])
    next_q = torch.tensor([[0.5, -torch.pi + 0.02]])
    qd = torch.ones_like(q)
    target = codec.targets(q, qd, next_q, qd * 2)
    torch.testing.assert_close(target[0, 1], torch.tensor(0.03), atol=1e-6, rtol=0)
    actual = codec.decode(q, qd, target)
    torch.testing.assert_close(actual[0], next_q)
    torch.testing.assert_close(actual[1], qd * 2)


def test_static_plane_mask_and_gradient() -> None:
    points = torch.tensor(
        [[[0.2, 0.1, 0.05], [0.4, -0.1, 0.5]]], dtype=torch.double, requires_grad=True
    )
    normal = points.new_tensor((0.0, 0.0, 1.0))
    result = plane_contact_features(points, points.new_tensor([0.02, 0.02]), normal)
    torch.testing.assert_close(
        result["contact_points_1"], points.new_tensor([[0.2, 0.1, 0.0, 0.4, -0.1, 0.0]])
    )
    assert result["contact_masks"].tolist() == [[True, False]]
    assert gradcheck(
        lambda x: tuple(
            plane_contact_features(x, x.new_tensor([0.02, 0.02]), normal)[k]
            for k in ("contact_points_1", "contact_depths")
        ),
        (points,),
    )


def test_ant_encode_mask_and_gradients() -> None:
    codec = NeRDStateCodec("ant", base_com=(0.1, -0.2, 0.3))
    q, qd = ant_state(1)
    effort = torch.randn(1, 8, dtype=torch.double, requires_grad=True)
    point = torch.tensor([[[0.2, 0.1, 0.05], [0.4, -0.1, 0.5]]], dtype=torch.double)
    contacts = plane_contact_features(
        point, point.new_tensor([0.02, 0.02]), point.new_tensor((0.0, 0.0, 1.0))
    )

    def function(
        q: torch.Tensor, qd: torch.Tensor, effort: torch.Tensor
    ) -> torch.Tensor:
        # Quaternion constraint stays on the manifold during finite differences.
        q = torch.cat(
            (q[:, :3], q[:, 3:7] / q[:, 3:7].norm(dim=-1, keepdim=True), q[:, 7:]), -1
        )
        return torch.cat(tuple(codec.encode(q, qd, effort, contacts).values()), -1)

    assert gradcheck(function, (q.requires_grad_(), qd.requires_grad_(), effort))
    output = codec.encode(q, qd, effort, contacts)
    assert torch.count_nonzero(output["contact_points_1"][:, 3:]) == 0


def test_decode_gradients_nonzero_com_and_orientation() -> None:
    codec = NeRDStateCodec("ant", base_com=(0.1, -0.2, 0.3))
    q, qd = ant_state(1)
    pred = torch.randn(1, 29, dtype=torch.double) * 0.01
    pred[:, 6] = 1.0

    def function(
        q: torch.Tensor, qd: torch.Tensor, pred: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = torch.cat(
            (q[:, :3], q[:, 3:7] / q[:, 3:7].norm(dim=-1, keepdim=True), q[:, 7:]), -1
        )
        return codec.decode(q, qd, pred)

    assert gradcheck(
        function, (q.requires_grad_(), qd.requires_grad_(), pred.requires_grad_())
    )


def test_history_partial_reset_retains_other_graphs() -> None:
    history = {"x": torch.zeros(2, 3, 1, dtype=torch.double)}
    lengths = torch.zeros(2, dtype=torch.long)
    tokens = [
        torch.tensor(
            [[float(i)], [float(i + 10)]], dtype=torch.double, requires_grad=True
        )
        for i in range(4)
    ]
    for token in tokens[:3]:
        history, lengths = append_history(history, lengths, {"x": token})
    saved = history["x"]
    history, lengths = reset_history(history, lengths, torch.tensor([True, False]))
    history, lengths = append_history(history, lengths, {"x": tokens[3]})
    assert lengths.tolist() == [1, 3]
    assert history["x"].squeeze(-1).tolist() == [[3.0, 0.0, 0.0], [11.0, 12.0, 13.0]]
    assert saved.squeeze(-1).tolist() == [[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]]
    history["x"].sum().backward()
    torch.testing.assert_close(
        tokens[1].grad, torch.tensor([[0.0], [1.0]], dtype=torch.double)
    )
