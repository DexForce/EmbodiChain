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

"""Explicit trusted reference qualification helper; never imported by production."""

from __future__ import annotations
import json
import hashlib
import platform
import subprocess
import sys
from pathlib import Path
import torch
from embodichain.learning.models.nerd import convert_legacy_checkpoint, load_model


def compare_codec(robot: str, destination: str) -> dict[str, float]:
    """Compare the actual upstream solver's pure encoding/target/decode methods.

    Construct only its math metadata: no reference simulation/model is initialized.
    Native qd is independently converted to the legacy world-origin twist at the
    comparison boundary, matching the upstream solver's expected input contract.
    """
    from newton import JointType
    from solvers.solver_neural import NeuralSolver
    from embodichain.compute.dynamics import NeRDStateCodec, quat_rotate

    is_ant = robot == "Ant"
    codec = NeRDStateCodec(
        robot.lower(), base_com=(0.1, -0.2, 0.3) if is_ant else (0.0, 0.0, 0.0)
    )
    reference = NeuralSolver.__new__(NeuralSolver)
    reference.torch_device = "cpu"
    reference.dof_q_per_env, reference.dof_qd_per_env = codec.q_dim, codec.qd_dim
    reference.state_dim = codec.q_dim + codec.qd_dim
    reference.prediction_dim = reference.state_dim
    reference.num_envs = 2
    reference.num_contacts_per_env = 25 if is_ant else 0
    reference.joint_types = (
        [JointType.FREE] + [JointType.REVOLUTE] * 8
        if is_ant
        else [JointType.PRISMATIC, JointType.REVOLUTE]
    )
    reference.num_joints_per_env = len(reference.joint_types)
    reference.joint_q_start = [0] + list(range(7, 15)) if is_ant else [0, 1]
    reference.joint_q_end = [7] + list(range(8, 16)) if is_ant else [1, 2]
    reference.is_continuous_dof = torch.zeros(reference.state_dim, dtype=torch.bool)
    if not is_ant:
        reference.is_continuous_dof[1] = True
    reference.states_frame = "body" if is_ant else "world"
    reference.anchor_frame_step = "every"
    reference.states_embedding_type = "identical"
    reference.orientation_prediction_parameterization = "quaternion"
    reference.prediction_type = "relative"
    result = {
        "max_codec_feature_error": 0.0,
        "max_codec_target_error": 0.0,
        "max_codec_decode_error": 0.0,
    }
    fixtures = {}
    for length in [1, 9, 10]:
        q = torch.randn(2, length, codec.q_dim) * 0.2
        qd = torch.randn(2, length, codec.qd_dim) * 0.2
        next_q = q + torch.randn_like(q) * 0.03
        next_qd = qd + torch.randn_like(qd) * 0.03
        if is_ant:
            q[..., 3:7] += torch.tensor([0.2, -0.1, 0.3, 1.0])
            q[..., 3:7] = q[..., 3:7] / q[..., 3:7].norm(dim=-1, keepdim=True)
            next_q[..., 3:7] = q[..., 3:7] + torch.randn_like(q[..., 3:7]) * 0.03
            next_q[..., 3:7] = next_q[..., 3:7] / next_q[..., 3:7].norm(
                dim=-1, keepdim=True
            )
        effort = torch.randn(2, length, codec.qd_dim)
        gravity = torch.tensor([0.0, 0.0, -1.0]).expand(2, length, 3)
        contacts = None
        if is_ant:
            depth = torch.randn(2, length, 25) * 0.1
            contacts = {
                "contact_depths": depth,
                "contact_normals": torch.tensor([0.0, 0.0, 1.0])
                .expand(2, length, 25, 3)
                .flatten(-2),
                "contact_points_1": torch.cat(
                    (torch.randn(2, length, 25, 2), torch.zeros(2, length, 25, 1)), -1
                ).flatten(-2),
                "contact_thicknesses_0": torch.full((2, length, 25), 0.05),
                "contact_thicknesses_1": torch.zeros(2, length, 25),
            }
            contacts["contact_masks"] = reference.get_contact_masks(
                depth,
                contacts["contact_thicknesses_0"],
                contacts["contact_thicknesses_1"],
            )

        def legacy(q: torch.Tensor, qd: torch.Tensor) -> torch.Tensor:
            if not is_ant:
                return torch.cat((q, qd), -1)
            # Definition v(x)=omega cross x + nu, evaluated at root COM.
            com = q[..., :3] + quat_rotate(q[..., 3:7], q.new_tensor(codec.base_com))
            omega = qd[..., 3:6]
            nu = qd[..., :3] - torch.linalg.cross(omega, com)
            return torch.cat((q, omega, nu, qd[..., 6:]), -1)

        state, next_state = legacy(q, qd), legacy(next_q, next_qd)
        root_q = (
            q[..., :7]
            if is_ant
            else torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).expand(2, length, 7)
        )
        raw = {
            "root_body_q": root_q.clone(),
            "states": state.clone(),
            "next_states": next_state.clone(),
            "joint_f": effort[..., -8:].clone() if is_ant else effort.clone(),
            "gravity_dir": gravity.clone(),
        }
        if contacts:
            raw.update({k: v.clone() for k, v in contacts.items()})
        processed = reference.process_neural_model_inputs(raw)
        encoded = codec.encode(q, qd, effort, contacts, gravity)
        for key, value in encoded.items():
            torch.testing.assert_close(value, processed[key], atol=1e-5, rtol=1e-4)
            result["max_codec_feature_error"] = max(
                result["max_codec_feature_error"],
                (value - processed[key]).abs().max().item(),
            )
        target = codec.targets(q, qd, next_q, next_qd)
        expected = reference.convert_next_states_to_prediction(
            processed["states"], processed["next_states"]
        )
        torch.testing.assert_close(target, expected, atol=1e-5, rtol=1e-4)
        result["max_codec_target_error"] = max(
            result["max_codec_target_error"], (target - expected).abs().max().item()
        )
        # Arbitrary predictions also exercise decode independently of target inverse.
        prediction = torch.randn_like(target) * 0.03
        if is_ant:
            prediction[..., 6] += 1.0
        actual_q, actual_qd = codec.decode(q, qd, prediction)
        predicted_body = reference.convert_prediction_to_next_states(
            processed["states"], prediction
        )
        # The upstream method converts one last step; flatten B*T into batch and T=1.
        world = reference.convert_states_back_to_world(
            root_q.reshape(-1, 1, 7), predicted_body.reshape(-1, reference.state_dim)
        ).reshape_as(state)
        reference.wrap2PI(world)
        actual = legacy(actual_q, actual_qd)
        torch.testing.assert_close(actual, world, atol=1e-5, rtol=1e-4)
        result["max_codec_decode_error"] = max(
            result["max_codec_decode_error"], (actual - world).abs().max().item()
        )
        fixtures[str(length)] = {
            "q": q,
            "qd": qd,
            "next_q": next_q,
            "next_qd": next_qd,
            "effort": effort,
            "contacts": contacts,
            "expected_features": {k: processed[k] for k in encoded},
            "expected_target": expected,
            "prediction": prediction,
            "expected_decoded_legacy": world,
        }
    torch.save(fixtures, Path(destination) / "codec-fixtures.pt")
    return result


def main() -> None:
    root, robot, destination = sys.argv[1:]
    sys.path.insert(0, root)
    torch.set_num_threads(2)
    path = (
        Path(root)
        / "pretrained_models"
        / "NeRD_models"
        / robot
        / "model"
        / "nn"
        / "model.pt"
    )
    convert_legacy_checkpoint(
        path,
        destination,
        {"codec": {"robot": robot.lower()}, "dt": 1 / 60},
        trust_pickle=True,
    )
    actual, metadata = load_model(destination)
    official = torch.load(path, map_location="cpu", weights_only=False)[0].eval()
    torch.manual_seed(7)
    result = {
        "robot": robot,
        "checkpoint_sha256": metadata["legacy_checkpoint_sha256"],
        "max_forward_error": 0.0,
        "max_input_gradient_error": 0.0,
        "max_parameter_gradient_error": 0.0,
        "max_adam_parameter_error": 0.0,
    }

    def reference_inputs(values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"joint_acts" if k == "joint_f" else k: v for k, v in values.items()}

    for length in [1, 9, 10]:
        features = {
            k: torch.randn(2, length, d, requires_grad=True)
            for k, d in actual.cfg.input_dims.items()
        }
        reference = {
            k: v.detach().clone().requires_grad_() for k, v in features.items()
        }
        predicted = actual(features)
        expected = official(reference_inputs(reference))
        torch.testing.assert_close(predicted, expected, atol=1e-5, rtol=1e-4)
        result["max_forward_error"] = max(
            result["max_forward_error"], (predicted - expected).abs().max().item()
        )
        predicted.square().mean().backward()
        expected.square().mean().backward()
        for k in features:
            torch.testing.assert_close(
                features[k].grad, reference[k].grad, atol=1e-5, rtol=1e-4
            )
            result["max_input_gradient_error"] = max(
                result["max_input_gradient_error"],
                (features[k].grad - reference[k].grad).abs().max().item(),
            )
        for (name, p), (ref_name, rp) in zip(
            actual.named_parameters(), official.named_parameters()
        ):
            assert name == ref_name
            torch.testing.assert_close(p.grad, rp.grad, atol=1e-5, rtol=1e-4)
            result["max_parameter_gradient_error"] = max(
                result["max_parameter_gradient_error"],
                (p.grad - rp.grad).abs().max().item(),
            )
        actual.zero_grad()
        official.zero_grad()
    # Deterministic three-step normalized MSE/Adam equivalence with shared stats.
    opt = torch.optim.Adam(actual.parameters(), lr=1e-3)
    ref_opt = torch.optim.Adam(official.parameters(), lr=1e-3)
    for _ in range(3):
        features = {k: torch.randn(2, 10, d) for k, d in actual.cfg.input_dims.items()}
        target = torch.randn(2, 10, actual.cfg.output_dim)
        out = actual(features)
        ref = official(reference_inputs({k: v.clone() for k, v in features.items()}))
        # Same variance normalizer in both runs, isolating trainer update math.
        loss = ((out - target).square() / (actual.output_rms.var + 1e-5)).mean()
        ref_loss = ((ref - target).square() / (official.output_rms.var + 1e-5)).mean()
        torch.testing.assert_close(loss, ref_loss, atol=1e-5, rtol=1e-4)
        opt.zero_grad()
        ref_opt.zero_grad()
        loss.backward()
        ref_loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(actual.parameters(), 1.0)
        ref_norm = torch.nn.utils.clip_grad_norm_(official.parameters(), 1.0)
        torch.testing.assert_close(norm, ref_norm, atol=1e-5, rtol=1e-4)
        opt.step()
        ref_opt.step()
        for (name, p), (ref_name, rp) in zip(
            actual.named_parameters(), official.named_parameters()
        ):
            assert name == ref_name
            torch.testing.assert_close(p, rp, atol=1e-5, rtol=1e-4)
            result["max_adam_parameter_error"] = max(
                result["max_adam_parameter_error"], (p - rp).abs().max().item()
            )
    result.update(compare_codec(robot, destination))
    result["reference_commit"] = subprocess.check_output(
        ["git", "-C", root, "rev-parse", "HEAD"], text=True
    ).strip()
    result["precision"] = "float32"
    result["device"] = "cpu"
    result["torch_version"] = str(torch.__version__)
    result["python_version"] = platform.python_version()
    result["history_lengths"] = [1, 9, 10]
    result["adam_updates"] = 3
    result["atol"] = 1e-5
    result["rtol"] = 1e-4
    result["qualification_scope"] = (
        "network, gradient, short optimizer and codec parity; not full-budget training quality"
    )
    result["codec_fixtures_sha256"] = hashlib.sha256(
        (Path(destination) / "codec-fixtures.pt").read_bytes()
    ).hexdigest()
    (Path(destination) / "parity-result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
