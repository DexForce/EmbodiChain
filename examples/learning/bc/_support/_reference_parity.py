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

"""Compare the shared BC model/update against the pinned PR828 visual model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from embodichain_models.policy.bc.model import ChunkBCModel, ChunkBCModelCfg
from embodichain.learning.sl import SupervisedBatch, SupervisedLearner
from embodichain_models.policy.bc.supervised import ChunkBCObjective
from ._pick_place import CAMERAS, TARGET_SEMANTICS, pose_targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    reference_file = (
        args.reference_root / "examples/ods/data_stream_demo/visual_policy.py"
    )
    from ._reference import _load_reference

    _, module = _load_reference(args.reference_root)
    torch.manual_seed(321)
    reference = module.VisualChunkBC(
        torch.zeros(27),
        torch.ones(27),
        torch.zeros(10),
        torch.ones(10),
        CAMERAS,
        prediction_horizon=32,
    )
    model = ChunkBCModel(
        ChunkBCModelCfg(
            observation_dim=27,
            target_dim=10,
            history_length=2,
            prediction_horizon=32,
            cameras=CAMERAS,
            target_semantics=TARGET_SEMANTICS,
        )
    )
    converted = {}
    for name, value in reference.state_dict().items():
        name = {
            "proprio_mean": "observation_mean",
            "proprio_std": "observation_std",
        }.get(name, name)
        converted[name] = value
    converted["_extra_state"] = model.get_extra_state()
    model.load_state_dict(converted)
    geometry = torch.randn(1, 33, 51) * 0.1
    geometry[..., 21:27] = torch.tensor([1.0, 0, 0, 0, 1, 0])
    actions = torch.randn(1, 33, 7) * 0.1
    images = {camera: torch.rand(1, 2, 3, 256, 256) for camera in CAMERAS}
    loss_mask = torch.ones(1, 33, dtype=torch.bool)
    loss_mask[:, 0] = False
    valid = torch.ones_like(loss_mask)
    reference_prediction = reference(geometry[:, :2, :27], images)
    actual_prediction = model(geometry[:, :2, :27], images)
    forward_error = float((actual_prediction - reference_prediction).abs().max())
    arm, hand = reference.loss(
        geometry, actions, images, loss_mask=loss_mask, valid_mask=valid
    )
    reference_loss = (9 * arm + hand) / 10
    batch = SupervisedBatch(
        {"history": geometry[:, :2, :27], "images": images},
        pose_targets(geometry[:, 1:], actions[:, 1:]),
        {"target_valid": loss_mask[:, 1:]},
    )
    objective = ChunkBCObjective()
    actual_loss = objective.compute_loss(model, batch).loss
    loss_error = float((reference_loss - actual_loss).abs())
    reference_loss.backward()
    actual_loss.backward()
    gradients_match = all(
        torch.allclose(old.grad, new.grad, atol=1e-5, rtol=1e-4)
        for old, new in zip(reference.parameters(), model.parameters())
    )
    gradient_error = max(
        float((old.grad - new.grad).abs().max())
        for old, new in zip(reference.parameters(), model.parameters())
    )
    reference_optimizer = torch.optim.AdamW(
        reference.parameters(), lr=3e-4, weight_decay=1e-6
    )
    reference_optimizer.zero_grad(set_to_none=True)
    arm, hand = reference.loss(
        geometry, actions, images, loss_mask=loss_mask, valid_mask=valid
    )
    ((9 * arm + hand) / 10).backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1, error_if_nonfinite=True)
    reference_optimizer.step()
    learner = SupervisedLearner(
        model,
        objective,
        torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-6),
        max_grad_norm=1,
    )
    learner.update(batch)
    parameters_match = all(
        torch.allclose(old, new, atol=1e-5, rtol=1e-4)
        for old, new in zip(reference.parameters(), model.parameters())
    )
    parameter_error = max(
        float((old - new).abs().max())
        for old, new in zip(reference.parameters(), model.parameters())
    )
    passed = (
        torch.allclose(reference_prediction, actual_prediction, atol=1e-5, rtol=1e-4)
        and torch.allclose(reference_loss, actual_loss, atol=1e-5, rtol=1e-4)
        and gradients_match
        and parameters_match
    )
    report = {
        "protocol": "sl-bc-pr828-parity-v1",
        "reference_commit": "51f1f0cd5a7602f23358c142997e9045323f547a",
        "reference_encoder_sha256": hashlib.sha256(
            (
                args.reference_root / "embodichain/learning/models/resnet_encoder.py"
            ).read_bytes()
        ).hexdigest(),
        "reference_target_sha256": hashlib.sha256(
            (args.reference_root / "examples/ods/simple_bc_demo/models.py").read_bytes()
        ).hexdigest(),
        "model_namespace": "embodichain_models.policy.bc",
        "reference_source_sha256": hashlib.sha256(
            reference_file.read_bytes()
        ).hexdigest(),
        "device": "cpu",
        "forward_max_abs": forward_error,
        "loss_abs": loss_error,
        "gradient_max_abs": gradient_error,
        "parameter_max_abs": parameter_error,
        "atol": 1e-5,
        "rtol": 1e-4,
        "passed": bool(passed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, allow_nan=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
