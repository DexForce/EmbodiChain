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

"""Executable CNN qualification against pinned RLInf and image-only targets.

This is an engineering qualification entry point, not a timing benchmark or a
policy-performance result. CUDA use is explicit; no GPU is selected implicitly.
"""

from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from functools import partial
import hashlib
import json
from pathlib import Path
import time
from typing import Any
from urllib.request import urlopen

import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, ResNet

from embodichain.learning.models import ResNet10Encoder
from examples.ods.data_stream_demo.visual_policy import VisualChunkBC

__all__ = []

_REFERENCE_COMMIT = "91a302ec50ddacd1fdabc5859b85e3661d768236"
_REFERENCE_SHA256 = "5776e8a65d1d14dbcfeae070cb0ef23773d2a00a21a59f4f5f5b1933dae4cd47"
_REFERENCE_URL = (
    "https://raw.githubusercontent.com/RLinf/RLinf/"
    f"{_REFERENCE_COMMIT}/rlinf/models/embodiment/modules/resnet_utils.py"
)


def _upstream_reference(source_path: Path | None) -> tuple[nn.Module, nn.Module]:
    source = (
        source_path.read_bytes()
        if source_path is not None
        else urlopen(_REFERENCE_URL, timeout=30).read()
    )
    digest = hashlib.sha256(source).hexdigest()
    if digest != _REFERENCE_SHA256:
        raise ValueError(f"Pinned RLInf source SHA256 mismatch: {digest}")
    selected = {"MyGroupNorm", "ResNet10", "SpatialLearnedEmbeddings"}
    tree = ast.Module(
        body=[
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.ClassDef) and node.name in selected
        ],
        type_ignores=[],
    )
    namespace = {
        "torch": torch,
        "nn": nn,
        "partial": partial,
        "BasicBlock": BasicBlock,
        "ResNet": ResNet,
    }
    # Only hash-verified, explicitly selected upstream class bodies are executed.
    exec(compile(tree, _REFERENCE_URL, "exec"), namespace)
    backbone = namespace["ResNet10"](pre_pooling=True)
    del backbone.fc
    del backbone.avgpool
    pooling = namespace["SpatialLearnedEmbeddings"](8, 8, 512, 8)
    return backbone, pooling


def _reference_check(source_path: Path | None) -> dict[str, Any]:
    torch.manual_seed(481)
    actual = ResNet10Encoder()
    backbone, pooling = _upstream_reference(source_path)
    backbone.load_state_dict(actual.resnet_backbone.state_dict())
    pooling.load_state_dict(actual.pooling_layer.state_dict())
    projection = deepcopy(actual.mlp)
    images = torch.rand(1, 3, 256, 256, requires_grad=True)
    upstream_images = images.detach().clone().requires_grad_()
    intermediates: dict[str, list[torch.Tensor]] = {}
    handles = []
    for name in ("conv1", "layer1", "layer2", "layer3", "layer4"):
        intermediates[name] = []
        for module in (getattr(actual.resnet_backbone, name), getattr(backbone, name)):
            handles.append(
                module.register_forward_hook(
                    lambda _module, _args, output, name=name: intermediates[
                        name
                    ].append(output.detach().clone())
                )
            )
    result = actual(images)
    upstream_features = backbone(
        (upstream_images - actual.image_mean) / actual.image_std
    )
    upstream_embedding = pooling(upstream_features)
    expected = projection(upstream_embedding)
    for handle in handles:
        handle.remove()
    for values in intermediates.values():
        torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)
    actual_embedding = actual.pooling_layer(upstream_features)
    torch.testing.assert_close(
        actual_embedding, upstream_embedding, rtol=2e-5, atol=5e-5
    )
    torch.testing.assert_close(result, expected, rtol=5e-5, atol=5e-6)
    objective = torch.linspace(-1, 1, 256)
    (result * objective).sum().backward()
    (expected * objective).sum().backward()
    torch.testing.assert_close(images.grad, upstream_images.grad, rtol=5e-4, atol=2e-5)
    differences = {}
    for section, current, reference in (
        ("backbone", actual.resnet_backbone, backbone),
        ("spatial", actual.pooling_layer, pooling),
        ("projection", actual.mlp, projection),
    ):
        for (name, parameter), (_, upstream_parameter) in zip(
            current.named_parameters(), reference.named_parameters(), strict=True
        ):
            assert parameter.grad is not None and upstream_parameter.grad is not None
            torch.testing.assert_close(
                parameter.grad, upstream_parameter.grad, rtol=5e-4, atol=3e-4
            )
            differences[f"{section}.{name}"] = float(
                (parameter.grad - upstream_parameter.grad).abs().max()
            )
    return {
        "passed": True,
        "commit": _REFERENCE_COMMIT,
        "source_sha256": _REFERENCE_SHA256,
        "intermediates": list(intermediates),
        "output_max_abs_error": float((result - expected).abs().max().detach()),
        "embedding_max_abs_error": float(
            (actual_embedding - upstream_embedding).abs().max().detach()
        ),
        "parameter_gradient_max_abs_errors": differences,
        "scope": "encoder structure/weights with freeze, detach and dropout removed",
    }


def _policy(cameras: tuple[str, ...], horizon: int = 2) -> VisualChunkBC:
    return VisualChunkBC(
        torch.zeros(27),
        torch.ones(27),
        torch.zeros(10),
        torch.ones(10),
        cameras,
        horizon,
    )


def _policy_contract_check() -> dict[str, Any]:
    torch.manual_seed(712)
    policy = _policy(("third_person", "wrist"))
    proprio = torch.rand(2, 2, 27)
    images = {camera: torch.rand(2, 2, 3, 256, 256) for camera in policy.cameras}
    observed = []
    handle = policy.encoder.register_forward_pre_hook(
        lambda _module, arguments: observed.append(arguments[0].detach().clone())
    )
    prediction = policy(proprio, images)
    handle.remove()
    expected_images = torch.stack([images[c] for c in policy.cameras], 2).flatten(0, 2)
    torch.testing.assert_close(observed[0], expected_images, rtol=0, atol=0)
    with torch.no_grad():
        batch_features = policy.encoder(expected_images)
        individual = torch.cat(
            [policy.encoder(frame[None]) for frame in expected_images]
        )
    torch.testing.assert_close(batch_features, individual, rtol=5e-5, atol=8e-6)
    for camera in policy.cameras:
        changed = dict(images)
        changed[camera] = 1 - images[camera]
        assert not torch.allclose(prediction, policy(proprio, changed))
    geometry = torch.rand(2, 3, 51)
    actions = torch.rand(2, 3, 7) * 0.1
    valid = torch.ones(2, 3, dtype=torch.bool)
    mask = valid.clone()
    mask[:, 0] = False
    original_inputs = []
    hook = policy.register_forward_pre_hook(
        lambda _module, args: original_inputs.append(args[0].detach().clone())
    )
    first_loss = policy.loss(
        geometry, actions, images, loss_mask=mask, valid_mask=valid
    )
    altered = geometry.clone()
    altered[:, 2:] += 0.3
    policy.loss(altered, actions, images, loss_mask=mask, valid_mask=valid)
    hook.remove()
    torch.testing.assert_close(original_inputs[0], original_inputs[1], rtol=0, atol=0)
    assert original_inputs[0].shape == (2, 2, 27)
    (first_loss[0] + first_loss[1]).backward()
    gradients = {}
    for name, parameter in policy.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.norm() > 0, name
        gradients[name] = float(parameter.grad.norm())
    restored = _policy(policy.cameras)
    restored.load_state_dict(deepcopy(policy.state_dict()))
    with torch.no_grad():
        torch.testing.assert_close(
            restored(proprio, images), prediction, rtol=0, atol=0
        )
    rejected = 0
    for invalid in (
        {"third_person": images["third_person"]},
        {**images, "extra": images["wrist"]},
    ):
        try:
            policy(proprio, invalid)
        except ValueError:
            rejected += 1
    try:
        _policy(tuple(reversed(policy.cameras))).load_state_dict(policy.state_dict())
    except ValueError:
        rejected += 1
    assert rejected == 3
    return {
        "passed": True,
        "flatten_order": "batch, history, configured camera",
        "future_geometry_does_not_enter_policy": True,
        "camera_and_checkpoint_rejections": rejected,
        "parameter_gradient_norms": gradients,
    }


def _fit_check(device: torch.device, steps: int) -> dict[str, Any]:
    torch.manual_seed(918)
    results = {}
    for cameras in (("third_person",), ("third_person", "wrist")):
        policy = _policy(cameras, horizon=2).to(device)
        count = 2 ** len(cameras)
        bits = torch.tensor(
            [
                [float((sample >> view) & 1) for view in range(len(cameras))]
                for sample in range(count)
            ],
            device=device,
        )
        proprio = torch.zeros(count, 2, 27, device=device)
        images = {}
        for view, camera in enumerate(cameras):
            image = torch.full((count, 2, 3, 256, 256), 0.25, device=device)
            # Spatially localized RGB information is supplied solely by this view.
            image[:, :, view, 32:224, 32:224] = (
                0.1 + bits[:, view, None, None, None] * 0.8
            )
            images[camera] = image
        targets = torch.zeros(count, 2, 10, device=device)
        for view in range(len(cameras)):
            targets[:, :, view] = (bits[:, view, None] * 2 - 1) * 0.5
        optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-4, weight_decay=1e-6)
        before = {name: p.detach().clone() for name, p in policy.named_parameters()}
        with torch.no_grad():
            initial = float((policy(proprio, images) - targets).square().mean())
        losses = []
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = (policy(proprio, images) - targets).square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1)
            optimizer.step()
            losses.append(float(loss.detach()))
        with torch.no_grad():
            fitted = policy(proprio, images)
            final = float((fitted - targets).square().mean())
            max_error = float((fitted - targets).abs().max())
        assert final < initial * 0.05 and max_error < 0.15, (initial, final, max_error)
        changed_stages = {}
        for stage in (
            "encoder.resnet_backbone.conv1",
            "encoder.resnet_backbone.layer1",
            "encoder.resnet_backbone.layer2",
            "encoder.resnet_backbone.layer3",
            "encoder.resnet_backbone.layer4",
            "encoder.pooling_layer",
            "encoder.mlp",
            "head",
        ):
            changed_stages[stage] = any(
                not torch.equal(parameter.detach(), before[name])
                for name, parameter in policy.named_parameters()
                if name.startswith(stage)
            )
        assert all(changed_stages.values())
        ablations = {}
        for camera in cameras:
            ablated = dict(images)
            ablated[camera] = images[camera][:1].expand_as(images[camera])
            with torch.no_grad():
                ablated_loss = float(
                    (policy(proprio, ablated) - targets).square().mean()
                )
            assert ablated_loss > max(final * 5, 0.01)
            ablations[camera] = ablated_loss
        results["single" if len(cameras) == 1 else "dual"] = {
            "initial_mse": initial,
            "final_mse": final,
            "max_abs_error": max_error,
            "steps": steps,
            "samples": count,
            "proprioception_constant": True,
            "view_ablation_mse": ablations,
            "updated_stages": changed_stages,
            "losses": losses,
        }
    return {"passed": True, "cases": results}


def _cuda_check(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(44)
    cpu = ResNet10Encoder().eval()
    image = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        expected = cpu(image)
    gpu = deepcopy(cpu).to(device)
    with torch.no_grad():
        actual = gpu(image.to(device)).cpu()
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=2e-4)
    del gpu
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = _policy(("third_person", "wrist"), horizon=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-6)
    images = {
        camera: torch.rand(32, 2, 3, 256, 256, device=device)
        for camera in model.cameras
    }
    optimizer.zero_grad(set_to_none=True)
    prediction = model(torch.zeros(32, 2, 27, device=device), images)
    prediction.square().mean().backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    return {
        "passed": True,
        "device": str(device),
        "name": torch.cuda.get_device_name(device),
        "cpu_cuda_max_abs_error": float((actual - expected).abs().max()),
        "b32_dual_h2_f32_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "b32_dual_h2_f32_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-source", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fit-steps", type=int, default=160)
    parser.add_argument("--skip-fit", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.fit_steps < 1 or args.cpu_threads < 1:
        parser.error("fit steps and CPU threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    result: dict[str, Any] = {
        "passed": False,
        "torch": torch.__version__,
        "device": args.device,
        "reference": _reference_check(args.reference_source),
        "policy_contract": _policy_contract_check(),
    }
    device = torch.device(args.device)
    if not args.skip_fit:
        result["image_only_fit"] = _fit_check(device, args.fit_steps)
    if device.type == "cuda":
        result["cuda"] = _cuda_check(device)
    result["passed"] = True
    result["wall_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "output": str(args.output),
                "seconds": result["wall_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
