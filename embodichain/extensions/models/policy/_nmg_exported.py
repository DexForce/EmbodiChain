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

"""Load and validate an upstream NMG exported PyTorch actor."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING, Self
from zipfile import BadZipFile

import torch
from torch import nn

from ._nmg_observation import _OnnxObservationDimensions, _WaypointObservationLayout

if TYPE_CHECKING:
    from torch.export import ExportedProgram

_PROGRAM_FORMAT = "nmg.actor.pt2.v1"
_RELEASED_SHA256: dict[str, str] = {
    "franka": "07ae3c861bbe7cc599fd12cecd0c0909d06c032c73623d7e36ceb798f48e2c87",
    "piper": "b0f027ba7010f84be917ce83fa60d97745c0fbe03e9e554982cae9d2d5e7afec",
}


class _FixedEvaluationActor(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self._module = module.requires_grad_(False)

    def train(self, mode: bool = True) -> Self:
        # ExportedProgram captures evaluation semantics and disables train/eval.
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        return self

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self._module(obs)


def _verify_sha256(data: bytes, expected: str | None) -> None:
    # torch.export.load can deserialize pickle. Check the very bytes passed to
    # the loader against the published release or a caller's trusted digest.
    actual = hashlib.sha256(data).hexdigest()
    if expected is None:
        if actual not in _RELEASED_SHA256.values():
            raise ValueError(
                "Provide a trusted sha256 for an unpublished NMG PT2 policy."
            )
        return
    if (
        not isinstance(expected, str)
        or re.fullmatch(r"[0-9a-fA-F]{64}", expected) is None
    ):
        raise ValueError("NMG checkpoint SHA256 must have 64 hexadecimal characters.")
    if actual != expected.lower():
        raise ValueError("NMG checkpoint SHA256 mismatch.")


def _tensor_signature(
    program: ExportedProgram, *, input_tensor: bool
) -> tuple[int | str, int]:
    signature = program.graph_signature
    specs = signature.input_specs if input_tensor else signature.output_specs
    kind = "USER_INPUT" if input_tensor else "USER_OUTPUT"
    matching = [spec.arg.name for spec in specs if spec.kind.name == kind]
    if len(matching) != 1:
        raise ValueError(
            "An NMG PT2 policy must expose exactly one input and one output."
        )
    values = {node.name: node.meta.get("val") for node in program.graph.nodes}
    val = values.get(matching[0])
    if not isinstance(val, torch.Tensor) or val.dtype != torch.float32 or val.ndim != 2:
        raise ValueError(
            "An NMG PT2 policy requires batched float32 input and output tensors."
        )
    batch, width = val.shape
    if type(width) is not int or width < 1 or (type(batch) is int and batch < 1):
        raise ValueError("An NMG PT2 policy requires known, positive feature widths.")
    return batch if type(batch) is int else str(batch), width


def _load_exported_actor(
    path: Path, *, device: torch.device, sha256: str | None
) -> tuple[
    _FixedEvaluationActor,
    tuple[int | str, int],
    tuple[int | str, int],
    dict[str, str],
]:
    data = path.read_bytes()
    _verify_sha256(data, sha256)
    files = {"metadata.json": "", "deployment.json": ""}
    try:
        program = torch.export.load(io.BytesIO(data), extra_files=files)
        deployment = json.loads(files["deployment.json"])
        metadata = json.loads(files["metadata.json"])
    except (BadZipFile, KeyError, RuntimeError, ValueError, TypeError) as exc:
        raise ValueError("Invalid or incompatible NMG PT2 policy archive.") from exc

    if not isinstance(deployment, dict) or deployment.get("format") != _PROGRAM_FORMAT:
        raise ValueError(f"Expected an NMG {_PROGRAM_FORMAT} policy archive.")
    if not isinstance(metadata, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in metadata.items()
    ):
        raise ValueError("NMG PT2 observation metadata must contain strings.")
    dims = _OnnxObservationDimensions.from_metadata(metadata)
    layout = _WaypointObservationLayout(
        dims.num_waypoints,
        dims.num_controlled_joints,
        metadata.get("nmg.use_relative_obs") == "true",
        metadata.get("nmg.canonicalize_quat_obs") == "true",
        "nmg.joint_periodic_encoding" in metadata,
    )
    if any(metadata.get(k) != v for k, v in layout.onnx_metadata.items()):
        raise ValueError("NMG PT2 observation fingerprint or layout mismatch.")

    obs_signature = _tensor_signature(program, input_tensor=True)
    action_signature = _tensor_signature(program, input_tensor=False)
    if obs_signature[1] != layout.dim or action_signature != (
        obs_signature[0],
        layout.num_controlled_joints,
    ):
        raise ValueError(
            "NMG PT2 observation and action signatures do not match its layout."
        )
    if device.type != "cpu":
        from torch.export.passes import move_to_device_pass

        program = move_to_device_pass(program, device)
    return (
        _FixedEvaluationActor(program.module()),
        obs_signature,
        action_signature,
        metadata,
    )
