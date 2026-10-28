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

"""Strict import of official RigidFormer weights without optimizer conversion."""

from __future__ import annotations
from collections.abc import Mapping
from pathlib import Path
import torch
from .model import RigidFormerCfg, RigidFormerModel

__all__ = ["load_rigidformer_checkpoint", "export_rigidformer_checkpoint"]


def load_rigidformer_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    attention_backend: str = "sdpa",
) -> RigidFormerModel:
    """Load an official model/EMA weights file and its architecture.

    Args:
        path: Official tensor/config checkpoint; decoded with weights_only=True.
        device: Destination device, independent of the file's saved device.
        attention_backend: Explicit native attention implementation.

    Returns:
        Evaluation model with every author weight loaded strictly.

    This is a weights import, not recovery of an official training process.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if (
        not isinstance(payload, Mapping)
        or "model_state_dict" not in payload
        or "config" not in payload
    ):
        raise ValueError(
            "Expected official model_state_dict and config; community/SL checkpoints are distinct formats"
        )
    cfg = RigidFormerCfg(
        architecture=dict(payload["config"]), attention_backend=attention_backend
    )
    model = RigidFormerModel(cfg)
    incoming = payload["model_state_dict"]
    if not isinstance(incoming, Mapping) or not all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in incoming.items()
    ):
        raise ValueError("Official weights must be a named tensor mapping")
    prefix = all(key.startswith("module.") for key in incoming)
    state = {
        key.removeprefix("module.") if prefix else key: value
        for key, value in incoming.items()
    }
    expected = model.network.state_dict()
    if state.keys() != expected.keys():
        raise ValueError(
            "Official checkpoint keys differ from the configured native architecture"
        )
    for name, value in expected.items():
        if state[name].shape != value.shape or state[name].dtype != value.dtype:
            raise ValueError(f"Official checkpoint tensor differs: {name}")
    model.network.load_state_dict(state, strict=True)
    return model.to(device).eval()


def export_rigidformer_checkpoint(
    source: str | Path, destination: str | Path, *, weights: str = "ema"
) -> Path:
    """Export a native SL training checkpoint to tensor/config inference weights.

    Args:
        source: Checkpoint produced by the native author-based SL recipe.
        destination: Fresh output path; existing files are rejected.
        weights: Explicit ema or raw model selection.

    Returns:
        Written inference checkpoint, loadable by load_rigidformer_checkpoint.
    """
    import numpy as np

    if weights not in ("ema", "raw"):
        raise ValueError("weights must be ema or raw")
    output = Path(destination)
    if output.exists():
        raise FileExistsError(output)
    try:
        reconstruct = np._core.multiarray._reconstruct
    except AttributeError:  # NumPy 1.x
        reconstruct = np.core.multiarray._reconstruct
    safe = [
        (reconstruct, "numpy.core.multiarray._reconstruct"),
        (reconstruct, "numpy._core.multiarray._reconstruct"),
        np.ndarray,
        np.dtype,
        type(np.dtype(np.uint32)),
    ]
    with torch.serialization.safe_globals(safe):
        state = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    signature = state.get("signature", {})
    if signature.get("family") != "rigidformer-author-v1":
        raise ValueError("Expected a native author-based RigidFormer SL checkpoint")
    raw = state["learner"]["model"]
    if weights == "ema":
        extra = state.get("training_states", {}).get("ema", {}).get("state")
        if extra is None:
            raise ValueError("Checkpoint has no EMA; select raw explicitly")
        selected = {k: v.to(raw[k].dtype) for k, v in extra["shadow"].items()}
    else:
        selected = raw
    if not all(k.startswith("network.") for k in selected):
        raise ValueError("Native model weight namespace differs")
    payload = {
        "model_state_dict": {
            k.removeprefix("network."): v.detach().clone() for k, v in selected.items()
        },
        "config": signature["reference_config"],
        "source": {
            "format": "embodichain.sl",
            "author_revision": signature["author_revision"],
            "num_updates": state["learner"]["num_updates"],
            "weights": weights,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    return output
