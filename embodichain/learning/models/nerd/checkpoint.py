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

"""Versioned portable NeRD exports and explicitly trusted legacy conversion."""

from __future__ import annotations
from collections.abc import Mapping
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any
import torch
from torch import nn
from .model import NeRDModel, NeRDModelCfg

__all__ = ["export_model", "load_model", "convert_legacy_checkpoint"]
_FORMAT_VERSION = 1


def _execution_metadata(manifest: dict[str, Any], cfg: NeRDModelCfg) -> str:
    """Validate execution labels without importing training into model loading."""
    training = manifest.get("training", {})
    training = training if isinstance(training, dict) else {}
    profile = manifest.get(
        "numerical_profile", training.get("numerical_profile", "default")
    )
    if not isinstance(profile, str) or profile not in {"default", "reference"}:
        raise ValueError("Unknown exported numerical_profile")
    if training.get("numerical_profile", profile) != profile:
        raise ValueError("Export numerical_profile metadata mismatch")
    if (
        manifest.get("attention_backend", cfg.attention_backend)
        != cfg.attention_backend
        or training.get("attention_backend", cfg.attention_backend)
        != cfg.attention_backend
    ):
        raise ValueError("Export attention_backend metadata mismatch")
    if profile == "reference" and cfg.attention_backend != "math":
        raise ValueError("Reference exports require math attention_backend")
    return profile


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_from_state(
    cfg: NeRDModelCfg,
    state: Mapping[str, torch.Tensor],
    device: str | torch.device = "cpu",
) -> NeRDModel:
    """Restore saved floating precision before copying parameters and statistics."""
    dtype = next(value.dtype for value in state.values() if value.is_floating_point())
    model = NeRDModel(cfg).to(dtype=dtype)
    model.load_state_dict(state, strict=True)
    return model.to(device)


def export_model(path: str | Path, model: NeRDModel, manifest: dict[str, Any]) -> None:
    """Write tensor weights and JSON schema without executable Python objects.

    Args:
        path: Output directory, which must not already contain an export.
        model: Model whose parameters and statistics are exported on CPU.
        manifest: Deployment metadata, including codec and timestep contracts.
    """
    path = Path(path)
    if (path / "manifest.json").exists() or (path / "weights.pt").exists():
        raise FileExistsError(f"Model export already exists: {path}")
    metadata = dict(manifest)
    profile = _execution_metadata(metadata, model.cfg)
    metadata.update(
        numerical_profile=profile, attention_backend=model.cfg.attention_backend
    )
    metadata.update(format_version=_FORMAT_VERSION, model_config=model.cfg.to_dict())
    json.dumps(metadata, allow_nan=False)
    path.mkdir(parents=True, exist_ok=True)
    weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    temporary = path / "weights.pt.tmp"
    torch.save(weights, temporary)
    temporary.replace(path / "weights.pt")
    metadata["weights_sha256"] = _digest(path / "weights.pt")
    temporary = path / "manifest.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    temporary.replace(path / "manifest.json")


def load_model(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[NeRDModel, dict[str, Any]]:
    """Verify and load an exported model; returned model is in evaluation mode."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("format_version") != _FORMAT_VERSION:
        raise ValueError("Unsupported NeRD export format_version")
    if _digest(path / "weights.pt") != manifest.get("weights_sha256"):
        raise ValueError("NeRD weights checksum mismatch")
    cfg = NeRDModelCfg(**manifest["model_config"])
    _execution_metadata(manifest, cfg)
    state = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
    return _model_from_state(cfg, state, device).eval(), manifest


class _LegacyModule(nn.Module):
    pass


class _LegacyObject:
    pass


class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        # Remap historical source locations without importing a checkout into runtime.
        if module in {
            "models.models",
            "models.base_models",
            "models.model_transformer",
        }:
            return _LegacyObject if name == "GPTConfig" else _LegacyModule
        if module == "utils.running_mean_std" and name == "RunningMeanStd":
            return _LegacyObject
        return super().find_class(module, name)


class _LegacyPickle:
    __name__ = "nerd_legacy_pickle"
    Unpickler = _LegacyUnpickler
    load = staticmethod(pickle.load)


def convert_legacy_checkpoint(
    checkpoint: str | Path,
    destination: str | Path,
    manifest: dict[str, Any],
    *,
    trust_pickle: bool = False,
) -> None:
    """Convert a trusted official Transformer checkpoint into a portable export.

    Args:
        checkpoint: Official model.pt, containing a serialized Python model.
        destination: New export directory.
        manifest: Deployment metadata validated by the eventual provider.
        trust_pickle: Must be explicitly true: legacy pickle can execute code.

    Only the official identity encoder/ReLU head/Transformer architecture is
    accepted. Arbitrary architectures must be migrated explicitly.
    """
    if not trust_pickle:
        raise ValueError("Legacy checkpoints require trust_pickle=True")
    old = torch.load(
        checkpoint, map_location="cpu", weights_only=False, pickle_module=_LegacyPickle
    )
    if isinstance(old, (list, tuple)):
        old = old[0]
    if not getattr(old, "is_transformer", False) or getattr(old, "is_rnn", False):
        raise ValueError("Only official Transformer checkpoints are supported")
    if list(old.encoders) != ["low_dim"] or len(old.encoders["low_dim"].body):
        raise ValueError("Only identity low-dimensional encoders are supported")
    layers = list(old.model.feature_net.body)
    if len(layers) % 2 or any(
        not isinstance(m, nn.Linear) if i % 2 == 0 else not isinstance(m, nn.ReLU)
        for i, m in enumerate(layers)
    ):
        raise ValueError("Only Linear/ReLU prediction heads are supported")

    def rename(name: str) -> str:
        return "joint_f" if name == "joint_acts" else name

    c = old.transformer_model.config
    dims = {
        rename(k): int(old.input_rms[k].mean.numel()) for k in old.low_dim_input_names
    }
    training = manifest.get("training", {})
    training = training if isinstance(training, dict) else {}
    profile = manifest.get(
        "numerical_profile", training.get("numerical_profile", "default")
    )
    cfg = NeRDModelCfg(
        input_dims=dims,
        output_dim=int(old.output_rms.mean.numel()),
        n_embd=c.n_embd,
        n_layer=c.n_layer,
        n_head=c.n_head,
        block_size=c.block_size,
        history_length=int(manifest.get("history_length", 10)),
        head_hidden_dims=tuple(
            m.out_features for m in layers if isinstance(m, nn.Linear)
        ),
        bias=c.bias,
        dropout=c.dropout,
        normalize_input=old.normalize_input,
        normalize_output=old.normalize_output,
        output_tanh=old.output_tanh,
        attention_backend=manifest.get(
            "attention_backend", "math" if profile == "reference" else "default"
        ),
    )
    model = NeRDModel(cfg)
    state = old.state_dict()
    result = model.load_state_dict(state, strict=False)
    expected_missing = {
        k for k in model.state_dict() if k.startswith(("input_rms.", "output_rms."))
    }
    if set(result.missing_keys) != expected_missing or result.unexpected_keys:
        raise ValueError(f"Unexpected legacy parameter schema: {result}")
    model.set_normalization(
        {rename(k): (v.mean, v.var) for k, v in old.input_rms.items()},
        old.output_rms.mean,
        old.output_rms.var,
    )
    metadata = dict(manifest)
    metadata["legacy_checkpoint_sha256"] = _digest(Path(checkpoint))
    export_model(destination, model, metadata)
