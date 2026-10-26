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

"""Lazy offline training, evaluation and model export entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["run_train", "run_eval", "run_export"]

# These fields may only come from the dataset's measured deployment contract.
# Even an absent field cannot be invented through a training annotation override.
_DEPLOYMENT_MANIFEST_FIELDS = frozenset(
    {
        "robot",
        "codec",
        "dt",
        "joint_names",
        "q_dim",
        "qd_dim",
        "asset_sha256",
        "assets",
        "asset_hashes",
        "physics_signature",
        "physics_signature_schema",
        "gravity",
        "contact_schema",
        "world_rotation_xyzw",
        "gravity_transport_override",
        "source_physical_arrays",
        "legacy_velocity",
        "reference_friction_mapping",
    }
)


def _arguments(value: dict[str, Any], allowed: set[str]) -> None:
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown dynamics workflow fields: {sorted(unknown)}")


def _load_training_checkpoint(path: str | Path) -> dict[str, Any]:
    """Read a tensor-only training payload without constructing a default model."""
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(state, dict)
        or state.get("format") != "embodichain.nerd.training.v1"
    ):
        raise ValueError("Expected a dynamics training checkpoint")
    return state


def run_train(config: dict[str, Any]) -> dict[str, Any]:
    """Train from scratch, resume a committed run, or fine-tune an export.

    Configuration keys are ``dataset``, ``output_dir``, ``model``, ``train``,
    ``manifest``, optional ``statistics``, ``resume``, ``fine_tune``,
    ``include_prefix``, ``sample_order`` and ``stop_after``. Relative paths should
    be resolved by the configuration-file entry point before this call.
    """
    _arguments(
        config,
        {
            "dataset",
            "output_dir",
            "model",
            "train",
            "manifest",
            "statistics",
            "resume",
            "fine_tune",
            "include_prefix",
            "sample_order",
            "stop_after",
        },
    )
    import torch
    from embodichain_models.dynamics.nerd.codec import NeRDStateCodec
    from embodichain_models.dynamics.nerd import NeRDModel, NeRDModelCfg, load_model
    from embodichain_models.dynamics.nerd.data import (
        DynamicsWindowDataset,
        load_manifest,
    )
    from ._trainer import DynamicsTrainCfg, DynamicsTrainer, load_statistics
    from ._runtime import _training_runtime_identity

    if config.get("resume") and config.get("statistics"):
        raise ValueError("Resume restores frozen statistics; omit statistics override")
    if config.get("resume") and config.get("fine_tune"):
        raise ValueError("resume and fine_tune are mutually exclusive")
    cfg = DynamicsTrainCfg(**config.get("train", {}))
    torch.manual_seed(cfg.seed)
    dataset_metadata = load_manifest(config["dataset"])["metadata"]
    supplied_manifest = config.get("manifest")
    manifest = dataset_metadata if supplied_manifest is None else supplied_manifest
    if not isinstance(manifest, dict):
        raise ValueError("Training manifest must be a mapping")
    # Canonical JSON data is also the export contract; copy before training so an
    # external dictionary mutation cannot relabel a run after validation.
    manifest = json.loads(json.dumps(manifest, allow_nan=False))
    missing = object()
    for key in set(dataset_metadata) | _DEPLOYMENT_MANIFEST_FIELDS:
        if manifest.get(key, missing) != dataset_metadata.get(key, missing):
            raise ValueError(
                f"Training manifest/data {key} mismatch; annotations may add fields but cannot replace or drop dataset provenance"
            )
    codec = NeRDStateCodec(**manifest["codec"])
    if config.get("fine_tune"):
        model, pretrained_manifest = load_model(config["fine_tune"], device=cfg.device)
        if pretrained_manifest["codec"] != manifest["codec"] or pretrained_manifest.get(
            "dt"
        ) != manifest.get("dt"):
            raise ValueError("Fine-tuning requires compatible codec and timestep")
    elif config.get("resume"):
        from embodichain_models.dynamics.nerd.checkpoint import _model_from_state

        saved = _load_training_checkpoint(config["resume"])
        # Keep the requested architecture/feature order for resume preflight,
        # while restoring precision before trainer construction. Release the
        # full optimizer payload before resume reads its committed state again.
        model = _model_from_state(NeRDModelCfg(**config["model"]), saved["model"])
        del saved
    else:
        model = NeRDModel(NeRDModelCfg(**config["model"]))
    train = DynamicsWindowDataset(
        config["dataset"],
        "train",
        model.cfg.history_length,
        include_prefix=config.get("include_prefix", False),
        verify=True,
    )
    validation = DynamicsWindowDataset(
        config["dataset"],
        "validation",
        model.cfg.history_length,
        include_prefix=config.get("include_prefix", False),
    )
    statistics = (
        load_statistics(config["statistics"]) if config.get("statistics") else None
    )
    order = None
    if config.get("sample_order"):
        order_path = Path(config["sample_order"])
        order = (
            order_path
            if order_path.suffix in {".npy", ".json"}
            else torch.load(order_path, map_location="cpu", weights_only=True)
        )
    try:
        trainer = DynamicsTrainer(
            model,
            codec,
            train,
            validation,
            cfg,
            config["output_dir"],
            manifest=manifest,
            statistics=statistics,
            initialize_normalization=not config.get("resume")
            and (not config.get("fine_tune") or statistics is not None),
            sample_order=order,
        )
        if config.get("resume"):
            trainer.resume(config["resume"])
        result = trainer.train(stop_after=config.get("stop_after"))
        result["runtime_identity"] = _training_runtime_identity(trainer.model)
        Path(config["output_dir"], "result.json").write_text(
            json.dumps(result, indent=2, allow_nan=False)
        )
        return result
    finally:
        train.close()
        validation.close()


def run_eval(config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate an exported model on a fixed offline split, without simulation."""
    _arguments(
        config,
        {
            "dataset",
            "model",
            "split",
            "batch_size",
            "device",
            "output",
            "include_prefix",
        },
    )
    from embodichain_models.dynamics.nerd.codec import NeRDStateCodec
    from embodichain_models.dynamics.nerd import load_model
    from embodichain_models.dynamics.nerd.data import DynamicsWindowDataset
    from ._trainer import evaluate_model

    device = config.get("device", "cpu")
    model, manifest = load_model(config["model"], device=device)
    codec = NeRDStateCodec(**manifest["codec"])
    dataset = DynamicsWindowDataset(
        config["dataset"],
        config.get("split", "test"),
        model.cfg.history_length,
        include_prefix=config.get("include_prefix", False),
        verify=True,
    )
    try:
        metadata = dataset.manifest["metadata"]
        if metadata["codec"] != manifest["codec"] or metadata.get("dt") != manifest.get(
            "dt"
        ):
            raise ValueError("Evaluation data/model codec or timestep mismatch")
        result = {
            **evaluate_model(
                model, codec, dataset, config.get("batch_size", 64), device
            ),
            "dataset_fingerprint": dataset.fingerprint,
            "qualification": "offline_only",
            "numerical_profile": manifest.get("numerical_profile", "default"),
            "attention_backend": model.cfg.attention_backend,
        }
        if config.get("output"):
            output = Path(config["output"])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2, allow_nan=False))
        return result
    finally:
        dataset.close()


def run_export(checkpoint: str | Path, output: str | Path) -> dict[str, Any]:
    """Export NeRD or shared-SL training tensors without arbitrary unpickling."""
    import importlib
    import numpy as np
    import torch

    from embodichain_models.dynamics.nerd import NeRDModelCfg, export_model
    from embodichain_models.dynamics.nerd.checkpoint import _model_from_state
    from embodichain_models.dynamics.nerd._numerics import _checkpoint_numerical_profile

    # Generic SL records NumPy's MT19937 RNG state, unlike the historical
    # tensor-only NeRD checkpoint. Restrict loading to those known array types.
    array_module = importlib.import_module(
        "numpy._core.multiarray" if hasattr(np, "_core") else "numpy.core.multiarray"
    )
    reconstruct = array_module._reconstruct
    with torch.serialization.safe_globals(
        [
            np.ndarray,
            np.dtype,
            type(np.dtype("uint32")),
            (reconstruct, "numpy.core.multiarray._reconstruct"),
            (reconstruct, "numpy._core.multiarray._reconstruct"),
        ]
    ):
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    # SL v2 adds rank-local recovery state; portable export still uses the
    # committed rank-zero model with the same recipe and ordered feature layout.
    if state.get("version") in (1, 2) and "learner" in state:
        signature = state["signature"]
        model_cfg = dict(signature["model"])
        weights = state["learner"]["model"]
        layout = signature.get("model_input_layout")
        if layout is None:
            # Earlier SL checkpoints canonicalized signature dictionaries, but
            # state_dict retains the model's ordered normalizer registration.
            names = [
                key[len("input_rms.") : -len(".mean")]
                for key in weights
                if key.startswith("input_rms.") and key.endswith(".mean")
            ]
            layout = [(name, model_cfg["input_dims"][name]) for name in names]
        if (
            len(layout) != len(model_cfg["input_dims"])
            or dict(layout) != model_cfg["input_dims"]
        ):
            raise ValueError("SL checkpoint has an invalid ordered NeRD feature layout")
        model_cfg["input_dims"] = dict(layout)
        profile = signature["objective"]["numerical_profile"]
        model = _model_from_state(NeRDModelCfg(**model_cfg), weights)
        manifest = {
            **signature["physics"],
            "numerical_profile": profile,
            "attention_backend": model.cfg.attention_backend,
            "training": {
                "global_step": state["learner"]["num_updates"],
                "qualification": "unqualified",
                "numerical_profile": profile,
                "attention_backend": model.cfg.attention_backend,
            },
        }
        export_model(output, model, manifest)
        return {
            "output": str(output),
            "global_step": state["learner"]["num_updates"],
            "qualification": "unqualified",
            "numerical_profile": profile,
            "attention_backend": model.cfg.attention_backend,
        }
    if state.get("format") != "embodichain.nerd.training.v1":
        raise ValueError("Unsupported NeRD or shared-SL training checkpoint")
    profile_state = state
    if state.get("resumable") is False and "train_cfg" not in state:
        # A selected-model snapshot has no optimizer/training state. Retain the
        # same execution-label validation without inventing resumable state.
        profile_state = {
            **state,
            "train_cfg": {
                "numerical_profile": state.get("numerical_profile", "default")
            },
        }
    profile = _checkpoint_numerical_profile(profile_state)
    model = _model_from_state(NeRDModelCfg(**state["model_cfg"]), state["model"])
    manifest = {
        **state["manifest"],
        "numerical_profile": profile,
        "attention_backend": model.cfg.attention_backend,
        "training": {
            "global_step": state["global_step"],
            "dataset_fingerprint": state["dataset_fingerprint"],
            "qualification": "unqualified",
            "numerical_profile": profile,
            "attention_backend": model.cfg.attention_backend,
        },
    }
    export_model(output, model, manifest)
    return {
        "output": str(output),
        "global_step": state["global_step"],
        "qualification": "unqualified",
        "numerical_profile": profile,
        "attention_backend": model.cfg.attention_backend,
    }
