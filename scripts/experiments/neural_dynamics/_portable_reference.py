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

"""Restore reference/math exports in the unchanged original model classes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from official_evaluation_data import _hash

__all__: list[str] = []
COMMIT = "370e89d2b8ce68aa3578aa7812859c681b7502bc"


def array_sha256(value: Any) -> str:
    """Bind shape, dtype and bytes of one raw state/action array."""
    import numpy as np

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"shape": array.shape, "dtype": array.dtype.str}, sort_keys=True
        ).encode()
    )
    digest.update(array.tobytes())
    return digest.hexdigest()


def verify_source(
    source: Path, metadata: dict[str, Any], transport: Path
) -> dict[str, Any]:
    """Verify source and postprocessing identities before external execution."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("This CPU-only runner requires CUDA_VISIBLE_DEVICES=''")
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != COMMIT:
        raise ValueError("Official source commit mismatch")
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "diff", "--name-only", "HEAD"], text=True
    )
    if dirty.strip():
        raise ValueError("Official tracked source files must remain unchanged")
    verified = {}
    for name, expected in metadata["source_runtime_files"].items():
        actual = _hash(source / name)
        if actual != expected:
            raise ValueError("Official runtime file checksum mismatch: " + name)
        verified[name] = actual
    # These files affect inference but were not all listed in the old teacher.
    for name in (
        "models/models.py",
        "models/model_transformer.py",
        "models/base_models.py",
        "utils/running_mean_std.py",
    ):
        verified[name] = _hash(source / name)
    verified_transport = {}
    for name, identity in metadata["canonical_transport_modules"].items():
        actual = _hash(transport / (name + ".py"))
        if actual != identity["sha256"]:
            raise ValueError("Frozen postprocessing module checksum mismatch: " + name)
        verified_transport[name] = actual
    return {
        "commit": commit,
        "tracked_clean": True,
        "files": verified,
        "postprocessing": verified_transport,
    }


def load_portable(
    directory: Path, robot: str, scope: str
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Restore FP32 weights/RMS exactly; attention is set by the caller's context.

    The portable JSON must explicitly declare reference/math and source world.
    Only the feature spelling joint_f -> joint_acts changes. RMS counts are not
    portable, and the frozen inference method reads only the stored mean/var.
    """
    import torch
    from models.models import ModelMixedInput
    from utils.running_mean_std import RunningMeanStd

    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    cfg = manifest["model_config"]
    if (
        manifest.get("robot") != robot
        or manifest.get("world_rotation_xyzw", "missing") is not None
    ):
        raise ValueError(
            "Portable model must declare the matching unrotated source world"
        )
    if (
        manifest.get("numerical_profile"),
        manifest.get("attention_backend"),
        cfg.get("attention_backend"),
    ) != ("reference", "math", "math"):
        raise ValueError(
            "Portable model must explicitly declare reference/math at every level"
        )
    if cfg["history_length"] != 10:
        raise ValueError("This fixed protocol requires the original ten-state history")
    training = manifest.get("training", {})
    if training:
        if (training.get("numerical_profile"), training.get("attention_backend")) != (
            "reference",
            "math",
        ):
            raise ValueError(
                "Training and model numerical policy declarations disagree"
            )
        updates = training["global_step"]
    else:
        initial_path = directory / "identity.json"
        if not initial_path.exists():
            raise ValueError("Missing exact update provenance")
        updates = json.loads(initial_path.read_text())["source_update_count"]
    if type(updates) is not int or updates < 0:
        raise ValueError("Invalid checkpoint update count")
    if updates == 0 and scope != "smoke":
        raise ValueError("An initial model can only be labeled smoke")
    weights = directory / "weights.pt"
    if _hash(weights) != manifest["weights_sha256"]:
        raise ValueError("Portable weights checksum mismatch")
    state = torch.load(weights, map_location="cpu", weights_only=True)
    if not state or any(
        not isinstance(v, torch.Tensor)
        or v.dtype != torch.float32
        or not torch.isfinite(v).all()
        for v in state.values()
    ):
        raise ValueError(
            "The original integrator requires finite FP32 portable tensors"
        )
    rename = lambda key: "joint_acts" if key == "joint_f" else key
    model = ModelMixedInput(
        input_sample={
            rename(k): torch.zeros((1, 1, v), dtype=torch.float32)
            for k, v in cfg["input_dims"].items()
        },
        output_dim=cfg["output_dim"],
        input_cfg={"low_dim": [rename(k) for k in cfg["input_dims"]]},
        network_cfg={
            "encoder": {
                "low_dim": {"activation": "relu", "layer_sizes": [], "layernorm": False}
            },
            "model": {
                "mlp": {
                    "activation": "relu",
                    "layer_sizes": cfg["head_hidden_dims"],
                    "layernorm": False,
                }
            },
            "normalize_input": cfg["normalize_input"],
            "normalize_output": cfg["normalize_output"],
            "transformer": {
                k: cfg[k]
                for k in (
                    "n_embd",
                    "n_layer",
                    "n_head",
                    "block_size",
                    "bias",
                    "dropout",
                )
            },
            "output_tanh": cfg["output_tanh"],
        },
        device="cpu",
    )
    torch.nn.Module.to(model, device="cpu", dtype=torch.float32)
    parameters = {
        k: v
        for k, v in state.items()
        if not k.startswith(("input_rms.", "output_rms."))
    }
    model.load_state_dict(parameters, strict=True)
    stats = {}
    expected = set(parameters)
    for key, width in cfg["input_dims"].items():
        rms = RunningMeanStd(shape=(width,), device="cpu")
        for field in ("mean", "var"):
            name = "input_rms." + key + "." + field
            value = state[name]
            if tuple(value.shape) != (width,):
                raise ValueError("Unexpected input RMS shape")
            setattr(rms, field, value.clone())
            expected.add(name)
        stats[rename(key)] = rms
    output = RunningMeanStd(shape=(cfg["output_dim"],), device="cpu")
    for field in ("mean", "var"):
        name = "output_rms." + field
        if tuple(state[name].shape) != (cfg["output_dim"],):
            raise ValueError("Unexpected output RMS shape")
        setattr(output, field, state[name].clone())
        expected.add(name)
    if expected != set(state):
        raise ValueError("Unexpected portable state fields")
    model.set_input_rms(stats)
    model.set_output_rms(output)
    model.eval()
    # Prove restoration didn't silently rename/reshape/drop any parameter value.
    if any(
        not torch.equal(
            model.state_dict()[k].view(torch.int32), value.view(torch.int32)
        )
        for k, value in parameters.items()
    ):
        raise AssertionError("Original model parameter restoration changed bits")
    for key in cfg["input_dims"]:
        for field in ("mean", "var"):
            if not torch.equal(
                getattr(model.input_rms[rename(key)], field).view(torch.int32),
                state["input_rms." + key + "." + field].view(torch.int32),
            ):
                raise AssertionError("Original RMS restoration changed bits")
    identity = {
        "directory": str(directory),
        "manifest_sha256": _hash(path),
        "weights_sha256": _hash(weights),
        "global_step": updates,
        "scope": scope,
        "numerical_profile": "reference",
        "attention_backend": "math",
        "strict_parameters_and_input_rms_bit_equal": True,
        "rms_counts_not_persisted": True,
        "rms_frozen_inference_uses_only_mean_and_variance": True,
        "not_a_default_official_training_reproduction": True,
    }
    return model, manifest, identity


def backend_flags(torch: Any) -> dict[str, bool]:
    return {
        "flash": torch.backends.cuda.flash_sdp_enabled(),
        "math": torch.backends.cuda.math_sdp_enabled(),
        "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
    }
