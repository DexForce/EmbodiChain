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

"""Explicitly trusted, isolated execution of the unmodified official NeRD trainer.

The adapter binds data, model initialization and normalizers to original
VanillaTrainer methods. It never imports external code until explicitly requested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

__all__ = [
    "OFFICIAL_COMMIT",
    "reference_identity",
    "run_reference",
    "export_legacy_dataset",
    "import_legacy_datasets",
    "generate_official_dataset",
    "repair_gravity_transport",
]
OFFICIAL_COMMIT = "370e89d2b8ce68aa3578aa7812859c681b7502bc"
_OFFICIAL_PHYSICS_CONTRACT_VERSION = "official-warp-physical-v2"


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_identity(source: str | Path) -> dict[str, Any]:
    """Verify pinned, unmodified tracked source before importing external code."""
    source = Path(source).resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != OFFICIAL_COMMIT:
        raise ValueError(f"Official trainer requires {OFFICIAL_COMMIT}, found {commit}")
    changed = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if changed:
        raise ValueError("Official reference checkout contains tracked modifications")
    paths = (
        "algorithms/vanilla_trainer.py",
        "integrators/integrator_neural.py",
        "models/models.py",
        "models/model_transformer.py",
        "models/base_models.py",
        "utils/torch_utils.py",
        "utils/running_mean_std.py",
    )
    return {
        "root": str(source),
        "commit": commit,
        "files": {path: _hash(source / path) for path in paths},
        "source_patches": [],
        "adapter": "common-state binding; original preprocess_data_batch, compute_loss, one_epoch and get_scheduled_learning_rate; constructor, logging, data I/O and shared validation evaluation supplied by EmbodiChain",
    }


def _legacy_states(q: Any, qd: Any, codec: Any) -> Any:
    import torch

    # Reversing cross-product operands and subtracting is algebraically equal,
    # but its FP32 fused operations can differ. The codec owns this transport
    # for both trainers, before the unmodified official preprocessing runs.
    return torch.cat((q, codec.native_to_legacy(q, qd)), -1)


def _official_batch(
    batch: dict[str, Any], codec: Any, *, device: str | None = None
) -> dict[str, Any]:
    """Use preserved source twists when present, otherwise transport native qd."""
    import torch

    q = batch["states"]["q"].to(device=device, dtype=torch.float32)
    qd = batch["states"]["qd"].to(device=q.device, dtype=torch.float32)
    transitions = {
        key: value.to(device=q.device, dtype=torch.float32)
        for key, value in batch["transitions"].items()
    }
    legacy_velocity = batch["states"].get("legacy_velocity")
    if legacy_velocity is None:
        states = _legacy_states(q, qd, codec)
    else:
        legacy_velocity = legacy_velocity.to(device=q.device, dtype=torch.float32)
        codec._check(q, qd)
        if legacy_velocity.shape != qd.shape:
            raise ValueError("Preserved legacy velocity shape must match native qd")
        states = torch.cat((q, legacy_velocity), -1)
    shape = q.shape[:2]
    shape = (shape[0], shape[1] - 1)
    root = batch["states"].get("root_body_q")
    if root is not None:
        root = root.to(device=q.device, dtype=torch.float32)
    if root is None:
        root = (
            q[..., :7]
            if codec.robot == "ant"
            else q.new_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).expand(
                *q.shape[:2], 7
            )
        )
    result = {
        "states": states[:, :-1],
        "next_states": states[:, 1:],
        "root_body_q": root[:, :-1].float(),
        "joint_acts": (
            transitions["joint_f"][..., -8:]
            if codec.robot == "ant"
            else transitions["joint_f"]
        ),
        "gravity_dir": transitions.get(
            "gravity_dir", q.new_tensor([0.0, 0.0, -1.0]).expand(*shape, 3)
        ),
    }
    slots = transitions.get("contact_depths", q.new_zeros(*shape, 0)).shape[-1]
    # Main's preprocessing reshapes contact tensors with an inferred final axis,
    # which cannot represent zero slots. A masked dummy is unobserved by the
    # contact-free Cartpole network and does not alter its input or loss.
    if not slots:
        if codec.robot != "cartpole":
            raise ValueError("Ant requires contact slots")
        slots = 1
        result.update(
            contact_depths=q.new_full((*shape, 1), 1e4),
            contact_thicknesses=q.new_zeros(*shape, 1),
            contact_normals=q.new_zeros(*shape, 3),
            contact_points_0=q.new_zeros(*shape, 3),
            contact_points_1=q.new_zeros(*shape, 3),
        )
    else:
        for name in (
            "contact_depths",
            "contact_normals",
            "contact_points_0",
            "contact_points_1",
        ):
            width = slots if name == "contact_depths" else slots * 3
            result[name] = transitions.get(name, q.new_zeros(*shape, width))
        result["contact_thicknesses"] = transitions.get(
            "contact_thicknesses_0", q.new_zeros(*shape, slots)
        ) + transitions.get("contact_thicknesses_1", q.new_zeros(*shape, slots))
    return result


def _reference_integrator(robot: str, slots: int, device: str) -> Any:
    import torch
    import warp as wp
    from integrators.integrator_neural import NeuralIntegrator

    value = NeuralIntegrator.__new__(NeuralIntegrator)
    ant = robot == "ant"
    value.torch_device = device
    value.dof_q_per_env = 15 if ant else 2
    value.dof_qd_per_env = 14 if ant else 2
    value.state_dim = value.prediction_dim = value.dof_q_per_env + value.dof_qd_per_env
    value.num_contacts_per_env = slots
    value.joint_types = (
        [wp.sim.JOINT_FREE] + [wp.sim.JOINT_REVOLUTE] * 8
        if ant
        else [wp.sim.JOINT_PRISMATIC, wp.sim.JOINT_REVOLUTE]
    )
    value.num_joints_per_env = len(value.joint_types)
    value.joint_q_start = [0] + list(range(7, 15)) if ant else [0, 1]
    value.joint_q_end = [7] + list(range(8, 16)) if ant else [1, 2]
    value.is_continuous_dof = torch.zeros(
        value.state_dim, dtype=torch.bool, device=device
    )
    if not ant:
        value.is_continuous_dof[1] = True
    value.states_frame = "body" if ant else "world"
    value.anchor_frame_step = "every"
    value.states_embedding_type = "identical"
    value.orientation_prediction_parameterization = "quaternion"
    value.prediction_type = "relative"
    return value


def _restore_official_model(
    common: Any,
    *,
    device: str,
    input_sample: dict[str, Any] | None = None,
) -> Any:
    """Reconstruct the original network from a portable model without aliasing it.

    The caller must first establish its explicit external-code trust boundary,
    verify the pinned checkout and expose its modules on the isolated subprocess
    import path. This helper performs no source discovery or simulator startup.
    It preserves all parameters, normalization values, ordered features and dtype,
    returns evaluation mode, and renames only joint_f to legacy joint_acts.
    Physics callers must enforce their own supported inference dtype (the old
    Warp integrator uses FP32); standalone network restoration also supports FP64.
    """
    import torch
    from models.models import ModelMixedInput
    from utils.running_mean_std import RunningMeanStd

    c = common.cfg
    dtype = next(common.parameters()).dtype
    rename = lambda key: "joint_acts" if key == "joint_f" else key
    if input_sample is None:
        input_sample = {
            rename(key): torch.zeros((1, 1, width), dtype=dtype, device=device)
            for key, width in c.input_dims.items()
        }
    model = ModelMixedInput(
        input_sample=input_sample,
        output_dim=c.output_dim,
        input_cfg={"low_dim": [rename(key) for key in c.input_dims]},
        network_cfg={
            "encoder": {
                "low_dim": {"activation": "relu", "layer_sizes": [], "layernorm": False}
            },
            "model": {
                "mlp": {
                    "activation": "relu",
                    "layer_sizes": list(c.head_hidden_dims),
                    "layernorm": False,
                }
            },
            "normalize_input": c.normalize_input,
            "normalize_output": c.normalize_output,
            "transformer": {
                "n_embd": c.n_embd,
                "n_layer": c.n_layer,
                "n_head": c.n_head,
                "block_size": c.block_size,
                "bias": c.bias,
                "dropout": c.dropout,
            },
            "output_tanh": c.output_tanh,
        },
        device=device,
    )
    shared = {
        key: value
        for key, value in common.state_dict().items()
        if not key.startswith(("input_rms.", "output_rms."))
    }
    # The original ModelMixedInput.to only visits some children; call the base
    # implementation so transformer parameters follow the requested dtype/device.
    torch.nn.Module.to(model, device=device, dtype=dtype)
    model.load_state_dict(shared, strict=True)
    stats = {}
    for key, value in common.input_rms.items():
        rms = RunningMeanStd(shape=value.mean.shape, device=device)
        rms.mean = value.mean.detach().to(device=device, dtype=dtype).clone()
        rms.var = value.var.detach().to(device=device, dtype=dtype).clone()
        stats[rename(key)] = rms
    output = RunningMeanStd(shape=common.output_rms.mean.shape, device=device)
    output.mean = common.output_rms.mean.detach().to(device=device, dtype=dtype).clone()
    output.var = common.output_rms.var.detach().to(device=device, dtype=dtype).clone()
    model.set_input_rms(stats)
    model.set_output_rms(output)
    return model.eval()


def run_reference(
    spec: dict[str, Any],
    *,
    trust_external_code: bool = False,
    resume: str | Path | None = None,
    stop_after: int | None = None,
) -> dict[str, Any]:
    """Run actual official trainer methods against matched immutable artifacts.

    Invoke in an isolated subprocess with legacy Warp dependencies on PYTHONPATH.
    This binds an original trainer instance without its constructor's eager
    dataset/GPU allocation; no original method is monkeypatched or replaced.
    """
    if not trust_external_code:
        raise ValueError("Official source execution requires trust_external_code=True")
    from ._numerics import _validate_numerical_profile

    if (
        _validate_numerical_profile(
            spec.get("train", {}).get("numerical_profile", "default")
        )
        != "default"
    ):
        raise ValueError(
            "Original-method reference adapter supports only default numerical_profile"
        )
    identity = reference_identity(spec["source"])
    sys.path.insert(0, identity["root"])
    import torch
    import numpy as np
    from algorithms.vanilla_trainer import VanillaTrainer
    from utils.time_report import TimeReport
    from embodichain.compute.dynamics import NeRDStateCodec
    from embodichain.data_pipeline.dynamics.shards import DynamicsWindowDataset
    from embodichain.learning.models.nerd import load_model, export_model
    from .trainer import (
        DynamicsTrainCfg,
        evaluate_model,
        _atomic_save,
        _capture_rng_state,
        _restore_rng_state,
        _PREPROCESSING_EXECUTION,
    )
    from ._runtime import _training_runtime_identity

    cfg = DynamicsTrainCfg(**spec["train"])
    torch.set_float32_matmul_precision("highest")
    device = cfg.device
    torch.set_num_threads(int(spec.get("cpu_threads", 1)))
    common, manifest = load_model(spec["initial_model"], device=device)
    if common.cfg.attention_backend != "default":
        raise ValueError(
            "Original-method reference adapter does not support math attention initial_model"
        )
    if (
        spec.get("initial_model_sha256")
        and _hash(Path(spec["initial_model"]) / "weights.pt")
        != spec["initial_model_sha256"]
    ):
        raise ValueError("Common initialization changed")
    codec = NeRDStateCodec(**manifest["codec"])
    training = DynamicsWindowDataset(
        spec["dataset"], "train", common.cfg.history_length, verify=True
    )
    validation = DynamicsWindowDataset(
        spec["dataset"], "validation", common.cfg.history_length
    )
    test = DynamicsWindowDataset(spec["dataset"], "test", common.cfg.history_length)
    if training.fingerprint != spec["dataset_fingerprint"]:
        raise ValueError("Reference dataset fingerprint mismatch")
    order_path = Path(spec["sample_order"])
    if _hash(order_path) != spec["sample_order_sha256"]:
        raise ValueError("Reference sample order checksum mismatch")
    from ._sample_order import _open_order

    order = _open_order(order_path)
    if order.shape != (cfg.total_steps, cfg.batch_size):
        raise ValueError("Reference sample-order shape mismatch")
    contract = {
        "dataset_fingerprint": training.fingerprint,
        "sample_order_sha256": spec["sample_order_sha256"],
        "initial_model_sha256": _hash(Path(spec["initial_model"]) / "weights.pt"),
        "train_cfg": cfg.to_dict(),
        "source_commit": identity["commit"],
        "source_files": identity["files"],
        "torch_version": str(torch.__version__),
        "adapter_sha256": _hash(Path(__file__)),
        "preprocessing_execution": _PREPROCESSING_EXECUTION,
    }
    restored = None
    start_step = 0
    if resume is not None:
        restored = torch.load(resume, map_location="cpu", weights_only=True)
        if (
            restored.get("format") != "embodichain.nerd.reference_training.v1"
            or restored.get("contract") != contract
        ):
            raise ValueError("Reference resume contract mismatch")
        common.load_state_dict(restored["model"])
        start_step = restored["global_step"]
    end_step = (
        cfg.total_steps if stop_after is None else min(stop_after, cfg.total_steps)
    )
    if type(end_step) is not int or not start_step <= end_step <= cfg.total_steps:
        raise ValueError("Invalid reference training stop cursor")
    sample = training.get_batch([0])
    raw = _official_batch(sample, codec, device=device)
    integrator = _reference_integrator(
        codec.robot, raw["contact_depths"].shape[-1], device
    )
    original = VanillaTrainer.__new__(VanillaTrainer)
    original.device = device
    original.neural_env = SimpleNamespace(frame_dt=manifest["dt"])
    original.neural_integrator = integrator
    processed = original.preprocess_data_batch({k: v.clone() for k, v in raw.items()})
    original.neural_model = _restore_official_model(
        common, device=device, input_sample=processed
    )
    integrator.neural_model = original.neural_model
    original.batch_size = cfg.batch_size
    original.truncate_grad = True
    original.grad_norm = cfg.max_grad_norm
    original.lr_start = cfg.learning_rate
    original.lr_end = cfg.final_learning_rate
    original.lr_schedule = "linear"
    original.optimizer = torch.optim.Adam(
        original.neural_model.parameters(), lr=cfg.learning_rate
    )
    if restored is not None:
        original.optimizer.load_state_dict(restored["optimizer"])
    original.time_report = TimeReport(cuda_synchronize=False)
    original.time_report.add_timers(["dataloader", "compute_loss", "backward", "other"])
    # Initialization and transport above may consume RNG; both runs begin their
    # optimizer updates with the configured seed, as DynamicsTrainer does.
    torch.manual_seed(cfg.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(cfg.seed)
    if cfg.resident_cache_bytes:
        used = training.enable_device_cache(device, cfg.resident_cache_bytes)
        validation.enable_device_cache(device, cfg.resident_cache_bytes - used)
    if torch.device(device).type == "cuda":
        warm_batch = training.get_batch([0] * cfg.batch_size)
        with torch.enable_grad():
            for _ in range(3):
                original.preprocess_data_batch(
                    _official_batch(warm_batch, codec, device=device)
                )
        del warm_batch
    destination = Path(spec["output_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    best_weights = None
    if restored is not None:
        best, best_weights = restored["best_validation"], restored["best_weights"]
        _restore_rng_state(restored["rng"])

    def transfer() -> None:
        current = common.state_dict()
        current.update(original.neural_model.state_dict())
        common.load_state_dict(current)

    def checkpoint(step: int) -> None:
        transfer()
        _atomic_save(
            destination / "reference-last.pt",
            {
                "format": "embodichain.nerd.reference_training.v1",
                "contract": contract,
                "model": common.state_dict(),
                "optimizer": original.optimizer.state_dict(),
                "global_step": step,
                "best_validation": best,
                "best_weights": best_weights,
                "rng": _capture_rng_state(device),
            },
        )

    try:
        log_path = destination / "metrics.jsonl"
        if log_path.exists():
            if resume is None:
                raise FileExistsError(
                    "Reference output already has progress; use explicit resume"
                )
            pending_log = log_path.with_suffix(".jsonl.pending")
            with log_path.open() as old, pending_log.open("w") as retained:
                for line in old:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        break
                    if record["step"] <= start_step:
                        retained.write(line)
            os.replace(pending_log, log_path)
        with log_path.open("a") as log:
            for step in range(start_step, end_step):
                if np.any(order[step] < 0) or np.any(order[step] >= len(training)):
                    raise ValueError("Reference sample order index out of bounds")
                batch = training.get_batch(order[step])
                transported = _official_batch(batch, codec, device=device)
                rate = original.get_scheduled_learning_rate(
                    step // cfg.lr_update_interval + cfg.lr_offset,
                    cfg.total_steps / cfg.lr_update_interval + cfg.lr_offset,
                )
                for group in original.optimizer.param_groups:
                    group["lr"] = rate
                loader = [transported]
                loss, _, grad = original.one_epoch(True, loader, iter(loader), 1)
                record = {
                    "step": step + 1,
                    "loss": loss,
                    "learning_rate": rate,
                    "grad_norm": float(grad["grad_norm_before_clip"]),
                }
                if (
                    step + 1
                ) % cfg.validation_interval == 0 or step + 1 == cfg.total_steps:
                    transfer()
                    metrics = evaluate_model(
                        common,
                        codec,
                        validation,
                        cfg.batch_size,
                        device,
                        max_batches=cfg.validation_batches,
                    )
                    record["validation"] = metrics
                    if metrics["normalized_mse"] < best:
                        best = metrics["normalized_mse"]
                        best_weights = {
                            key: value.detach().cpu().clone()
                            for key, value in common.state_dict().items()
                        }
                if (
                    (step + 1) % cfg.log_interval == 0
                    or step + 1 == end_step
                    or "validation" in record
                ):
                    log.write(json.dumps(record, allow_nan=False) + "\n")
                    log.flush()
                if (step + 1) % cfg.checkpoint_interval == 0:
                    checkpoint(step + 1)
        checkpoint(end_step)
        if end_step != cfg.total_steps:
            result = {
                "status": "interrupted",
                "completed_updates": end_step,
                "budget": cfg.total_steps,
                "checkpoint": str(destination / "reference-last.pt"),
            }
            (destination / "result.json").write_text(json.dumps(result, indent=2))
            return result
        transfer()
        export_model(destination / "last-export", common, manifest)
        common.load_state_dict(best_weights)
        export_model(destination / "best-export", common, manifest)
        metrics = evaluate_model(common, codec, test, cfg.batch_size, device)
        result = {
            "status": "completed",
            "completed_updates": cfg.total_steps,
            "budget": cfg.total_steps,
            "best_validation": best,
            "test": metrics,
            "reference": identity,
            "dataset_fingerprint": training.fingerprint,
            "sample_order_sha256": spec["sample_order_sha256"],
            "seed": cfg.seed,
            "qualification": "matched_training_run_only",
            "torch_version": str(torch.__version__),
            "runtime_identity": _training_runtime_identity(original.neural_model),
            "device": device,
            "adapter_source_sha256": _hash(Path(__file__)),
            "best_model_sha256": _hash(destination / "best-export" / "weights.pt"),
            "last_model_sha256": _hash(destination / "last-export" / "weights.pt"),
            "training_methods": [
                "VanillaTrainer.preprocess_data_batch",
                "VanillaTrainer.compute_loss",
                "VanillaTrainer.one_epoch",
                "VanillaTrainer.get_scheduled_learning_rate",
            ],
        }
        (destination / "result.json").write_text(
            json.dumps(result, indent=2, allow_nan=False)
        )
        return result
    finally:
        training.close()
        validation.close()
        test.close()


def export_legacy_dataset(
    root: str | Path, output: str | Path, split: str
) -> dict[str, Any]:
    """Stream canonical trajectories to official T,B HDF5 transport arrays.

    Variable lengths are retained via traj_lengths; padding is never sampled by
    the official TrajectoryDataset. No source dataset or split is modified.
    Preserved source velocities bypass native conversion when available.
    """
    import h5py
    import numpy as np
    import torch
    from embodichain.compute.dynamics import NeRDStateCodec
    from embodichain.data_pipeline.dynamics.shards import (
        load_manifest,
        dataset_fingerprint,
    )

    root = Path(root)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifest = load_manifest(root, verify=True)
    codec = NeRDStateCodec(**manifest["metadata"]["codec"])
    rows = [
        (shard["file"], row)
        for shard in manifest["shards"]
        for row in shard["trajectories"]
        if row["split"] == split
    ]
    if not rows:
        raise ValueError("No trajectories for requested split")
    max_length = max(row["length"] for _, row in rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".pending")
    with h5py.File(temporary, "w") as target:
        group = target.create_group("data")
        group.attrs["mode"] = "trajectory"
        group.attrs["total_transitions"] = sum(row["length"] for _, row in rows)
        group.create_dataset(
            "traj_lengths",
            data=np.array([row["length"] for _, row in rows], dtype=np.int32),
        )
        for index, (name, row) in enumerate(rows):
            with h5py.File(root / name, "r") as source:
                trajectory = source[row["group"]]
                batch = {
                    kind: {
                        key: torch.from_numpy(value[()]).unsqueeze(0)
                        for key, value in trajectory[kind].items()
                    }
                    for kind in ("states", "transitions")
                }
            raw = _official_batch(batch, codec)
            for key, value in raw.items():
                array = value[0].numpy()
                if key not in group:
                    group.create_dataset(
                        key,
                        shape=(max_length, len(rows), array.shape[-1]),
                        dtype=np.float32,
                        chunks=(min(max_length, 128), 1, array.shape[-1]),
                        compression="lzf",
                        fillvalue=0.0,
                    )
                group[key][: row["length"], index] = array
    os.replace(temporary, output)
    result = {
        "dataset_fingerprint": dataset_fingerprint(manifest),
        "split": split,
        "trajectories": len(rows),
        "path": str(output.resolve()),
        "sha256": _hash(output),
        "layout": "T,B,features",
        "native_to_legacy_qd": "legacy_velocity" not in manifest["fields"]["states"],
        "cartpole_dummy_masked_contact": codec.robot == "cartpole",
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(result, indent=2))
    return result


def import_legacy_datasets(
    sources: dict[str, str | Path],
    output: str | Path,
    metadata: dict[str, Any],
    *,
    trajectories_per_shard: int = 64,
    world_rotation_xyzw: tuple[float, float, float, float] | None = None,
    gravity_direction: tuple[float, float, float] | None = None,
    preserve_legacy_velocity: bool = False,
) -> dict[str, Any]:
    """Import official world-frame HDF5 trajectories into canonical shards.

    Source files explicitly name their split; the importer never re-splits
    official validation/test trajectories. Only one trajectory is staged at a
    time. Adjacent states must be continuous within each declared valid length.
    Legacy combined contact thickness is stored in side zero, side one zero;
    the combined feature/mask is preserved without inventing separate margins.

    Args:
        sources: Original HDF5 paths keyed by explicit dataset split.
        output: New canonical shard directory.
        metadata: Codec and dataset provenance, copied without modifying it.
        trajectories_per_shard: Maximum trajectories per committed shard.
        world_rotation_xyzw: Optional unit quaternion rotating world coordinates.
        gravity_direction: Optional explicit unit gravity transport correction.
        preserve_legacy_velocity: Also retain original FP32 velocities in
            ``states/legacy_velocity`` before COM conversion. This requires
            unrotated source-world coordinates and bitwise continuous original
            states, including signed zero. Native ``q``/``qd`` remain available
            to normal runtime consumers; the original input file is read only.

    Returns:
        Imported dataset identity, trajectory count and immutable fingerprint.
    """
    import h5py
    import numpy as np
    import torch
    from embodichain.compute.dynamics import NeRDStateCodec
    from embodichain.data_pipeline.dynamics.shards import (
        DynamicsShardWriter,
        load_manifest,
        dataset_fingerprint,
    )

    if not sources or set(sources) - {"train", "validation", "test"}:
        raise ValueError(
            "Legacy sources must explicitly name train/validation/test splits"
        )
    if preserve_legacy_velocity and world_rotation_xyzw is not None:
        raise ValueError(
            "Preserved legacy velocity requires the source world frame without rotation"
        )
    codec = NeRDStateCodec(**metadata["codec"])
    from embodichain.compute.dynamics import quat_rotate, quat_multiply

    rotation = (
        None
        if world_rotation_xyzw is None
        else torch.tensor(world_rotation_xyzw, dtype=torch.float32)
    )
    if rotation is not None and (
        rotation.shape != (4,)
        or not torch.isfinite(rotation).all()
        or not torch.isclose(rotation.norm(), torch.tensor(1.0), atol=1e-6)
    ):
        raise ValueError("World rotation must be a unit xyzw quaternion")
    source_identity = {
        split: {"path": str(Path(path).resolve()), "sha256": _hash(Path(path))}
        for split, path in sources.items()
    }
    manifest_metadata = {
        **metadata,
        "legacy_sources": source_identity,
        "legacy_thickness_transport": "combined_in_side_zero",
        "world_rotation_xyzw": None if rotation is None else rotation.tolist(),
        "gravity_transport_override": gravity_direction,
    }
    if preserve_legacy_velocity:
        manifest_metadata["legacy_velocity"] = {
            "version": 1,
            "field": "states/legacy_velocity",
            "convention": "official_nerd_raw_velocity",
            "layout": (
                "angular_first_world_origin_spatial"
                if codec.robot == "ant"
                else "joint_velocity"
            ),
            "world_frame": "source",
            "source_field": "data/states[..., q_dim:]",
            "terminal_source_field": "data/next_states[valid_length-1, ..., q_dim:]",
            "dtype": "float32",
        }
    else:
        # This importer owns the field declaration. Reusing a prior preserved
        # manifest cannot label a native-only import as lossless source data.
        manifest_metadata.pop("legacy_velocity", None)
    total = 0
    with DynamicsShardWriter(
        output, manifest_metadata, trajectories_per_shard=trajectories_per_shard
    ) as writer:
        for split, path in sources.items():
            with h5py.File(path, "r") as file:
                data = file["data"]
                if data.attrs.get("mode") != "trajectory":
                    raise ValueError("Expected official trajectory HDF5 layout")
                if (
                    data["states"].ndim != 3
                    or data["states"].shape[-1] != codec.q_dim + codec.qd_dim
                ):
                    raise ValueError("Legacy state layout does not match the codec")
                horizon, trajectories, _ = data["states"].shape
                lengths = (
                    data["traj_lengths"][()]
                    if "traj_lengths" in data
                    else np.full(trajectories, horizon)
                )
                for index, valid in enumerate(lengths):
                    length = int(valid)
                    if not 1 <= length <= horizon:
                        raise ValueError("Invalid official trajectory valid length")
                    current = data["states"][:length, index].astype(np.float32)
                    following = data["next_states"][:length, index].astype(np.float32)
                    if length > 1:
                        if preserve_legacy_velocity:
                            if not np.array_equal(
                                current[1:].view(np.uint32),
                                following[:-1].view(np.uint32),
                            ):
                                raise ValueError(
                                    "Preserved legacy velocity requires bitwise continuous "
                                    "FP32 states, including signed zero"
                                )
                        elif not np.allclose(
                            current[1:], following[:-1], atol=1e-6, rtol=1e-5
                        ):
                            raise ValueError(
                                "Legacy trajectory crosses an undeclared reset/discontinuity"
                            )
                    legacy = torch.from_numpy(
                        np.concatenate((current, following[-1:]), axis=0)
                    )
                    q = legacy[:, : codec.q_dim]
                    qd = codec.legacy_to_native(q, legacy[:, codec.q_dim :])
                    if rotation is not None and codec.robot == "ant":
                        q = torch.cat(
                            (
                                quat_rotate(rotation, q[:, :3]),
                                quat_multiply(rotation, q[:, 3:7]),
                                q[:, 7:],
                            ),
                            -1,
                        )
                        qd = torch.cat(
                            (
                                quat_rotate(rotation, qd[:, :3]),
                                quat_rotate(rotation, qd[:, 3:6]),
                                qd[:, 6:],
                            ),
                            -1,
                        )
                    states = {"q": q.numpy(), "qd": qd.numpy()}
                    if preserve_legacy_velocity:
                        states["legacy_velocity"] = legacy[:, codec.q_dim :].numpy()
                    if "root_body_q" in data:
                        root = data["root_body_q"][:length, index].astype(np.float32)
                        if rotation is not None:
                            root_tensor = torch.from_numpy(root)
                            root = torch.cat(
                                (
                                    quat_rotate(rotation, root_tensor[:, :3]),
                                    quat_multiply(rotation, root_tensor[:, 3:7]),
                                ),
                                -1,
                            ).numpy()
                        last = q[-1:, :7].numpy() if codec.robot == "ant" else root[-1:]
                        states["root_body_q"] = np.concatenate((root, last), axis=0)
                    effort = data["joint_acts"][:length, index].astype(np.float32)
                    if codec.robot == "ant" and effort.shape[-1] == 8:
                        effort = np.concatenate(
                            (np.zeros((length, 6), np.float32), effort), axis=-1
                        )
                    transitions = {"joint_f": effort}
                    if "gravity_dir" in data:
                        transitions["gravity_dir"] = data["gravity_dir"][
                            :length, index
                        ].astype(np.float32)
                    for name in (
                        "contact_depths",
                        "contact_normals",
                        "contact_points_0",
                        "contact_points_1",
                    ):
                        if name in data:
                            transitions[name] = data[name][:length, index].astype(
                                np.float32
                            )
                    if "contact_thicknesses" in data:
                        thickness = data["contact_thicknesses"][:length, index].astype(
                            np.float32
                        )
                        transitions["contact_thicknesses_0"] = thickness
                        transitions["contact_thicknesses_1"] = np.zeros_like(thickness)
                    if gravity_direction is not None:
                        direction = np.asarray(gravity_direction, dtype=np.float32)
                        if (
                            direction.shape != (3,)
                            or not np.isfinite(direction).all()
                            or not np.isclose(np.linalg.norm(direction), 1.0, atol=1e-6)
                        ):
                            raise ValueError(
                                "Gravity transport direction must be a finite unit vector"
                            )
                        transitions["gravity_dir"] = np.broadcast_to(
                            direction, (length, 3)
                        ).copy()
                    if rotation is not None:
                        for field in (
                            "gravity_dir",
                            "contact_normals",
                            "contact_points_1",
                        ):
                            if field in transitions:
                                array = transitions[field]
                                transitions[field] = (
                                    quat_rotate(
                                        rotation, torch.from_numpy(array.reshape(-1, 3))
                                    )
                                    .reshape(array.shape)
                                    .numpy()
                                )
                    writer.add_trajectory(
                        f"{split}:{index}",
                        states,
                        transitions,
                        split=split,
                        metadata={
                            "legacy_source_sha256": source_identity[split]["sha256"],
                            "legacy_index": index,
                        },
                    )
                    total += 1
    manifest = load_manifest(output, verify=True)
    return {
        "path": str(Path(output).resolve()),
        "trajectories": total,
        "fingerprint": dataset_fingerprint(manifest),
        "sources": source_identity,
    }


def repair_gravity_transport(
    gravity: Any, direction: tuple[float, float, float], *, enabled: bool = False
) -> tuple[Any, dict[str, Any]]:
    """Explicitly repair upstream uninitialized gravity components in a copy.

    The original sampler writes only its up-axis into ``torch.empty``. Preserve
    raw returned arrays separately; only this declared field is repaired.
    """
    import numpy as np

    values = np.asarray(gravity)
    expected = np.asarray(direction, dtype=values.dtype)
    if (
        values.shape[-1] != 3
        or expected.shape != (3,)
        or not np.isfinite(expected).all()
        or not np.isclose(np.linalg.norm(expected), 1.0, atol=1e-6)
    ):
        raise ValueError("Invalid gravity transport shape/direction")
    mismatch = ~np.isfinite(values) | ~np.isclose(values, expected, atol=1e-7, rtol=0.0)
    evidence = {
        "enabled": enabled,
        "changed_components": int(mismatch.sum()),
        "nonfinite_components": int((~np.isfinite(values)).sum()),
        "physical_direction": expected.tolist(),
        "source_issue": "TrajectorySampler allocates torch.empty gravity_dir and writes only up-axis",
    }
    if mismatch.any() and not enabled:
        raise ValueError(
            "Official sampler returned uninitialized gravity values; explicitly enable gravity repair and retain raw artifacts"
        )
    return (
        np.broadcast_to(expected, values.shape).copy() if enabled else values.copy()
    ), evidence


def generate_official_dataset(
    spec: dict[str, Any], *, trust_external_code: bool = False
) -> dict[str, Any]:
    """Collect real original Warp trajectories and retain raw HDF5 evidence.

    Uses untouched original NeuralEnvironment, TrajectorySampler[Ant], commons,
    sampling, collision and integration. Gravity repair and global frame rotation
    are explicit transport operations, recorded separately from raw artifacts.
    Use a fresh subprocess with the requested isolated Warp version on PYTHONPATH.
    """
    if not trust_external_code:
        raise ValueError("Official generation requires trust_external_code=True")
    identity = reference_identity(spec["source"])
    sys.path.insert(0, identity["root"])
    import h5py
    import numpy as np
    import torch
    import warp as wp
    from envs.neural_environment import NeuralEnvironment
    from generate.trajectory_sampler import TrajectorySampler
    from generate.trajectory_sampler_ant import TrajectorySamplerAnt
    from utils.commons import (
        JOINT_Q_MIN,
        JOINT_Q_MAX,
        JOINT_QD_MIN,
        JOINT_QD_MAX,
        JOINT_ACT_SCALE,
    )
    from utils.python_utils import set_random_seed
    import math
    import time
    import xml.etree.ElementTree as ET
    from embodichain.compute.dynamics import quat_multiply

    robot = str(spec["robot"]).lower()
    if robot not in {"cartpole", "ant"}:
        raise ValueError("Official generation supports cartpole and ant")
    expected_warp = spec.get("warp_version", "1.5.1" if robot == "ant" else "1.8.0")
    if wp.__version__ != expected_warp:
        raise ValueError(
            f"Expected isolated Warp {expected_warp}, loaded {wp.__version__}"
        )
    device = spec.get("device", "cpu")
    torch.set_num_threads(int(spec.get("cpu_threads", 1)))
    batch = int(spec.get("num_envs", 8))
    horizon = int(spec.get("horizon", 100))
    splits = spec.get("splits", {"train": 32, "validation": 8, "test": 8})
    if (
        min(batch, horizon) < 1
        or set(splits) != {"train", "validation", "test"}
        or any(int(v) < 1 for v in splits.values())
    ):
        raise ValueError(
            "Positive batch/horizon and all three explicit split counts are required"
        )
    output = Path(spec["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Official generation output must be empty")
    set_random_seed(int(spec.get("seed", 0)))
    started = time.perf_counter()
    environment = NeuralEnvironment(
        env_name="Ant" if robot == "ant" else "Cartpole",
        num_envs=batch,
        neural_integrator_cfg={},
        neural_model=None,
        default_env_mode="ground-truth",
        warp_env_cfg={"seed": int(spec.get("seed", 0))},
        device=device,
        render=False,
    )
    name = environment.robot_name
    sampler_type = TrajectorySamplerAnt if robot == "ant" else TrajectorySampler
    sampler = sampler_type(
        environment,
        JOINT_Q_MIN[name],
        JOINT_Q_MAX[name],
        JOINT_QD_MIN[name],
        JOINT_QD_MAX[name],
        JOINT_ACT_SCALE[name],
        contact_prob=0.0,
    )
    source_dir = Path(identity["root"]) / "envs" / "warp_sim_envs" / "assets"
    asset = source_dir / ("ant.xml" if robot == "ant" else "cartpole_single.urdf")
    scalar_names = [
        joint.attrib["name"]
        for joint in ET.parse(asset).getroot().iter("joint")
        if "name" in joint.attrib and joint.attrib.get("type") != "free"
    ]
    model = environment.model
    nbody = environment.env.bodies_per_env
    base_com = model.body_com.numpy()[0].tolist()
    direction = [0.0, 0.0, 0.0]
    direction[int(model.up_axis)] = -1.0
    physical = {}
    for field in (
        "body_mass",
        "body_inertia",
        "body_com",
        "joint_armature",
        "joint_limit_lower",
        "joint_limit_upper",
        "joint_limit_ke",
        "joint_limit_kd",
        "joint_axis_mode",
        "joint_target_ke",
        "joint_target_kd",
    ):
        value = getattr(model, field, None)
        if value is not None:
            array = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
            physical[field] = array.tolist()
    for field in ("mu", "ke", "kd", "kf"):
        value = getattr(model.shape_materials, field, None)
        if value is not None:
            physical["shape_material_" + field] = value.numpy().tolist()
    physical["shape_thickness"] = model.shape_geo.thickness.numpy().tolist()
    physical["shape_body"] = model.shape_body.numpy().tolist()
    physical["ground_plane"] = model.ground_plane.numpy().tolist()
    gravity_vector = np.asarray(model.gravity, dtype=np.float64).reshape(3)
    if not np.isfinite(gravity_vector).all():
        raise ValueError("Official teacher gravity must be finite")
    physical["gravity"] = gravity_vector.tolist()
    generator_paths = (
        "generate/trajectory_sampler.py",
        "generate/simulation_sampler.py",
        "utils/commons.py",
        "utils/env_utils.py",
        "utils/warp_utils.py",
        "envs/neural_environment.py",
        "envs/abstract_contact.py",
        "envs/abstract_contact_environment.py",
        "envs/warp_sim_envs/environment.py",
        f"envs/warp_sim_envs/env_{robot}.py",
    )
    if robot == "ant":
        generator_paths += ("generate/trajectory_sampler_ant.py",)
    generator_files = {
        name: _hash(Path(identity["root"]) / name) for name in generator_paths
    }
    generator_hash = hashlib.sha256(
        json.dumps(generator_files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    metadata = {
        "robot": robot,
        "dt": environment.frame_dt,
        "codec": {
            "robot": robot,
            "base_com": base_com,
            "continuous_q_indices": [1] if robot == "cartpole" else [],
        },
        "asset_sha256": _hash(asset),
        "joint_names": scalar_names,
        "q_dim": environment.dof_q_per_env,
        "qd_dim": environment.dof_qd_per_env,
        "teacher_origin": {
            "kind": "official_warp",
            "source_commit": OFFICIAL_COMMIT,
            "warp_version": str(wp.__version__),
            "integrator": str(environment.env.integrator_type),
            "num_substeps": environment.env.sim_substeps,
            "up_axis": int(model.up_axis),
            "gravity": gravity_vector.tolist(),
            "gravity_magnitude": float(np.linalg.norm(gravity_vector)),
            "physics_contract_version": _OFFICIAL_PHYSICS_CONTRACT_VERSION,
            "bodies_per_env": nbody,
            "device": device,
        },
        "physics_contract_version": _OFFICIAL_PHYSICS_CONTRACT_VERSION,
        "source_physical_arrays": physical,
        "source_generator_files": generator_files,
        "source_generator_sha256": generator_hash,
        "generation_adapter_sha256": _hash(Path(__file__)),
        "source": identity,
        "sampling": {
            "seed": spec.get("seed", 0),
            "initial_states_source": spec.get("initial_states_source", "sample"),
            "passive": bool(spec.get("passive", False)),
            "batch_size": batch,
            "horizon": horizon,
        },
    }
    rotation = None
    if spec.get("canonical_world", "source") == "z_up":
        rotation_tensor = torch.tensor([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])
        if robot == "cartpole":
            rotation_tensor = quat_multiply(
                torch.tensor([0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]),
                rotation_tensor,
            )
        rotation = tuple(rotation_tensor.tolist())
    elif spec.get("canonical_world", "source") != "source":
        raise ValueError("canonical_world must be source or z_up")
    sources = {}
    corrections = {}
    source_counts = {}
    for split_index, (split, count) in enumerate(splits.items()):
        set_random_seed(int(spec.get("seed", 0)) + split_index)
        path = output / f"{split}.raw.h5"
        written = 0
        correction = None
        with h5py.File(path, "w") as target:
            data = target.create_group("data")
            data.attrs["mode"] = "trajectory"
            while written < int(count):
                # Invoke the unchanged sampler for at most one environment batch
                # and stream it immediately. Its numerical-invalid retry remains
                # upstream behavior; no entire split is retained on CPU or GPU.
                requested = min(int(count) - written, batch)
                rollouts = sampler.sample_trajectories_joint_act_mode(
                    requested * horizon,
                    horizon,
                    passive=bool(spec.get("passive", False)),
                    initial_states_source=spec.get("initial_states_source", "sample"),
                    render=False,
                )
                arrays = {
                    key: value[:, :requested].detach().cpu().numpy()
                    for key, value in rollouts.items()
                    if not isinstance(value, dict)
                }
                arrays.update(
                    {
                        key: value[:, :requested].detach().cpu().numpy()
                        for key, value in rollouts["contacts"].items()
                    }
                )
                produced = arrays["states"].shape[1]
                for key, value in arrays.items():
                    value = value.reshape(*value.shape[:2], -1)
                    if key not in data:
                        data.create_dataset(
                            key,
                            shape=(horizon, 0, value.shape[-1]),
                            maxshape=(horizon, None, value.shape[-1]),
                            chunks=(min(horizon, 128), 1, value.shape[-1]),
                            dtype=value.dtype,
                            compression="lzf",
                        )
                    data[key].resize(written + produced, axis=1)
                    data[key][:, written : written + produced] = value
                written += produced
                data.attrs["total_transitions"] = horizon * written
                data.attrs["total_trajectories"] = written
                target.flush()
                _, observed = repair_gravity_transport(
                    arrays["gravity_dir"],
                    tuple(direction),
                    enabled=bool(spec.get("repair_uninitialized_gravity", False)),
                )
                if correction is None:
                    correction = {**observed, "sampler_calls": 1}
                else:
                    correction["changed_components"] += observed["changed_components"]
                    correction["nonfinite_components"] += observed[
                        "nonfinite_components"
                    ]
                    correction["sampler_calls"] += 1
                del rollouts, arrays
        corrections[split] = correction
        source_counts[split] = written
        sources[split] = path
    metadata["gravity_repair"] = corrections
    metadata["source_raw_hdf5"] = {
        split: {"path": str(path), "sha256": _hash(path)}
        for split, path in sources.items()
    }
    result = import_legacy_datasets(
        sources,
        output / "canonical",
        metadata,
        world_rotation_xyzw=rotation,
        gravity_direction=(
            tuple(direction)
            if spec.get("repair_uninitialized_gravity", False)
            else None
        ),
        preserve_legacy_velocity=spec.get("preserve_legacy_velocity", False),
    )
    result.update(
        status="completed",
        source_trajectory_counts=source_counts,
        teacher_origin=metadata["teacher_origin"],
        gravity_repair=corrections,
        source_bug_corrected=bool(spec.get("repair_uninitialized_gravity", False)),
        elapsed_seconds=time.perf_counter() - started,
        scope="real_official_teacher_data_not_full_training_qualification",
    )
    (output / "generation-result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)
    )
    (output / "source-physics.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--trust-external-code", action="store_true", required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Run real original teacher collection instead of training",
    )
    args = parser.parse_args()
    operation = generate_official_dataset if args.generate else run_reference
    extra = (
        {} if args.generate else {"resume": args.resume, "stop_after": args.stop_after}
    )
    result = operation(
        json.loads(args.spec.read_text()),
        trust_external_code=args.trust_external_code,
        **extra,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
