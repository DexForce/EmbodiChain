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

"""Checkpoint validation, random streams, and atomic persistence for RL trainers."""

from __future__ import annotations

import os
import random
import shutil
import tempfile
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__: list[str] = []

_SCHEMA_VERSION = 2


def _atomic_write(path: str | Path, write: Any) -> None:
    """Replace a file only after its complete contents have been written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _capture_rng(device: torch.device) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def _atomic_copy(source: Path, destination: Path) -> None:
    with source.open("rb") as incoming:
        _atomic_write(destination, lambda stream: shutil.copyfileobj(incoming, stream))


def _restore_rng(state: Mapping[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def _plain(value: Any) -> Any:
    """Convert configuration and space values to weights-only-compatible data."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return str(value)


def _type_name(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _space_layout(space: Any) -> Any:
    if space is None:
        return None
    result = {"type": _type_name(space)}
    spaces = getattr(space, "spaces", None)
    if isinstance(spaces, Mapping):
        result["spaces"] = [[k, _space_layout(v)] for k, v in spaces.items()]
    else:
        for name in ("shape", "dtype", "low", "high", "n", "nvec"):
            if hasattr(space, name):
                result[name] = _plain(getattr(space, name))
    return result


def _policy_normalizers(policy: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in policy.state_dict().items()
        if "normalizer" in key.split(".")[0]
    }


def _metadata(trainer: Any) -> dict[str, Any]:
    policy = getattr(trainer.policy, "module", trainer.policy)
    algorithm = trainer.algorithm
    normalizer = getattr(trainer, "observation_normalizer", None)
    attributes = {
        name: _plain(getattr(policy, name, None))
        for name in (
            "obs_dim",
            "action_dim",
            "critic_obs_dim",
            "actor_obs_groups",
            "critic_obs_groups",
            "uses_separate_critic_obs",
            "action_std_range",
        )
    }
    normalizers = {
        name: {"type": _type_name(module), "epsilon": getattr(module, "epsilon", None)}
        for name, module in policy.named_modules()
        if "normalizer" in name
    }
    if normalizer is not None:
        normalizers["observation_normalizer"] = {
            "type": _type_name(normalizer),
            "normalize_mask": _plain(normalizer.normalize_mask),
        }
    return {
        "trainer": _type_name(trainer),
        "algorithm": _type_name(algorithm),
        "algorithm_config": _plain(
            {
                k: v
                for k, v in vars(algorithm.cfg).items()
                if k not in {"device", "optimizer", "lr_scheduler"}
            }
        ),
        "policy": _type_name(policy),
        "policy_attributes": attributes,
        "modules": [
            [name, _type_name(module), module.extra_repr()]
            for name, module in policy.named_modules()
            if "normalizer" not in name
        ],
        "normalizers": normalizers,
        "environment": _type_name(trainer.env),
        "observation_space": _space_layout(
            getattr(
                trainer.env,
                "single_observation_space",
                getattr(trainer.env, "observation_space", None),
            )
        ),
        "action_space": _space_layout(
            getattr(
                trainer.env,
                "single_action_space",
                getattr(trainer.env, "action_space", None),
            )
        ),
        "optimizer": _type_name(algorithm.optimizer),
        "optimizer_kwargs": _plain(algorithm.cfg.optimizer.kwargs),
        "optimizer_groups": [
            [list(p.shape) for p in group["params"]]
            for group in algorithm.optimizer.param_groups
        ],
        "evaluation": {
            "metric": getattr(
                trainer,
                "best_eval_metric",
                getattr(getattr(trainer, "cfg", None), "best_eval_metric", None),
            ),
            "mode": getattr(
                trainer,
                "best_eval_mode",
                getattr(getattr(trainer, "cfg", None), "best_eval_mode", None),
            ),
        },
        "context": _plain(getattr(trainer, "_checkpoint_context", {})),
    }


def _differences(saved: Any, current: Any, path: str = "metadata") -> list[str]:
    if isinstance(saved, Mapping) and isinstance(current, Mapping):
        differences = []
        for key in sorted(saved.keys() | current.keys()):
            child = f"{path}.{key}"
            if key not in saved or key not in current:
                differences.append(child)
            else:
                differences.extend(_differences(saved[key], current[key], child))
        return differences
    return [] if saved == current else [path]


def _sync_errors(error: Exception | None, distributed: bool) -> None:
    """Make every rank fail before any peer enters the next collective."""
    if not distributed:
        if error is not None:
            raise error
        return
    results = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(
        results, None if error is None else f"{type(error).__name__}: {error}"
    )
    failures = [f"rank {rank}: {value}" for rank, value in enumerate(results) if value]
    if failures:
        raise RuntimeError("Checkpoint operation failed: " + "; ".join(failures))


def _validate(trainer: Any, payload: Any, scheduler: str) -> list[str]:
    if scheduler not in {"restore", "restart"}:
        raise ValueError("resume_scheduler must be 'restore' or 'restart'.")
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint must contain a training-state mapping.")
    version = payload.get("schema_version")
    if (version is not None and type(version) is not int) or version not in {
        None,
        1,
        _SCHEMA_VERSION,
    }:
        raise ValueError(f"Unsupported checkpoint schema version {version!r}.")
    if version != _SCHEMA_VERSION and (version == 1) != hasattr(
        trainer, "_rollout_index"
    ):
        raise ValueError("Legacy checkpoint trainer type is incompatible.")
    required = {"policy", "optimizer", "global_step", "num_updates"}
    if version == _SCHEMA_VERSION:
        required |= {
            "metadata",
            "rng_by_rank",
            "normalizer_by_rank",
            "world_size",
            "best_eval_value",
            "best_checkpoint",
            "lr_scheduler",
            "lr_scheduler_cfg",
        }
        if hasattr(trainer, "_rollout_index"):
            required.add("rollout_index")
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {sorted(missing)}")
    if int(payload["global_step"]) < 0 or int(payload["num_updates"]) < 0:
        raise ValueError("Checkpoint training counters cannot be negative.")
    policy = getattr(trainer.policy, "module", trainer.policy)
    current_state = policy.state_dict()
    saved_state = payload["policy"]
    if current_state.keys() != saved_state.keys() or any(
        current_state[key].shape != saved_state[key].shape
        or current_state[key].dtype != saved_state[key].dtype
        for key in current_state
    ):
        raise ValueError(
            "Incompatible checkpoint policy state keys, shapes, or dtypes."
        )
    if version == _SCHEMA_VERSION:
        differences = _differences(payload["metadata"], _metadata(trainer))
        if differences:
            raise ValueError(
                "Incompatible checkpoint configuration: " + ", ".join(differences)
            )
        world_size = getattr(trainer, "world_size", 1)
        if payload["world_size"] != world_size:
            raise ValueError(
                "Checkpoint world_size differs from the current training job."
            )
        for name in ("rng_by_rank", "normalizer_by_rank"):
            if len(payload[name]) != world_size:
                raise ValueError(f"Checkpoint {name} must contain one entry per rank.")
        rank = getattr(trainer, "rank", 0)
        rng = payload["rng_by_rank"][rank]
        if not {"python", "numpy", "torch"} <= rng.keys():
            raise ValueError("Checkpoint RNG state is incomplete.")
        if trainer.algorithm.device.type == "cuda" and "cuda" not in rng:
            raise ValueError("Checkpoint has no CUDA RNG for this rank.")
        random.Random().setstate(rng["python"])
        numpy_state = rng["numpy"]
        np.random.RandomState().set_state(
            (
                numpy_state[0],
                np.asarray(numpy_state[1], dtype=np.uint32),
                *numpy_state[2:],
            )
        )
        torch.Generator().set_state(rng["torch"].cpu())
        if trainer.algorithm.device.type == "cuda":
            torch.Generator(device=trainer.algorithm.device).set_state(
                rng["cuda"].cpu()
            )
        normalizers = payload["normalizer_by_rank"][rank]
        expected = _policy_normalizers(policy)
        if normalizers.keys() != expected.keys() or any(
            normalizers[k].shape != expected[k].shape for k in expected
        ):
            raise ValueError("Checkpoint rank normalizer state is incompatible.")
    normalizer = getattr(trainer, "observation_normalizer", None)
    saved_normalizer = payload.get("observation_normalizer")
    if (normalizer is None) != (saved_normalizer is None):
        raise ValueError(
            "Checkpoint observation normalization differs from the trainer."
        )
    if normalizer is not None:
        expected = normalizer.state_dict()
        if float(saved_normalizer["count"]) <= 0:
            raise ValueError(
                "Checkpoint observation normalizer count must be positive."
            )
        for key in ("mean", "var", "normalize_mask"):
            if saved_normalizer[key].shape != expected[key].shape:
                raise ValueError(
                    f"Checkpoint observation_normalizer.{key} shape differs."
                )
        if not torch.equal(
            saved_normalizer["normalize_mask"].cpu(), expected["normalize_mask"].cpu()
        ):
            raise ValueError("Checkpoint observation normalization mask differs.")
    trainer.algorithm.validate_state_dict(payload, resume_scheduler=scheduler)
    return (
        []
        if version == _SCHEMA_VERSION
        else ["configuration metadata", "per-rank RNG", "per-rank normalizers"]
    )


def _save(trainer: Any, path: str | Path, extra: Mapping[str, Any]) -> str:
    """Collect local state on all ranks; rank zero owns files and callbacks."""
    distributed = getattr(trainer, "distributed", False)
    rank = getattr(trainer, "rank", 0)
    policy = getattr(trainer.policy, "module", trainer.policy)
    error = None
    local = None
    try:
        local = {
            "rng": _capture_rng(trainer.algorithm.device),
            "normalizer": _policy_normalizers(policy),
        }
    except Exception as exc:
        error = exc
    _sync_errors(error, distributed)
    ranks = [local]
    if distributed:
        ranks = [None] * trainer.world_size
        torch.distributed.all_gather_object(ranks, local)
    error = None
    if rank == 0:
        try:
            best = trainer.best_checkpoint_path
            payload = {
                "schema_version": _SCHEMA_VERSION,
                "metadata": _metadata(trainer),
                "world_size": len(ranks),
                "rng_by_rank": [value["rng"] for value in ranks],
                "normalizer_by_rank": [value["normalizer"] for value in ranks],
                "global_step": trainer.global_step,
                "num_updates": trainer.num_updates,
                "policy": policy.state_dict(),
                "config_snapshot": _plain(getattr(trainer, "_checkpoint_config", None)),
                **trainer.algorithm.state_dict(),
                "best_eval_value": trainer.best_eval_value,
                "best_checkpoint": (
                    os.path.relpath(best, Path(path).parent) if best else None
                ),
                **extra,
            }
            _atomic_write(path, lambda stream: torch.save(payload, stream))
            previous = trainer.latest_checkpoint_path
            trainer.latest_checkpoint_path = str(path)
            try:
                callback = getattr(trainer, "_checkpoint_callback", None)
                if callback is not None:
                    callback(trainer.get_summary())
            except Exception:
                trainer.latest_checkpoint_path = previous
                raise
        except Exception as exc:
            error = exc
    _sync_errors(error, distributed)
    return str(path)


def _load(trainer: Any, path: str | Path, scheduler: str) -> dict[str, Any]:
    distributed = getattr(trainer, "distributed", False)
    payload = None
    error = None
    missing = []
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError as exc:
            if "weights_only" not in str(exc):
                raise
            payload = torch.load(path, map_location="cpu")
        missing = _validate(trainer, payload, scheduler)
        if payload.get("schema_version") != _SCHEMA_VERSION:
            _validate_legacy_context(trainer, Path(path))
    except Exception as exc:
        error = exc
    _sync_errors(error, distributed)
    error = None
    try:
        policy = getattr(trainer.policy, "module", trainer.policy)
        policy.load_state_dict(payload["policy"])
        trainer.algorithm.load_state_dict(payload, resume_scheduler=scheduler)
        rank = getattr(trainer, "rank", 0)
        if "normalizer_by_rank" in payload:
            policy.load_state_dict(payload["normalizer_by_rank"][rank], strict=False)
        normalizer = getattr(trainer, "observation_normalizer", None)
        if normalizer is not None:
            normalizer.load_state_dict(payload["observation_normalizer"])
        trainer.global_step = int(payload["global_step"])
        trainer.num_updates = int(payload["num_updates"])
        trainer.best_eval_value = payload.get("best_eval_value")
        trainer.latest_checkpoint_path = str(path)
        if rank == 0:
            _restore_best(trainer, payload, Path(path))
        if hasattr(trainer.collector, "obs_td"):
            trainer.collector.obs_td = trainer.collector._reset_env()
            trainer.curr_ret.zero_()
            trainer.curr_len.zero_()
        else:
            trainer.collector._observation = None
            trainer._episode_return.zero_()
            trainer._episode_length.zero_()
        trainer.ret_window.clear()
        trainer.len_window.clear()
    except Exception as exc:
        error = exc
    _sync_errors(error, distributed)
    error = None
    if "rng_by_rank" in payload:
        try:
            _restore_rng(
                payload["rng_by_rank"][getattr(trainer, "rank", 0)],
                trainer.algorithm.device,
            )
        except Exception as exc:
            error = exc
    _sync_errors(error, distributed)
    if missing and getattr(trainer, "rank", 0) == 0:
        warnings.warn(
            "Legacy checkpoint has no "
            + ", ".join(missing)
            + "; restoring available learning state.",
            UserWarning,
            stacklevel=3,
        )
    return payload


def _restore_best(trainer: Any, payload: Mapping[str, Any], source: Path) -> None:
    best = payload.get("best_checkpoint")
    best_path = source.parent / best if best else None
    if best_path is None or not best_path.is_file():
        from embodichain.learning.rl.policy_evaluation.manifest import RunManifest

        try:
            best_path = RunManifest.load(source.resolve().parents[1]).checkpoints.get(
                "best"
            )
        except (OSError, ValueError, TypeError):
            pass
    trainer.best_checkpoint_path = None
    if best_path is not None and best_path.is_file():
        cfg = getattr(trainer, "cfg", None)
        directory = getattr(trainer, "checkpoint_dir", None) or cfg.checkpoint_dir
        destination = Path(directory) / "resumed_best.pt"
        if best_path.resolve() != destination.resolve():
            _atomic_copy(best_path, destination)
        trainer.best_checkpoint_path = str(destination)


def _validate_legacy_context(trainer: Any, source: Path) -> None:
    """Use a source run's config snapshots when an old payload has no metadata."""
    current = getattr(trainer, "_checkpoint_config", None)
    if current is None:
        return
    from embodichain.learning.rl.policy_evaluation.manifest import RunManifest
    from embodichain.learning.rl.train import _checkpoint_context
    from embodichain.utils.utility import load_config

    try:
        manifest = RunManifest.load(source.resolve().parents[1])
    except FileNotFoundError:
        return
    saved = load_config(str(manifest.configs["train"]))
    gym = manifest.configs.get("gym")
    saved_context = _checkpoint_context(saved, load_config(str(gym)) if gym else None)
    differences = _differences(
        _plain(saved_context), _plain(trainer._checkpoint_context), "config"
    )
    if saved["algorithm"]["name"].lower() != current["algorithm"]["name"].lower():
        differences.append("config.algorithm.name")

    def optimization(config: dict) -> dict:
        cfg = dict(config["algorithm"].get("cfg", {}))
        cfg.pop("lr_scheduler", None)
        optimizer = dict(cfg.pop("optimizer", {}))
        optimizer.pop("learning_rate", None)
        cfg["optimizer"] = optimizer
        return cfg

    differences.extend(
        _differences(optimization(saved), optimization(current), "config.algorithm.cfg")
    )
    if differences:
        raise ValueError(
            "Incompatible legacy checkpoint configuration: " + ", ".join(differences)
        )


def _resolve_resume(resume: str | Path | None, run: str | Path | None) -> Path | None:
    if resume is None:
        if run is not None:
            raise ValueError("--run requires --resume latest or best.")
        return None
    if str(resume) in {"latest", "best"}:
        if run is None:
            raise ValueError("--resume latest/best requires --run.")
        from embodichain.learning.rl.policy_evaluation.manifest import RunManifest

        selected, path = RunManifest.load(run).select_checkpoint(str(resume))
        print(f"Resume checkpoint ({selected}): {path}")
    else:
        if run is not None:
            raise ValueError("--run is only used with --resume latest/best.")
        path = Path(resume).expanduser().resolve()
        print(f"Resume checkpoint: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    return path
