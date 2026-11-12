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

"""Private paired-checkpoint persistence for the native Sharpa example."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from embodichain.learning.rl.experimental._host_replay import (
    _host_replay_state_dict,
    _load_host_replay_state_dict,
)

__all__: list[str] = []

_PHYSICAL_FILES = (
    "_full_native_task.py",
    "_native_task.py",
    "_simulation.py",
    "_native_asset.py",
)
_ADAPTER_FILES = (
    "objects/articulation.py",
    "physics/_joint_drive_view.py",
    "physics/_cylinder_variants.py",
    "physics/_contact_material_view.py",
    "physics/_source_joint_properties.py",
    "sensors/contact/_geometry_contact_sensor.py",
    "sensors/contact/_warp/geometry_contact.py",
)


def _verify_resume(
    checkpoint: Path, snapshot: dict[str, Any], hashes: dict[str, str], profile: str
) -> None:
    """Reject source/layout changes before constructing a physical recovery scene."""
    if (
        not snapshot.get("native_environment")
        or len(snapshot.get("native_identity", {}).get("scales", [])) != 8
    ):
        raise ValueError("Resume requires a native eight-scale training checkpoint")
    experiment = json.loads((checkpoint.parent / "experiment.json").read_text())
    if experiment["resource_profile"] != profile:
        raise ValueError("Resume resource profile must match the saved experiment")
    for name in _PHYSICAL_FILES:
        if experiment["source_hashes"].get(name) != hashes[name]:
            raise ValueError(
                f"Physical implementation changed: {name}; use an explicit implementation-transfer evaluation"
            )
    root = Path(__file__).resolve().parents[3] / "embodichain/lab/sim"
    for name in _ADAPTER_FILES:
        original = checkpoint.parent / "sources" / name
        if (
            hashlib.sha256((root / name).read_bytes()).digest()
            != hashlib.sha256(original.read_bytes()).digest()
        ):
            raise ValueError(f"Native physics/sensor adapter changed: {name}")


def _save_replay(output: Path, learner: Any, model: Path, iteration: int) -> None:
    """Synchronously serialize borrowed host replay before allowing new writes."""
    if not hasattr(learner.buffer, "storage"):
        return
    temporary = output / "native_replay_latest.tmp"
    torch.save(
        {
            "model_checkpoint": model.name,
            "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "iteration": iteration,
            "replay": _host_replay_state_dict(learner.buffer.storage),
            "torch_cpu_rng": torch.get_rng_state(),
            "torch_cuda_rng": torch.cuda.get_rng_state_all(),
            "numpy_rng": np.random.get_state(),
            "exact_simulation_trace_resume": False,
        },
        temporary,
    )
    temporary.replace(output / "native_replay_latest.pt")


def _restore_learner(
    learner: Any,
    snapshot: dict[str, Any],
    checkpoint: Path,
    replay: Path | None,
    output: Path,
) -> bool:
    """Restore native model/optimizer state and optionally a paired CPU ring.

    The caller constructs a fresh simulator first. A previous final sampler row
    closes the interrupted sequence; new observations start at episode step zero.
    This preserves historical transitions without joining unrelated trajectories.
    """
    learner.actor.load_state_dict(snapshot["training_actor"], strict=True)
    learner.critic.load_state_dict(snapshot["critic"], strict=True)
    learner.wm.load_state_dict(snapshot["world_model"], strict=True)
    learner.opt_pi.load_state_dict(snapshot["actor_optimizer"])
    learner.opt_v.load_state_dict(snapshot["critic_optimizer"])
    learner.opt_wm.load_state_dict(snapshot["world_model_optimizer"])
    if learner.actor_avg is not None:
        learner.actor_avg.load_state_dict(snapshot["actor_average"], strict=True)
    model_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if replay is not None:
        if not hasattr(learner.buffer, "storage"):
            raise ValueError("Paired CPU replay requires reference_host")
        receipt = torch.load(replay, map_location="cpu", weights_only=False)
        if (
            receipt["iteration"] != snapshot["iteration"]
            or receipt["model_sha256"] != model_sha
        ):
            raise ValueError("Replay/model checkpoint mismatch")
        storage = learner.buffer.storage
        _load_host_replay_state_dict(storage, receipt["replay"])
        if storage.size:
            storage.data["done"][(storage.ptr - 1) % storage.steps].fill_(True)
        torch.set_rng_state(receipt["torch_cpu_rng"])
        torch.cuda.set_rng_state_all(receipt["torch_cuda_rng"])
        np.random.set_state(receipt["numpy_rng"])
    parent = checkpoint.parent / "recovery.json"
    lineage = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": model_sha,
        "completed_iteration": snapshot["iteration"],
        "source_native_environment": True,
        "initialized_from_native_checkpoint": True,
        "actor_optimizer_restarted": False,
        "ema_restarted": False,
        "replay_rebuilt": replay is None,
        "restored_replay": str(replay.resolve()) if replay else None,
        "environment_and_physical_rng_reinitialized": True,
        "global_rng_restored": replay is not None,
        "exact_training_trace_resume": False,
        "sampler_boundary": "Close the last historical sampler row before new simulator-reset transitions; stored states/actions/rewards remain unchanged.",
    }
    if parent.is_file():
        lineage["parent_recovery"] = json.loads(parent.read_text())
    (output / "recovery.json").write_text(json.dumps(lineage, indent=2) + "\n")
    return replay is not None
