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

"""Configuration and provenance for LeRobot's official ACT training workflow."""

from __future__ import annotations

import hashlib
from importlib.metadata import version
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from lerobot.configs.train import TrainPipelineConfig

__all__ = ["build_act_train_config", "dataset_fingerprint"]


def _require_sdk() -> None:
    if version("lerobot") != "0.6.1":
        raise RuntimeError(
            "The official ACT example requires Python 3.12 and lerobot[training]==0.6.1"
        )


def dataset_fingerprint(root: str | Path) -> str:
    """Hash recorded metadata and frame/media content without loading the corpus.

    Args:
        root: Finalized local LeRobot dataset directory.

    Returns:
        SHA-256 binding relative filenames and file contents. SDK caches are excluded.
    """
    root = Path(root).resolve()
    digest = hashlib.sha256()
    for directory in ("meta", "data", "images", "videos"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file():
                digest.update(path.relative_to(root).as_posix().encode() + b"\0")
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
    return digest.hexdigest()


def build_act_train_config(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    schema: Mapping[str, Any],
    image_key: str = "observation.images.cam_high",
    device: str = "cuda:0",
    seed: int = 19001,
    updates: int = 30000,
    batch_size: int = 32,
    save_freq: int = 2000,
    num_workers: int = 4,
    smoke: bool = False,
) -> TrainPipelineConfig:
    """Build an upstream training config with explicit observation/action semantics.

    Args:
        dataset_root: Finalized training-only dataset; validation uses a separate root.
        output_dir: New output directory owned by LeRobot.
        schema: Ordered observation/action names, action semantics and control period.
        image_key: One recorded RGB camera feature.
        device: Torch training device.
        seed: Training seed.
        updates: Number of upstream optimizer updates.
        batch_size: Training minibatch size.
        save_freq: Upstream checkpoint interval.
        num_workers: Upstream DataLoader worker count.
        smoke: Use a small uninitialized model for an engineering check.

    Returns:
        LeRobot's native config, ready for serialization and its official CLI.

    Raises:
        ValueError: If the stored schema differs from the declared live control contract.
    """
    _require_sdk()
    import torch
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.act.configuration_act import ACTConfig

    root = Path(dataset_root).resolve()
    meta = LeRobotDatasetMetadata(root.name, root=root)
    for key, declaration in (
        ("observation.state", schema["observation"]),
        ("action", schema["action"]),
    ):
        feature = meta.features[key]
        if list(feature.get("names") or []) != list(declaration["names"]):
            raise ValueError(f"{key} feature order differs from the declared schema")
        if tuple(feature["shape"]) != (len(declaration["names"]),):
            raise ValueError(f"{key} feature shape differs from the declared schema")
        if (
            feature["dtype"] != "float32"
            or declaration.get("dtype", "float32") != "float32"
        ):
            raise ValueError(f"{key} must declare float32 values for ACT")
    if not schema["action"].get("semantics"):
        raise ValueError("An explicit action meaning is required")
    import math

    if not math.isclose(meta.fps * schema["action"]["step_dt"], 1.0, abs_tol=1e-8):
        raise ValueError("Dataset FPS differs from the action control period")
    image = meta.features[image_key]
    if image["dtype"] != "image" or len(image["shape"]) != 3 or image["shape"][-1] != 3:
        raise ValueError("ACT requires an HWC RGB image feature")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA training was requested but this Torch installation has no usable GPU"
        )
    if torch.device(device).type == "cuda" and torch.device(device).index not in (
        None,
        0,
    ):
        raise ValueError(
            "Select one GPU with CUDA_VISIBLE_DEVICES and use device='cuda:0' for the official trainer"
        )
    model_options = {}
    if smoke:
        model_options = dict(
            dim_model=32,
            n_heads=4,
            dim_feedforward=64,
            n_encoder_layers=1,
            n_decoder_layers=1,
            n_vae_encoder_layers=1,
            latent_dim=4,
            dropout=0.0,
            pretrained_backbone_weights=None,
        )
    policy = ACTConfig(
        device=device,
        push_to_hub=False,
        use_amp=False,
        input_features={
            "observation.state": PolicyFeature(
                FeatureType.STATE, (len(schema["observation"]["names"]),)
            ),
            image_key: PolicyFeature(FeatureType.VISUAL, (3, *image["shape"][:2])),
        },
        output_features={
            "action": PolicyFeature(
                FeatureType.ACTION, (len(schema["action"]["names"]),)
            )
        },
        chunk_size=30,
        n_action_steps=1,
        temporal_ensemble_coeff=0.01,
        kl_weight=10.0,
        optimizer_lr=1e-4,
        optimizer_lr_backbone=1e-5,
        optimizer_weight_decay=1e-4,
        **model_options,
    )
    return TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=root.name, root=str(root), eval_split=0.0, video_backend="pyav"
        ),
        policy=policy,
        output_dir=Path(output_dir).resolve(),
        seed=seed,
        steps=updates,
        batch_size=batch_size,
        save_freq=save_freq,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        env_eval_freq=0,
        eval_steps=0,
        log_freq=100,
        wandb=WandBConfig(enable=False),
    )
