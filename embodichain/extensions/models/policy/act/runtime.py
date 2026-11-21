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

"""Native LeRobot ACT loading and device-resident simulation inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .training import _require_sdk

__all__ = ["ACTInferencePolicy", "load_act_policy", "policy_fingerprint"]


def policy_fingerprint(path: str | Path) -> str:
    """Hash the portable native policy artifact, including processor statistics.

    Args:
        path: LeRobot pretrained-model directory.

    Returns:
        SHA-256 of filenames and contents, excluding the redundant run manifest.
    """
    root = Path(path)
    digest = hashlib.sha256()
    for file in sorted(root.iterdir()):
        if file.is_file() and file.name != "embodichain_act.json":
            digest.update(file.name.encode() + b"\0")
            with file.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


class ACTInferencePolicy:
    """Run the upstream policy and its frozen processors on the simulation device.

    Args:
        policy: Official LeRobot ACTPolicy.
        preprocessor: Official observation processor pipeline.
        postprocessor: Official action processor pipeline.
        metadata: Task, data and action identities associated with training.
    """

    def __init__(
        self,
        policy: Any,
        preprocessor: Any,
        postprocessor: Any,
        metadata: Mapping[str, Any],
    ) -> None:
        self.policy = policy.eval()
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.metadata = dict(metadata)
        self.image_key = self.metadata["image_key"]
        self.reset()

    def reset(self) -> None:
        """Forget action chunks and ensemble history at every environment reset."""
        self.policy.reset()

    @torch.inference_mode()
    def evaluate_dataset(
        self, root: str | Path, *, batch_size: int = 32, samples: int = 128
    ) -> dict[str, float]:
        """Measure prior-prediction L1 on independent recorded action windows.

        Args:
            root: Validation dataset with the training observation/action schema.
            batch_size: Bounded evaluation minibatch size.
            samples: Maximum number of anchors, chosen with a fixed CPU seed.

        Returns:
            Normalized action L1 and number of valid supervised action elements.
        """
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from torch.utils.data import DataLoader, Subset

        if batch_size < 1 or samples < 1:
            raise ValueError("Validation batch and sample counts must be positive")
        root = Path(root).resolve()
        schema = self.metadata["schema"]
        dataset = LeRobotDataset(
            root.name,
            root=root,
            delta_timestamps={
                "action": [
                    i * schema["action"]["step_dt"]
                    for i in range(self.policy.config.chunk_size)
                ]
            },
            return_uint8=True,
            video_backend="pyav",
        )
        for key, declared in (
            ("observation.state", schema["observation"]),
            ("action", schema["action"]),
        ):
            if list(dataset.features[key]["names"]) != declared["names"]:
                raise ValueError("Validation feature order differs from training")
        import math

        if not math.isclose(
            dataset.fps * schema["action"]["step_dt"], 1.0, abs_tol=1e-8
        ):
            raise ValueError("Validation cadence differs from training")
        indices = torch.randperm(
            len(dataset), generator=torch.Generator().manual_seed(8128)
        )[:samples].tolist()
        total, count = 0.0, 0
        for batch in DataLoader(
            Subset(dataset, indices), batch_size=batch_size, num_workers=0
        ):
            batch[self.image_key] = batch[self.image_key].float().div(255)
            batch = self.preprocessor(batch)
            loss, _ = self.policy(batch)
            valid = int((~batch["action_is_pad"]).sum()) * len(
                schema["action"]["names"]
            )
            total += float(loss) * valid
            count += valid
        if count == 0:
            raise ValueError("Validation contains no supervised action elements")
        return {"action_l1": total / count, "valid_action_elements": count}

    @torch.inference_mode()
    def select_action(self, rgb: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Return a physical action after the official temporal ensemble.

        Args:
            rgb: Current uint8 RGB pixels, shaped ``[1, 3, H, W]``.
            state: Current ordered joint state, shaped ``[1, state_dim]``.

        Returns:
            Finite commands ``[1, action_dim]`` on the policy device.
        """
        if rgb.dtype != torch.uint8 or rgb.ndim != 4 or rgb.shape[0] != 1:
            raise ValueError("ACT rollout expects one uint8 BCHW RGB observation")
        batch = self.preprocessor(
            {self.image_key: rgb.float().div(255), "observation.state": state.float()}
        )
        action = self.postprocessor(self.policy.select_action(batch))
        expected = (1, len(self.metadata["schema"]["action"]["names"]))
        if action.shape != expected or not torch.isfinite(action).all():
            raise ValueError("ACT produced non-finite or incompatible control commands")
        return action


def load_act_policy(
    path: str | Path, device: str = "cpu", *, untrained: bool = False
) -> ACTInferencePolicy:
    """Load a native ACT directory with its processors and EmbodiChain contract.

    Args:
        path: Native pretrained-model directory, including ``embodichain_act.json``.
        device: Target policy and action device.
        untrained: Reinitialize the model with the original backbone initializer
            and frozen data statistics, for the untrained baseline.

    Returns:
        Stateful single-environment inference adapter. Call reset after env.reset.

    Raises:
        ValueError: If the directory is a historical custom ACT artifact or its
            task/processor manifest is missing or inconsistent.
    """
    _require_sdk()
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        policy_action_to_transition,
        transition_to_policy_action,
    )

    root = Path(path).resolve()
    if root.is_file():
        raise ValueError(
            "Historical EmbodiChain ACT .pt files are incompatible; retrain with LeRobot 0.6.1"
        )
    manifest = root / "embodichain_act.json"
    if not manifest.is_file():
        raise ValueError(
            "The native policy is missing its EmbodiChain task/action manifest"
        )
    metadata = json.loads(manifest.read_text())
    if metadata.get("recipe") != "lerobot-act-v1" or metadata.get(
        "policy_sha256"
    ) != policy_fingerprint(root):
        raise ValueError("ACT policy manifest or artifact fingerprint differs")
    config = ACTConfig.from_pretrained(root, local_files_only=True)
    config.device = device
    config.pretrained_backbone_weights = (
        metadata["backbone_initialization"] if untrained else None
    )
    policy = (
        ACTPolicy(config)
        if untrained
        else ACTPolicy.from_pretrained(
            root, config=config, local_files_only=True, strict=True
        )
    )
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        root,
        config_filename="policy_preprocessor.json",
        local_files_only=True,
        overrides={
            "device_processor": {"device": device},
            "normalizer_processor": {"device": device},
        },
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        root,
        config_filename="policy_postprocessor.json",
        local_files_only=True,
        overrides={"device_processor": {"device": device}},
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return ACTInferencePolicy(policy.to(device), preprocessor, postprocessor, metadata)
