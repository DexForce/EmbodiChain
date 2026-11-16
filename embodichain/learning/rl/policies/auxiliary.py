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

"""Composable auxiliary training objectives with host-owned neural components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

import torch
from torch import nn

from embodichain.learning.models import MLP

__all__ = ["AuxiliaryTask", "ContactPrediction"]


class AuxiliaryTask(ABC):
    """Describe a training objective whose modules are registered by the policy.

    Component names are local to the task. A task named ``contact`` with a
    ``decoder`` component is registered as ``contact_decoder``. The same prefix
    applies to fields, outputs and buffers, preserving established checkpoint
    paths without an extra neural container. Read live buffers from the loss
    argument so device/dtype changes and checkpoint loads remain authoritative.
    """

    modules: Mapping[str, nn.Module]
    training_observation_groups: Mapping[str, tuple[str, ...]]
    training_field_specs: Mapping[str, tuple[tuple[int, ...], torch.dtype]]
    output_names: tuple[str, ...]

    def initial_buffers(self) -> dict[str, torch.Tensor | None]:
        """Return initial persistent buffers for registration on the policy.

        Returns:
            Local buffer names and initial values. The host moves and saves them.
        """
        return {}

    def validate_observation_groups(self, actor_groups: Sequence[str] | None) -> None:
        """Validate task-specific information boundaries before collection.

        Args:
            actor_groups: Selected actor observation paths, or a flat input.
        """

    def checkpoint_contract(self) -> dict[str, object]:
        """Describe stable objective semantics independently of parameter shapes.

        Returns:
            Serializable identity and data declarations. Custom objectives add
            loss settings that change behavior without changing neural state.
        """
        return {
            "type": f"{type(self).__module__}.{type(self).__qualname__}",
            "observations": dict(self.training_observation_groups),
            "fields": {
                name: {"shape": shape, "dtype": str(dtype)}
                for name, (shape, dtype) in self.training_field_specs.items()
            },
            "outputs": self.output_names,
        }

    @abstractmethod
    def evaluate(self, features: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Produce named predictions from graph-connected actor features.

        Args:
            features: Named tensors returned by the actor's feature forward pass.

        Returns:
            Local output names declared in ``output_names`` and their tensors.
        """
        raise NotImplementedError

    @abstractmethod
    def loss(
        self,
        labels: Mapping[str, torch.Tensor],
        outputs: Mapping[str, torch.Tensor],
        buffers: Mapping[str, torch.Tensor | None],
    ) -> torch.Tensor:
        """Return a scalar loss from aligned labels, predictions and live buffers.

        Args:
            labels: Local fields copied from the observation before the action.
            outputs: Local predictions from the current minibatch's forward pass.
            buffers: Live host-owned buffers after device/dtype changes or loading.

        Returns:
            Graph-connected scalar loss, before PPO's auxiliary coefficient.
        """
        raise NotImplementedError


class ContactPrediction(AuxiliaryTask):
    """Classify contact states at fixed sample points from an actor feature.

    Args:
        input_dim: Dimension of the selected actor feature.
        contact_points: Number of sample-point labels per environment.
        contact_classes: Number of mutually exclusive classes.
        hidden_sizes: Decoder MLP hidden layer sizes.
        class_weights: Optional positive cross-entropy weight per class.
        observation_group: Environment observation group containing integer labels.
        feature_key: Named feature returned by the actor's feature forward pass.
    """

    output_names = ("logits",)

    def __init__(
        self,
        input_dim: int,
        contact_points: int,
        contact_classes: int = 2,
        hidden_sizes: Sequence[int] = (128, 128),
        class_weights: Sequence[float] | None = None,
        observation_group: str = "contact_labels",
        feature_key: str = "latent",
    ) -> None:
        if contact_points <= 0 or contact_classes < 2:
            raise ValueError(
                "Contact prediction needs positive points and at least two classes."
            )
        self._class_weight_tensor(class_weights, contact_classes)
        self.contact_points = contact_points
        self.contact_classes = contact_classes
        self.class_weights = None if class_weights is None else tuple(class_weights)
        self.observation_group = observation_group
        self.feature_key = feature_key
        self.decoder = nn.Sequential(
            *MLP(input_dim, contact_points * contact_classes, hidden_sizes).children()
        )
        self.modules = {"decoder": self.decoder}
        self.training_observation_groups = {"labels": (observation_group,)}
        self.training_field_specs = {"labels": ((contact_points,), torch.long)}

    @staticmethod
    def _class_weight_tensor(
        weights: Sequence[float] | None, classes: int
    ) -> torch.Tensor | None:
        """Validate and construct optional FP32 categorical loss weights."""
        if weights is None:
            return None
        if len(weights) != classes or any(weight <= 0 for weight in weights):
            raise ValueError(
                "contact_class_weights must contain one positive weight per class."
            )
        return torch.tensor(weights, dtype=torch.float32)

    def initial_buffers(self) -> dict[str, torch.Tensor | None]:
        """Create the class-weight buffer owned by the policy checkpoint."""
        return {
            "class_weights": self._class_weight_tensor(
                self.class_weights, self.contact_classes
            )
        }

    def checkpoint_contract(self) -> dict[str, object]:
        """Record contact target and feature semantics for strict training resume."""
        return {
            **super().checkpoint_contract(),
            "feature_key": self.feature_key,
            "contact_points": self.contact_points,
            "contact_classes": self.contact_classes,
            "class_weights": self.class_weights,
        }

    def validate_observation_groups(self, actor_groups: Sequence[str] | None) -> None:
        """Require actor groups that exclude the label group and its ancestors."""
        if not actor_groups:
            raise ValueError(
                "Contact supervision requires explicit actor observation groups."
            )
        label = self.observation_group
        if any(
            group == label
            or group.startswith(label + ".")
            or label.startswith(group + ".")
            for group in actor_groups
        ):
            raise ValueError("Contact labels cannot enter actor observations.")

    def evaluate(self, features: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Decode one logit vector per sample point without detaching features."""
        logits = self.decoder(features[self.feature_key])
        return {"logits": logits.reshape(-1, self.contact_points, self.contact_classes)}

    def loss(
        self,
        labels: Mapping[str, torch.Tensor],
        outputs: Mapping[str, torch.Tensor],
        buffers: Mapping[str, torch.Tensor | None],
    ) -> torch.Tensor:
        """Return weighted mean categorical contact cross-entropy."""
        target = labels["labels"]
        if target.dtype != torch.long:
            raise TypeError(
                "contact_labels must contain integer class indices (torch.long)."
            )
        return nn.functional.cross_entropy(
            outputs["logits"].flatten(0, 1),
            target.flatten(),
            weight=buffers["class_weights"],
        )
