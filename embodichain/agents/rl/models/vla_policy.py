# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

"""VLA Policy for RL training with pretrained models.

This module provides VLAPolicy that inherits from Policy base class,
just like ActorCritic. VLAPolicy loads pretrained VLA model components
and exposes the same interface as other policies.
"""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
from tensordict import TensorDict

from .policy import Policy


class VLAPolicy(Policy):
    """VLA Policy that loads pretrained vision-language-action models.

    Similar to ActorCritic, this class inherits from Policy and implements
    the required methods. The difference is that VLAPolicy loads pretrained
    model components instead of training from scratch.

    VLA model components are loaded by the VLA team's implementation and
    should provide the necessary interfaces for action generation and value
    estimation.
    """

    def __init__(
        self,
        action_dim: int,
        device: torch.device,
        vla_model: nn.Module,
    ):
        """Initialize VLA policy with pretrained model.

        Args:
            action_dim: Dimension of action space
            device: Device to place policy on
            vla_model: Pretrained VLA model (vision encoder, language model,
                      action head, value head, etc.)
        """
        super().__init__()
        self.action_dim = action_dim
        self.device = device

        # Store VLA model
        self.vla_model = vla_model
        self.vla_model.to(self.device)

    @torch.no_grad()
    def forward(
        self, tensordict: TensorDict, deterministic: bool = False
    ) -> TensorDict:
        """Forward pass: generate action and value from VLA model.

        Args:
            tensordict: Must contain "observation" key with observation data
            deterministic: If True, use deterministic actions (passed to VLA model)

        Returns:
            Same tensordict with added keys:
                - "action": Sampled or deterministic action
                - "sample_log_prob": Log probability of action
                - "value": Value estimate
        """
        # VLA team should implement forward logic here
        # This is a template - actual implementation depends on VLA model structure
        obs = tensordict["observation"]

        # Example: VLA model generates action and value
        action, log_prob, value = self.vla_model(obs, deterministic=deterministic)

        tensordict["action"] = action
        tensordict["sample_log_prob"] = log_prob
        tensordict["value"] = value.squeeze(-1)

        return tensordict

    @torch.no_grad()
    def get_value(self, tensordict: TensorDict) -> TensorDict:
        """Get value estimate from VLA model.

        Args:
            tensordict: Must contain "observation" key

        Returns:
            Same tensordict with added "value" key
        """
        obs = tensordict["observation"]

        # VLA team implements value computation
        value = self.vla_model.get_value(obs)

        tensordict["value"] = value.squeeze(-1)
        return tensordict

    def evaluate_actions(self, tensordict: TensorDict) -> TensorDict:
        """Evaluate actions using VLA model.

        Args:
            tensordict: Must contain:
                - "observation": Observation data
                - "action": Actions to evaluate

        Returns:
            Same tensordict with added keys:
                - "sample_log_prob": Log probability of actions
                - "entropy": Entropy of action distribution
                - "value": Value estimate
        """
        obs = tensordict["observation"]
        actions = tensordict["action"]

        # VLA team implements action evaluation
        log_prob, entropy, value = self.vla_model.evaluate_actions(obs, actions)

        tensordict["sample_log_prob"] = log_prob
        tensordict["entropy"] = entropy
        tensordict["value"] = value.squeeze(-1)

        return tensordict


def load_vla_model(
    model_path: str,
    model_class: Optional[str] = None,
    model_config: Optional[dict] = None,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Load VLA model from checkpoint.

    This function should be implemented by the VLA team to load their
    pretrained VLA model (vision encoder, language model, action head, etc.).

    The returned module should have methods:
    - forward(obs) -> (action, log_prob, value)
    - get_value(obs) -> value
    - evaluate_actions(obs, actions) -> (log_prob, entropy, value)

    Args:
        model_path: Path to checkpoint file
        model_class: Fully qualified class name for VLA model
        model_config: Configuration dict for model initialization
        device: Device to load model on

    Returns:
        Initialized VLA model module

    Example implementation by VLA team:
    ```python
    def load_vla_model(model_path, model_class, model_config, device):
        import importlib

        # Import VLA model class
        module_name, class_name = model_class.rsplit(".", 1)
        module = importlib.import_module(module_name)
        ModelClass = getattr(module, class_name)

        # Initialize model
        model = ModelClass(**model_config)

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        model.to(device)
        model.eval()

        return model
    ```
    """
    raise NotImplementedError(
        "load_vla_model() must be implemented. "
        f"Model path: {model_path}, class: {model_class}, config: {model_config}"
    )


def build_vla_policy(
    policy_block: dict,
    action_dim: int,
    device: torch.device,
) -> VLAPolicy:
    """Build VLA policy from configuration.

    Args:
        policy_block: Configuration dict
        action_dim: Dimension of action space
        device: Device to place policy on

    Returns:
        Initialized VLAPolicy instance
    """
    vla_config = policy_block.get("vla_config")
    if vla_config is None:
        raise ValueError("VLA policy requires 'vla_config' in policy block")

    model_path = vla_config.get("model_path")
    if model_path is None:
        raise ValueError("VLA config requires 'model_path'")

    model_class = vla_config.get("model_class")
    model_config = vla_config.get("model_config", {})
    model_config["action_dim"] = action_dim

    # Load VLA model
    vla_model = load_vla_model(
        model_path=model_path,
        model_class=model_class,
        model_config=model_config,
        device=device,
    )

    # Create VLAPolicy instance
    policy = VLAPolicy(
        action_dim=action_dim,
        device=device,
        vla_model=vla_model,
    )

    return policy
