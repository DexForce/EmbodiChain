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

import torch


class QposSeedSampler:
    """
    Standard joint seed sampler for IK solving.

    Generates joint seed samples for each target pose in a batch, including the provided seed and random samples within joint limits.

    Args:
        num_samples (int): Number of samples per batch (including the seed).
        dof (int): Degrees of freedom.
        device (torch.device): Target device.
    """

    def __init__(self, num_samples: int, dof: int, device: torch.device):
        self.num_samples = num_samples
        self.dof = dof
        self.device = device

    def sample(
        self,
        qpos_seed: torch.Tensor,
        lower_limits: torch.Tensor,
        upper_limits: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Generate joint seed samples for IK solving.

        Args:
            qpos_seed (torch.Tensor): (batch_size, dof) or (1, dof) initial seed.
            lower_limits (torch.Tensor): (dof,) lower joint limits.
            upper_limits (torch.Tensor): (dof,) upper joint limits.
            batch_size (int): Batch size.

        Returns:
            torch.Tensor: (batch_size * num_samples, dof) joint seeds.
        """
        joint_seeds_list = []
        for i in range(batch_size):
            current_seed = (
                qpos_seed[i].unsqueeze(0)
                if qpos_seed.shape[0] == batch_size
                else qpos_seed
            )
            if self.num_samples > 1:
                rand_part = lower_limits + (upper_limits - lower_limits) * torch.rand(
                    (self.num_samples - 1, self.dof), device=self.device
                )
            else:
                rand_part = torch.empty((0, self.dof), device=self.device)
            seeds = torch.cat([current_seed, rand_part], dim=0)
            joint_seeds_list.append(seeds)
        return torch.cat(joint_seeds_list, dim=0)

    def repeat_target_xpos(
        self, target_xpos: torch.Tensor, num_samples: int
    ) -> torch.Tensor:
        """Repeat each target pose num_samples times for batch processing.

        Args:
            target_xpos (torch.Tensor): (batch_size, 4, 4) or (batch_size, 3, 3) target poses.
            num_samples (int): Number of repeats per batch.

        Returns:
            torch.Tensor: (batch_size * num_samples, 4, 4) or (batch_size * num_samples, 3, 3)
        """
        repeated_list = [
            target_xpos[i].unsqueeze(0).repeat(num_samples, 1, 1)
            for i in range(target_xpos.shape[0])
        ]
        return torch.cat(repeated_list, dim=0)
