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

import torch
from embodichain.utils import logger


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
        if qpos_seed.shape == (batch_size, self.dof):
            seed_head = qpos_seed[:, None, :]
        elif qpos_seed.shape == (self.dof,):
            seed_head = qpos_seed.unsqueeze(0).repeat(batch_size, 1)[:, None, :]
        else:
            logger.log_error(
                f"Invalid qpos_seed shape {qpos_seed.shape} for batch_size {batch_size} and dof {self.dof}",
                ValueError,
            )
        n_random_samples = self.num_samples - 1

        # seed_random = torch.rand(
        #     size=(batch_size, n_random_samples, self.dof), device=self.device
        # )

        # save sampling time, repeat for each batch and sample in one go
        seed_random = torch.rand(
            size=(1, n_random_samples, self.dof), device=self.device
        )
        seed_random = seed_random.repeat(batch_size, 1, 1)
        seed_random = lower_limits + (upper_limits - lower_limits) * seed_random
        joint_seeds = torch.cat([seed_head, seed_random], dim=1)
        return joint_seeds.reshape(-1, self.dof)

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

        target_xpos_repeated = target_xpos.unsqueeze(1).repeat(1, num_samples, 1, 1)
        return target_xpos_repeated.reshape(-1, 4, 4)
