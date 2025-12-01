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

from dataclasses import dataclass


@dataclass
class ContactReport:
    """Data structure for contact report in the simulation engine.
    Attributes:
        contact_data (torch.Tensor): Contact data tensor.
        contact_user_ids (torch.Tensor): DexSim user IDs for the contacts.
        contact_env_ids (torch.Tensor): Environment IDs for the contacts.
    """

    contact_data: torch.Tensor
    """
    contact data:
        [num_contacts, 11] tensor with each row representing a contact point:
        [0] - position x
        [1] - position y
        [2] - position z
        [3] - normal x
        [4] - normal y
        [5] - normal z
        [6] - friction x
        [7] - friction y
        [8] - friction z
        [9] - impulse
        [10] - distance
    """

    contact_user_ids: torch.Tensor
    """contact user ids, use rigid_object.get_user_id() and find which object it belongs to."""

    contact_env_ids: torch.Tensor
    """which arena the contact belongs to."""
