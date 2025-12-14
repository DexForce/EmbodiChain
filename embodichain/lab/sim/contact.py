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

import numpy as np
import torch
import dexsim
from typing import Optional, Sequence
import uuid
from embodichain.utils import logger


class ContactReport:
    """Data structure for contact report in the simulation engine.
    Attributes:
        contact_data (torch.Tensor): Contact data tensor.
        contact_user_ids (torch.Tensor): DexSim user IDs for the contacts.
        contact_env_ids (torch.Tensor): Environment IDs for the contacts.
    """

    def __init__(
        self,
        contact_data: torch.Tensor,
        contact_user_ids: torch.Tensor,
        contact_env_ids: torch.Tensor,
    ):
        self.contact_data: torch.Tensor = contact_data
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

        self.contact_user_ids: torch.Tensor = contact_user_ids
        """[num_contacts, 2] of int, contact user ids, use rigid_object.get_user_id() and find which object it belongs to."""

        self.contact_env_ids: torch.Tensor = contact_env_ids
        """[num_contacts, ] of int, which arena the contact belongs to."""

        self._visualizer: Optional[dexsim.models.PointCloud] = None
        """contact point visualizer. Default to None"""

    def filter_by_user_ids(self, item_user_ids: torch.Tensor):
        """Filter contact report by specific user IDs.

        Args:
            item_user_ids (torch.Tensor): Tensor of user IDs to filter by.

        Returns:
            ContactReport: A new ContactReport instance containing only the filtered contacts.
        """
        filter0_mask = torch.isin(self.contact_user_ids[:, 0], item_user_ids)
        filter1_mask = torch.isin(self.contact_user_ids[:, 1], item_user_ids)
        filter_mask = torch.logical_or(filter0_mask, filter1_mask)
        filtered_contact_data = self.contact_data[filter_mask]
        filtered_contact_user_ids = self.contact_user_ids[filter_mask]
        filtered_contact_env_ids = self.contact_env_ids[filter_mask]
        return ContactReport(
            contact_data=filtered_contact_data,
            contact_user_ids=filtered_contact_user_ids,
            contact_env_ids=filtered_contact_env_ids,
        )

    def set_contact_point_visibility(
        self,
        sim: "SimulationManager",
        visible: bool = True,
        rgba: Optional[Sequence[int]] = None,
        point_size: float = 3.0,
    ):
        if visible:
            contact_position_arena = self.contact_data[:, :3]
            contact_offsets = sim.arena_offsets[self.contact_env_ids]
            contact_position_world = contact_position_arena + contact_offsets
            if self._visualizer is None:
                # create new visualizer
                temp_str = uuid.uuid4().hex
                self._visualizer = sim.get_env().create_point_cloud(name=temp_str)
            else:
                # update existing visualizer points
                self._visualizer.clear()
            rgba = rgba if rgba is not None else (0.8, 0.2, 0.2, 1.0)
            if len(rgba) != 4:
                logger.log_error(
                    f"Invalid rgba {rgba}, should be a sequence of 4 floats."
                )
            rgba = np.array(
                [
                    rgba[0],
                    rgba[1],
                    rgba[2],
                    rgba[3],
                ]
            )
            self._visualizer.add_points(
                points=contact_position_world.to("cpu").numpy(),
            )
            self._visualizer.set_point_size(point_size)
            self._visualizer.set_color(rgba)

        if not visible and isinstance(self._visualizer, dexsim.models.PointCloud):
            # clean visualizer points
            self._visualizer.clear()
