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
import numpy as np
from typing import List, Sequence
from dexsim.render import Light as _Light
from embodichain.lab.sim.cfg import LightCfg
from embodichain.lab.sim.common import BatchEntity
from embodichain.utils import logger


class Light(BatchEntity):
    """Light represents a batch of lights in the simulation.

    Each light supports the following properties:
        - Color (3 floats)
        - Intensity (1 float)
        - Falloff (1 float)
        - Location (3 floats)
    """

    def __init__(
        self,
        cfg: LightCfg,
        entities: List[_Light] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:

        super().__init__(cfg, entities, device)

    def set_color(
        self, colors: torch.Tensor, env_ids: Sequence[int] | None = None
    ) -> None:
        """Set color for one or more lights.

        Args:
            colors (torch.Tensor): Tensor of shape (M, 3) or (3,), representing RGB values.
                - If shape is (3,), the same color is applied to all targeted instances.
                - If shape is (M, 3), M must match the number of targeted instances.
            env_ids (Sequence[int] | None): Indices of instances to set. If None:
                - For colors.shape == (3,), applies to all instances.
                - For colors.shape == (M, 3), M must equal num_instances, applies per-instance.
        """
        self._apply_vector3(colors, env_ids, "set_color")

    def set_intensity(
        self, intensities: torch.Tensor, env_ids: Sequence[int] | None = None
    ) -> None:
        """Set intensity for one or more lights.

        Args:
            intensities (torch.Tensor): Tensor of shape (M,), (1,), or scalar (0-dim).
                - If scalar or shape (1,), the same intensity is applied to all targeted instances.
                - If shape (M,), M must match the number of targeted instances.
            env_ids (Sequence[int] | None): Indices of instances to set. If None:
                - For scalar/shape (1,), applies to all instances.
                - For shape (M,), M must equal num_instances, applies per-instance.
        """
        self._apply_scalar(intensities, env_ids, "set_intensity")

    def set_falloff(
        self, falloffs: torch.Tensor, env_ids: Sequence[int] | None = None
    ) -> None:
        """Set falloff (radius) for one or more lights.

        Args:
            falloffs (torch.Tensor): Tensor of shape (M,), (1,), or scalar (0-dim).
                - If scalar or shape (1,), the same falloff is applied to all targeted instances.
                - If shape (M,), M must match the number of targeted instances.
            env_ids (Sequence[int] | None): Indices of instances to set. If None:
                - For scalar/shape (1,), applies to all instances.
                - For shape (M,), M must equal num_instances, applies per-instance.
        """
        self._apply_scalar(falloffs, env_ids, "set_falloff")

    def set_local_pose(
        self,
        pose: torch.Tensor,
        env_ids: Sequence[int] | None = None,
        to_matrix: bool = False,
    ) -> None:
        """Set local pose (translation) for one or more lights.

        Args:
            pose (torch.Tensor):
                - If to_matrix=False: shape (3,) or (M, 3), representing (x, y, z).
                - If to_matrix=True: shape (4, 4) or (M, 4, 4); translation extracted automatically.
            env_ids (Sequence[int] | None): Indices to set. If None:
                - For vector input (3,) broadcast to all, or (M,3) with M == num_instances.
                - For matrix input (4,4) broadcast to all, or (M,4,4) with M == num_instances.
            to_matrix (bool): Interpret `pose` as full 4x4 matrix if True, else as vector(s).
        """
        if not torch.is_tensor(pose):
            logger.log_error(
                f"set_local_pose requires a torch.Tensor, got {type(pose)}"
            )
            return

        cpu = pose.detach().cpu()
        if to_matrix:
            if cpu.ndim == 2 and cpu.shape == (4, 4):
                trans = cpu[:3, 3]
            elif cpu.ndim == 3 and cpu.shape[1:] == (4, 4):
                trans = cpu[..., 0:3, 3]
            else:
                logger.log_error(
                    f"set_local_pose matrix: expected (4,4) or (N,4,4), got {tuple(cpu.shape)}"
                )
                return
        else:
            trans = cpu  # expect (3,) or (M,3)

        try:
            self._apply_vector3(trans, env_ids, setter_name="set_location")
        except Exception as e:
            logger.log_error(f"set_local_pose: error while applying translation: {e}")

    def get_local_pose(self, to_matrix: bool = False) -> torch.Tensor:
        """Get local pose of each light, either as full matrix or translation vector.

        Args:
            to_matrix (bool, optional): If True, return poses as 4×4 matrices.
                If False, return translations only as (x, y, z). Defaults to False.
        Returns:
            torch.Tensor:
                - If to_matrix=True: Tensor of shape (N, 4, 4), where N == num_instances.
                - If to_matrix=False: Tensor of shape (N, 3), containing translations.
                On error or empty instances, returns an empty tensor with shape (0, 4, 4) or (0, 3) respectively, and logs via logger.log_error.
        """
        mats = []
        for i in range(self.num_instances):
            try:
                mat = self._entities[i].get_local_pose()  # expect numpy (4,4)
                arr = np.array(mat, dtype=np.float32)
                if arr.shape != (4, 4):
                    logger.log_error(
                        f"get_local_pose: unexpected shape {arr.shape} for instance {i}"
                    )
                    return torch.empty(
                        (0, 4, 4) if to_matrix else (0, 3), dtype=torch.float32
                    )
                mats.append(arr)
            except Exception as e:
                logger.log_error(f"get_local_pose: error for instance {i}: {e}")
                return torch.empty(
                    (0, 4, 4) if to_matrix else (0, 3), dtype=torch.float32
                )

        if not mats:
            return torch.empty((0, 4, 4) if to_matrix else (0, 3), dtype=torch.float32)

        stacked = np.stack(mats, axis=0)  # (N,4,4)
        tensor4 = torch.from_numpy(stacked)
        if to_matrix:
            return tensor4
        # else return translations
        return tensor4[:, 0:3, 3].clone()

    def _apply_vector3(
        self,
        tensor: torch.Tensor,
        env_ids: Sequence[int] | None,
        setter_name: str,
    ) -> None:
        """
        Generic helper for 3-element vectors (color, location).
        Expects tensor shape: (3,), or (M,3) with M == num_instances or M == len(env_ids).
        env_ids: Sequence[int] | None
        """
        # Validate tensor type
        if not torch.is_tensor(tensor):
            logger.log_error(
                f"{setter_name} requires a torch.Tensor, got {type(tensor)}"
            )
            return

        cpu = tensor.detach().cpu()
        # Determine target indices
        if env_ids is None:
            all_ids = list(range(self.num_instances))
        else:
            all_ids = list(env_ids)

        # Cases:
        # 1) cpu.ndim == 1 and size == 3: broadcast to all_ids
        if cpu.ndim == 1 and cpu.shape[0] == 3:
            arr = cpu.numpy()
            for i in all_ids:
                getattr(self._entities[i], setter_name)(
                    float(arr[0]), float(arr[1]), float(arr[2])
                )
            return

        # 2) cpu.ndim == 2 and cpu.shape == (num_instances, 3), env_ids None or full
        if cpu.ndim == 2 and cpu.shape == (self.num_instances, 3) and env_ids is None:
            arr_all = cpu.numpy()
            for i in range(self.num_instances):
                getattr(self._entities[i], setter_name)(
                    float(arr_all[i, 0]), float(arr_all[i, 1]), float(arr_all[i, 2])
                )
            return

        # 3) cpu.ndim == 2 and env_ids provided, cpu.shape == (len(env_ids), 3)
        if (
            cpu.ndim == 2
            and env_ids is not None
            and cpu.shape[0] == len(all_ids)
            and cpu.shape[1] == 3
        ):
            arr_sel = cpu.numpy()
            for idx, i in enumerate(all_ids):
                getattr(self._entities[i], setter_name)(
                    float(arr_sel[idx, 0]),
                    float(arr_sel[idx, 1]),
                    float(arr_sel[idx, 2]),
                )
            return

        logger.log_error(
            f"{setter_name}: tensor shape {tuple(cpu.shape)} is invalid for broadcasting "
            f"(expected (3,) or ({self.num_instances},3) or ({len(all_ids)},3))."
        )

    def _apply_scalar(
        self,
        tensor: torch.Tensor,
        env_ids: Sequence[int] | None,
        setter_name: str,
    ) -> None:
        """
        Generic helper for scalar floats (intensity, falloff).
        Accepts tensor shape: () (0-dim), (1,), or (M,) with M == num_instances or M == len(env_ids).
        env_ids: Sequence[int] | None
        """
        if not torch.is_tensor(tensor):
            logger.log_error(
                f"{setter_name} requires a torch.Tensor, got {type(tensor)}"
            )
            return

        cpu = tensor.detach().cpu()
        if env_ids is None:
            all_ids = list(range(self.num_instances))
        else:
            all_ids = list(env_ids)

        # 1) scalar tensor: broadcast
        if cpu.ndim == 0:
            val = float(cpu.item())
            for i in all_ids:
                getattr(self._entities[i], setter_name)(val)
            return

        # 2) 1D tensor:
        if cpu.ndim == 1:
            length = cpu.shape[0]
            arr = cpu.numpy()
            # a) length == num_instances and env_ids None: map one-to-one
            if length == self.num_instances and env_ids is None:
                for i in range(self.num_instances):
                    getattr(self._entities[i], setter_name)(float(arr[i]))
                return
            # b) length == len(env_ids) when env_ids provided: map one-to-one
            if env_ids is not None and length == len(all_ids):
                for idx, i in enumerate(all_ids):
                    getattr(self._entities[i], setter_name)(float(arr[idx]))
                return
            # c) length == 1: broadcast
            if length == 1:
                val = float(arr[0])
                for i in all_ids:
                    getattr(self._entities[i], setter_name)(val)
                return

        logger.log_error(
            f"{setter_name}: tensor shape {tuple(cpu.shape)} is invalid for broadcasting "
            f"(expected scalar, (1,), ({self.num_instances},) or ({len(all_ids)},))."
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.cfg: LightCfg
        self.set_color(torch.as_tensor(self.cfg.color), env_ids=env_ids)
        self.set_intensity(torch.as_tensor(self.cfg.intensity), env_ids=env_ids)
        self.set_falloff(torch.as_tensor(self.cfg.radius), env_ids=env_ids)
        self.set_local_pose(torch.as_tensor(self.cfg.init_pos), env_ids=env_ids)
