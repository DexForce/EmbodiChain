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

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, List, Sequence
from dexsim.render import Light as _Light
from embodichain.lab.sim.cfg import LightCfg
from embodichain.lab.sim.common import BatchEntity
from embodichain.utils import logger

if TYPE_CHECKING:
    from dexsim.models import MeshObject


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

    def set_direction(
        self, directions: torch.Tensor, env_ids: Sequence[int] | None = None
    ) -> None:
        """Set direction for directional-type lights.

        Only applies to ``sun``, ``direction``, ``spot``, ``rect``, and ``mesh``
        light types. Logs a warning and no-ops for other types.

        Args:
            directions (torch.Tensor): Tensor of shape (3,) or (M, 3), representing
                (x, y, z) direction vectors.
            env_ids (Sequence[int] | None): Indices of instances to set. If None:
                - For shape (3,), applies to all instances.
                - For shape (M, 3), M must equal num_instances, applies per-instance.
        """
        if self.cfg.light_type not in ("sun", "direction", "spot", "rect", "mesh"):
            logger.warning(
                f"set_direction not applicable to light type "
                f"'{self.cfg.light_type}', ignoring."
            )
            return
        self._apply_vector3(directions, env_ids, "set_direction")

    def set_spot_angle(
        self,
        inner_angles: torch.Tensor,
        outer_angles: torch.Tensor,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Set inner and outer cone angles for spot lights.

        Only applies to ``spot`` light type. Logs a warning and no-ops for other types.

        Args:
            inner_angles (torch.Tensor): Tensor of shape () (0-dim), (1,), or (M,)
                representing inner cone angle in degrees.
            outer_angles (torch.Tensor): Tensor of shape () (0-dim), (1,), or (M,)
                representing outer cone angle in degrees.
            env_ids (Sequence[int] | None): Indices of instances to set.
        """
        if self.cfg.light_type != "spot":
            logger.warning(
                f"set_spot_angle not applicable to light type "
                f"'{self.cfg.light_type}', ignoring."
            )
            return

        if not torch.is_tensor(inner_angles) or not torch.is_tensor(outer_angles):
            logger.log_error("set_spot_angle requires torch.Tensor arguments")
            return

        inner_cpu = inner_angles.detach().cpu()
        outer_cpu = outer_angles.detach().cpu()

        if env_ids is None:
            all_ids = list(range(self.num_instances))
        else:
            all_ids = list(env_ids)

        # Both scalar (0-dim): broadcast to all_ids
        if inner_cpu.ndim == 0 and outer_cpu.ndim == 0:
            iv = float(inner_cpu.item())
            ov = float(outer_cpu.item())
            for i in all_ids:
                self._entities[i].set_spot_angle(iv, ov)
            return

        # Both 1D with matching length
        if inner_cpu.ndim == 1 and outer_cpu.ndim == 1:
            ilen, olen = inner_cpu.shape[0], outer_cpu.shape[0]
            inner_arr = inner_cpu.numpy()
            outer_arr = outer_cpu.numpy()

            if ilen == olen == self.num_instances and env_ids is None:
                for i in range(self.num_instances):
                    self._entities[i].set_spot_angle(
                        float(inner_arr[i]), float(outer_arr[i])
                    )
                return

            if env_ids is not None and ilen == olen == len(all_ids):
                for idx, i in enumerate(all_ids):
                    self._entities[i].set_spot_angle(
                        float(inner_arr[idx]), float(outer_arr[idx])
                    )
                return

            # length-1 broadcast
            if ilen == olen == 1:
                iv = float(inner_arr[0])
                ov = float(outer_arr[0])
                for i in all_ids:
                    self._entities[i].set_spot_angle(iv, ov)
                return

        logger.log_error(
            f"set_spot_angle: invalid tensor shapes "
            f"inner={tuple(inner_cpu.shape)}, outer={tuple(outer_cpu.shape)}"
        )

    def set_rect_wh(
        self,
        widths: torch.Tensor,
        heights: torch.Tensor,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Set width and height for rectangular area lights.

        Only applies to ``rect`` light type. Logs a warning and no-ops for other types.

        Args:
            widths (torch.Tensor): Tensor of shape () (0-dim), (1,), or (M,)
                representing width of the rectangular light.
            heights (torch.Tensor): Tensor of shape () (0-dim), (1,), or (M,)
                representing height of the rectangular light.
            env_ids (Sequence[int] | None): Indices of instances to set.
        """
        if self.cfg.light_type != "rect":
            logger.warning(
                f"set_rect_wh not applicable to light type "
                f"'{self.cfg.light_type}', ignoring."
            )
            return

        if not torch.is_tensor(widths) or not torch.is_tensor(heights):
            logger.log_error("set_rect_wh requires torch.Tensor arguments")
            return

        w_cpu = widths.detach().cpu()
        h_cpu = heights.detach().cpu()

        if env_ids is None:
            all_ids = list(range(self.num_instances))
        else:
            all_ids = list(env_ids)

        # Both scalar (0-dim): broadcast to all_ids
        if w_cpu.ndim == 0 and h_cpu.ndim == 0:
            wv = float(w_cpu.item())
            hv = float(h_cpu.item())
            for i in all_ids:
                self._entities[i].set_rect_wh(wv, hv)
            return

        # Both 1D with matching length
        if w_cpu.ndim == 1 and h_cpu.ndim == 1:
            wlen, hlen = w_cpu.shape[0], h_cpu.shape[0]
            w_arr = w_cpu.numpy()
            h_arr = h_cpu.numpy()

            if wlen == hlen == self.num_instances and env_ids is None:
                for i in range(self.num_instances):
                    self._entities[i].set_rect_wh(float(w_arr[i]), float(h_arr[i]))
                return

            if env_ids is not None and wlen == hlen == len(all_ids):
                for idx, i in enumerate(all_ids):
                    self._entities[i].set_rect_wh(float(w_arr[idx]), float(h_arr[idx]))
                return

            # length-1 broadcast
            if wlen == hlen == 1:
                wv = float(w_arr[0])
                hv = float(h_arr[0])
                for i in all_ids:
                    self._entities[i].set_rect_wh(wv, hv)
                return

        logger.log_error(
            f"set_rect_wh: invalid tensor shapes "
            f"width={tuple(w_cpu.shape)}, height={tuple(h_cpu.shape)}"
        )

    def set_mesh(
        self,
        mesh: "MeshObject",
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Set the mesh for mesh-type lights.

        Only applies to ``mesh`` light type. Logs a warning and no-ops for other types.
        This is NOT tensor-batched — the same MeshObject is assigned to all targeted
        instances.

        Args:
            mesh (MeshObject): The mesh object to assign to the light.
            env_ids (Sequence[int] | None): Indices of instances to set. If None,
                applies to all instances.
        """
        if self.cfg.light_type != "mesh":
            logger.warning(
                f"set_mesh not applicable to light type "
                f"'{self.cfg.light_type}', ignoring."
            )
            return

        if env_ids is None:
            target_ids = list(range(self.num_instances))
        else:
            target_ids = list(env_ids)

        for i in target_ids:
            try:
                self._entities[i].set_mesh(mesh)
            except Exception as e:
                logger.log_error(f"set_mesh: error for instance {i}: {e}")

    def enable_shadow(
        self,
        flags: torch.Tensor,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Enable or disable shadow casting.

        Applies to all light types.

        Args:
            flags (torch.Tensor): Boolean tensor of shape () (0-dim), (1,), or (M,).
                Non-zero values enable shadows; zero disables.
            env_ids (Sequence[int] | None): Indices of instances to set.
        """
        if not torch.is_tensor(flags):
            logger.log_error(
                f"enable_shadow requires a torch.Tensor, got {type(flags)}"
            )
            return

        cpu = flags.detach().cpu()
        if env_ids is None:
            all_ids = list(range(self.num_instances))
        else:
            all_ids = list(env_ids)

        # Scalar: broadcast
        if cpu.ndim == 0:
            val = bool(cpu.item() != 0)
            for i in all_ids:
                self._entities[i].set_shadow(val)
            return

        # 1D tensor
        if cpu.ndim == 1:
            length = cpu.shape[0]
            arr = cpu.numpy()
            if length == self.num_instances and env_ids is None:
                for i in range(self.num_instances):
                    self._entities[i].set_shadow(bool(arr[i] != 0))
                return
            if env_ids is not None and length == len(all_ids):
                for idx, i in enumerate(all_ids):
                    self._entities[i].set_shadow(bool(arr[idx] != 0))
                return
            if length == 1:
                val = bool(arr[0] != 0)
                for i in all_ids:
                    self._entities[i].set_shadow(val)
                return

        logger.log_error(
            f"enable_shadow: tensor shape {tuple(cpu.shape)} is invalid for broadcasting"
        )

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
        if self.cfg.light_type == "direction":
            logger.warning(
                "set_local_pose not applicable to 'direction' light type "
                "(infinite distance, direction only). Use set_direction() instead."
            )
            return

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
        """Reset the light to its initial configuration state.

        Applies only the properties relevant to ``self.cfg.light_type``.

        Args:
            env_ids (Sequence[int] | None): The environment IDs to reset.
                If None, resets all environments.
        """
        self.cfg: LightCfg
        light_type = self.cfg.light_type

        # Universal properties
        self.set_color(torch.as_tensor(self.cfg.color), env_ids=env_ids)
        self.set_intensity(torch.as_tensor(self.cfg.intensity), env_ids=env_ids)
        self.enable_shadow(
            torch.as_tensor(float(self.cfg.enable_shadow)), env_ids=env_ids
        )

        # Position (all types except direction)
        if light_type != "direction":
            self.set_local_pose(torch.as_tensor(self.cfg.init_pos), env_ids=env_ids)

        # Point light: falloff
        if light_type == "point":
            self.set_falloff(torch.as_tensor(self.cfg.radius), env_ids=env_ids)

        # Directional types: direction vector
        if light_type in ("sun", "direction", "spot", "rect", "mesh"):
            self.set_direction(torch.as_tensor(self.cfg.direction), env_ids=env_ids)

        # Spot light: cone angles
        if light_type == "spot":
            self.set_spot_angle(
                torch.as_tensor(self.cfg.spot_angle_inner),
                torch.as_tensor(self.cfg.spot_angle_outer),
                env_ids=env_ids,
            )

        # Rect light: dimensions
        if light_type == "rect":
            self.set_rect_wh(
                torch.as_tensor(self.cfg.rect_width),
                torch.as_tensor(self.cfg.rect_height),
                env_ids=env_ids,
            )

        # Mesh light: mesh_path is stored in cfg but actual mesh assignment
        # is done via set_mesh() which requires a MeshObject.
        # Sun-specific angular_radius/halo are reserved for future backend support.
