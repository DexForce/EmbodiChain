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

import math
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch.distributions.normal import Normal

from .policy import Policy

__all__ = ["VLAPolicy", "load_vla_model", "build_vla_policy"]


class VLAPolicy(Policy):
    """Wrap a pretrained DexForceVLA model with the RL Policy interface."""

    def __init__(
        self,
        action_dim: int,
        device: torch.device,
        vla_model: nn.Module,
        instruction: str = "Stack the bowls.",
        inference_horizon: int = 32,
        action_std_init: float = 0.02,
        robot_type: str = "CobotMagic",
        gripper_open_value: float = 0.05,
        gripper_closed_value: float = 0.0,
        action_key_order: Optional[list[str]] = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.instruction = instruction
        self.inference_horizon = inference_horizon
        self.robot_type = robot_type
        self.gripper_open_value = gripper_open_value
        self.gripper_closed_value = gripper_closed_value

        self.vla_model = vla_model.to(self.device)
        self.vla_model.eval()

        self._workspace_root = Path(__file__).resolve().parents[5]
        self._dexechain_root = self._workspace_root / "embodichain"
        if str(self._workspace_root) not in sys.path:
            sys.path.append(str(self._workspace_root))
        if str(self._dexechain_root) not in sys.path:
            sys.path.append(str(self._dexechain_root))

        from dexechain.data.data_engine.indices_unifier import (  # pyright: ignore[reportMissingImports]
            ActionIndicesGenerator,
        )
        from dexechain.data.enum import (  # pyright: ignore[reportMissingImports]
            ActionMode,
            ControlParts,
            EefNormalizer,
            EndEffector,
            JointType,
            Modality,
        )
        from dexechain.data.global_mapping import (
            GlobalMapping,
        )  # pyright: ignore[reportMissingImports]
        from dexechain.lab.gym.utils.gym_utils import (  # pyright: ignore[reportMissingImports]
            get_pk_serial_chain_from_robot_type,
        )
        from dexechain.lab.gym.utils.misc import (  # pyright: ignore[reportMissingImports]
            _data_key_to_control_part,
        )
        from dexechain.utils.utility import (
            get_right_name,
        )  # pyright: ignore[reportMissingImports]

        self.ActionMode = ActionMode
        self.ControlParts = ControlParts
        self.EefNormalizer = EefNormalizer
        self.EndEffector = EndEffector
        self.JointType = JointType
        self.Modality = Modality
        self._data_key_to_control_part = _data_key_to_control_part
        self._get_right_name = get_right_name

        self.indices_generator = ActionIndicesGenerator(self.vla_model.arm_dofs)
        self.global_mapping = GlobalMapping(self.vla_model.arm_dofs)
        self.pk_chain = get_pk_serial_chain_from_robot_type(self.robot_type)

        self.state_history_len = int(self.vla_model.state_history_len)
        self.img_history_size = int(self.vla_model.img_history_size)
        self.state_token_dim = int(self.vla_model.state_token_dim)
        self.camera_used = list(getattr(self.vla_model, "camera_used", []))
        self.action_key_order = self._resolve_action_key_order(action_key_order)
        self.action_dim = sum(
            len(self.indices_generator.get([key])) for key in self.action_key_order
        )
        if action_dim != self.action_dim:
            raise ValueError(
                f"Configured action_dim={action_dim} does not match decoded VLA "
                f"action_dim={self.action_dim} for keys {self.action_key_order}."
            )
        self.full_action_indices = self.indices_generator.get(self.vla_model.output)

        self.log_std = nn.Parameter(
            torch.full(
                (self.action_dim,),
                float(math.log(max(action_std_init, 1e-6))),
                device=self.device,
            )
        )
        self.log_std_min = -5.0
        self.log_std_max = 2.0
        critic_input_dim = self.state_history_len * self.state_token_dim
        self.value_head = nn.Sequential(
            nn.Linear(critic_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        ).to(self.device)

        self._runtime_env = None
        self._runtime_robot = None
        self._state_history: torch.Tensor | None = None
        self._image_history: torch.Tensor | None = None

    def bind_env(self, env) -> None:
        self._runtime_env = env
        if env is None:
            self._runtime_robot = None
            return
        try:
            self._runtime_robot = env.get_wrapper_attr("robot")
        except Exception:
            self._runtime_robot = None

    def _resolve_action_key_order(
        self, action_key_order: Optional[list[str]]
    ) -> list[str]:
        output_keys = list(self.vla_model.output)
        if action_key_order:
            return [key for key in action_key_order if key in output_keys]

        preferred_order = [
            self.ControlParts.LEFT_ARM.value
            + self.ActionMode.RELATIVE.value
            + self.JointType.QPOS.value,
            self.ControlParts.LEFT_ARM.value + self.JointType.QPOS.value,
            self.ControlParts.LEFT_EEF.value + self.EndEffector.GRIPPER.value,
            self.ControlParts.RIGHT_ARM.value
            + self.ActionMode.RELATIVE.value
            + self.JointType.QPOS.value,
            self.ControlParts.RIGHT_ARM.value + self.JointType.QPOS.value,
            self.ControlParts.RIGHT_EEF.value + self.EndEffector.GRIPPER.value,
        ]
        resolved = [key for key in preferred_order if key in output_keys]
        if not resolved:
            raise ValueError(f"No supported VLA outputs found in {output_keys}")
        return resolved

    def _fit_state_value(
        self, key: str, value: torch.Tensor | object, dtype: torch.dtype
    ) -> torch.Tensor:
        tensor = (
            value.to(self.device, dtype=dtype)
            if isinstance(value, torch.Tensor)
            else torch.as_tensor(value, device=self.device, dtype=dtype)
        )
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)

        target_width = len(self.global_mapping.get_indices([key]))
        if tensor.shape[-1] != target_width:
            if target_width == 1:
                tensor = tensor.mean(dim=-1, keepdim=True)
            elif tensor.shape[-1] > target_width:
                tensor = tensor[..., :target_width]
            else:
                raise ValueError(
                    f"State '{key}' width {tensor.shape[-1]} cannot fit target width {target_width}."
                )
        return tensor

    def _normalize_gripper(self, qpos: torch.Tensor, key: str) -> torch.Tensor:
        if self._runtime_robot is not None:
            normalized = self.EefNormalizer.normalize_cobotmagic_gripper(
                qpos, key, is_action=False, robot=self._runtime_robot
            )
            return self._fit_state_value(key, normalized, qpos.dtype).clamp(0.0, 1.0)

        if qpos.dim() >= 2 and qpos.shape[-1] > 1:
            qpos = qpos.mean(dim=-1, keepdim=True)
        denom = max(self.gripper_open_value - self.gripper_closed_value, 1e-6)
        normalized = 1.0 - (qpos - self.gripper_closed_value) / denom
        return self._fit_state_value(key, normalized.clamp(0.0, 1.0), qpos.dtype)

    def _resolve_camera_image(
        self, sensor_obs: TensorDict, camera_name: str
    ) -> torch.Tensor | None:
        if camera_name in sensor_obs:
            return sensor_obs[camera_name]["color"][..., :3].to(self.device)

        for base_camera_name in sensor_obs.keys():
            if (
                self._get_right_name(base_camera_name) == camera_name
                and "color_right" in sensor_obs[base_camera_name]
            ):
                return sensor_obs[base_camera_name]["color_right"][..., :3].to(
                    self.device
                )

        return None

    def _resize_camera_image(self, image: torch.Tensor) -> torch.Tensor:
        target_size = int(getattr(self.vla_model, "img_size", 0) or 0)
        if target_size <= 0:
            return image
        if image.shape[-3:-1] == (target_size, target_size):
            return image

        resized = F.interpolate(
            image.permute(0, 3, 1, 2).float(),
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )
        return resized.permute(0, 2, 3, 1).to(dtype=image.dtype)

    def _extract_current_images(self, observation: TensorDict) -> torch.Tensor:
        sensor_obs = observation["sensor"]
        images = []
        for camera_name in self.camera_used:
            image = self._resolve_camera_image(sensor_obs, camera_name)
            if image is None:
                raise KeyError(f"Camera '{camera_name}' not found in observation.")
            images.append(self._resize_camera_image(image))
        return torch.stack(images, dim=1)

    def _split_qpos(self, qpos: torch.Tensor) -> dict[str, torch.Tensor]:
        arm_dofs_per_side = self.vla_model.arm_dofs // 2
        eef_dofs_total = qpos.shape[-1] - self.vla_model.arm_dofs
        eef_dofs_per_side = max(eef_dofs_total // 2, 0)

        left_arm_end = arm_dofs_per_side
        left_eef_end = left_arm_end + eef_dofs_per_side
        right_arm_end = left_eef_end + arm_dofs_per_side

        return {
            self.ControlParts.LEFT_ARM.value
            + self.JointType.QPOS.value: qpos[:, :left_arm_end],
            self.ControlParts.LEFT_EEF.value
            + self.EndEffector.GRIPPER.value: qpos[:, left_arm_end:left_eef_end],
            self.ControlParts.RIGHT_ARM.value
            + self.JointType.QPOS.value: qpos[:, left_eef_end:right_arm_end],
            self.ControlParts.RIGHT_EEF.value
            + self.EndEffector.GRIPPER.value: qpos[:, right_arm_end:],
        }

    def _build_state_vector(
        self, observation: TensorDict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qpos = observation["robot"][self.JointType.QPOS.value].to(self.device)
        qpos_chunks = self._split_qpos(qpos)
        state_entries: dict[str, torch.Tensor] = {}

        if self._runtime_env is not None and self._runtime_robot is not None:
            control_parts = (
                self._runtime_env.metadata.get("dataset", {})
                .get("robot_meta", {})
                .get("control_parts", [])
            )
            if not control_parts:
                control_parts = [
                    self.ControlParts.LEFT_ARM.value,
                    self.ControlParts.LEFT_EEF.value,
                    self.ControlParts.RIGHT_ARM.value,
                    self.ControlParts.RIGHT_EEF.value,
                ]
            for key in self.vla_model.state_meta:
                part = self._data_key_to_control_part(
                    robot=self._runtime_robot,
                    control_parts=control_parts,
                    data_key=key,
                )
                if part is None:
                    continue
                indices = self._runtime_robot.get_joint_ids(part, remove_mimic=True)
                qpos_data = qpos[:, indices]
                if self.EndEffector.GRIPPER.value in key:
                    state_entries[key] = self._normalize_gripper(qpos_data, key)
                else:
                    normalized = self.EefNormalizer.normalize_eef(
                        qpos_data, part, robot=self._runtime_robot
                    )
                    state_entries[key] = self._fit_state_value(
                        key, normalized, qpos.dtype
                    )
        else:
            state_entries = {
                self.ControlParts.LEFT_ARM.value
                + self.JointType.QPOS.value: qpos_chunks[
                    self.ControlParts.LEFT_ARM.value + self.JointType.QPOS.value
                ],
                self.ControlParts.RIGHT_ARM.value
                + self.JointType.QPOS.value: qpos_chunks[
                    self.ControlParts.RIGHT_ARM.value + self.JointType.QPOS.value
                ],
                self.ControlParts.LEFT_EEF.value
                + self.EndEffector.GRIPPER.value: self._normalize_gripper(
                    qpos_chunks[
                        self.ControlParts.LEFT_EEF.value
                        + self.EndEffector.GRIPPER.value
                    ],
                    self.ControlParts.LEFT_EEF.value + self.EndEffector.GRIPPER.value,
                ),
                self.ControlParts.RIGHT_EEF.value
                + self.EndEffector.GRIPPER.value: self._normalize_gripper(
                    qpos_chunks[
                        self.ControlParts.RIGHT_EEF.value
                        + self.EndEffector.GRIPPER.value
                    ],
                    self.ControlParts.RIGHT_EEF.value + self.EndEffector.GRIPPER.value,
                ),
            }

        if self.pk_chain is not None:
            from dexechain.lab.gym.utils.gym_utils import (  # pyright: ignore[reportMissingImports]
                map_qpos_to_eef_pose,
            )

            arm_dofs_per_side = self.vla_model.arm_dofs // 2
            arm_qpos = torch.cat(
                [
                    qpos_chunks[
                        self.ControlParts.LEFT_ARM.value + self.JointType.QPOS.value
                    ],
                    qpos_chunks[
                        self.ControlParts.RIGHT_ARM.value + self.JointType.QPOS.value
                    ],
                ],
                dim=-1,
            )
            eef_pose_dict = map_qpos_to_eef_pose(
                self.pk_chain,
                arm_qpos.to("cpu"),
                control_parts=[
                    self.ControlParts.LEFT_ARM.value,
                    self.ControlParts.RIGHT_ARM.value,
                ],
                control_ids=[
                    list(range(0, arm_dofs_per_side)),
                    list(range(arm_dofs_per_side, arm_dofs_per_side * 2)),
                ],
            )
            eef_pose_dict = {
                key: (
                    value.to(self.device, dtype=qpos.dtype)
                    if isinstance(value, torch.Tensor)
                    else torch.as_tensor(value, device=self.device, dtype=qpos.dtype)
                )
                for key, value in eef_pose_dict.items()
            }
            state_entries.update(eef_pose_dict)

        state_vector = torch.zeros(
            (qpos.shape[0], self.state_token_dim),
            device=self.device,
            dtype=qpos.dtype,
        )
        state_indicator = torch.zeros_like(state_vector)
        for key in self.vla_model.state_meta:
            if key not in state_entries:
                continue
            indices = self.global_mapping.get_indices([key])
            state_vector[:, indices] = state_entries[key]
            state_indicator[:, indices] = 1
        return state_vector, state_indicator

    def _roll_history(
        self,
        history: torch.Tensor | None,
        current: torch.Tensor,
        history_len: int,
    ) -> torch.Tensor:
        if history is None or history.shape[0] != current.shape[0]:
            return current.unsqueeze(1).repeat(
                [1, history_len] + [1] * (current.dim() - 1)
            )
        if history_len == 1:
            return current.unsqueeze(1)
        return torch.cat([history[:, 1:], current.unsqueeze(1)], dim=1)

    def _build_policy_context(
        self,
        observation: TensorDict,
        update_history: bool,
        cached_context: TensorDict | None = None,
    ) -> tuple[dict[str, torch.Tensor | list[str]], torch.Tensor, TensorDict]:
        current_state, current_state_indicator = self._build_state_vector(observation)
        current_images = self._extract_current_images(observation)

        if cached_context is not None:
            state_history = cached_context["state_history"].to(self.device)
            image_history = cached_context["image_history"].to(self.device)
        else:
            state_history = self._roll_history(
                self._state_history, current_state, self.state_history_len
            )
            image_history = self._roll_history(
                self._image_history, current_images, self.img_history_size
            )
            if update_history:
                self._state_history = state_history.detach().clone()
                self._image_history = image_history.detach().clone()

        state_indicator = current_state_indicator.unsqueeze(1).repeat(
            1, state_history.shape[1], 1
        )
        action_indicator = torch.zeros(
            (
                current_state.shape[0],
                self.inference_horizon,
                self.state_token_dim,
            ),
            device=self.device,
            dtype=current_state.dtype,
        )
        action_indicator[:, :, self.full_action_indices] = 1

        batch = {
            self.Modality.IMAGES.value: image_history,
            self.Modality.STATES.value: state_history,
            self.Modality.STATE_INDICATOR.value: state_indicator,
            self.Modality.ACTION_INDICATOR.value: action_indicator,
            "instruction": [self.instruction] * current_state.shape[0],
        }
        critic_input = state_history.reshape(state_history.shape[0], -1).float()
        context = TensorDict(
            {
                "state_history": state_history.detach(),
                "image_history": image_history.detach(),
            },
            batch_size=[current_state.shape[0]],
            device=self.device,
        )
        return batch, critic_input, context

    def _predict_chunk_actions(
        self, batch: dict[str, torch.Tensor | list[str]]
    ) -> torch.Tensor:
        self.vla_model.eval()
        data = self.vla_model.brain_infer(
            batch,
            action_mask=batch[self.Modality.ACTION_INDICATOR.value],
            precomp_lang_embed=True,
            use_fix_aug=False,
        )
        data = self.vla_model._compute_priviliges(data)
        data = self.vla_model._compute_adaptors(data)
        data = self.vla_model.cerebellum(data, None)

        from dexechain.agents.dexforce_vla.models.utils import (  # pyright: ignore[reportMissingImports]
            post_process,
        )

        data = post_process(
            data,
            is_training=False,
            **self.vla_model.global_collection,
        )
        return data[self.Modality.ACTIONS.value]

    def _decode_first_action(
        self, trajectory: torch.Tensor, observation: TensorDict
    ) -> torch.Tensor:
        first_step = trajectory[:, 0]
        current_qpos = observation["robot"][self.JointType.QPOS.value].to(self.device)
        qpos_chunks = self._split_qpos(current_qpos)
        decoded_parts: list[torch.Tensor] = []

        for key in self.action_key_order:
            indices = self.indices_generator.get([key])
            value = first_step[:, indices]
            if (
                self.ActionMode.RELATIVE.value in key
                and self.JointType.QPOS.value in key
            ):
                if key.startswith(self.ControlParts.LEFT_ARM.value):
                    value = (
                        qpos_chunks[
                            self.ControlParts.LEFT_ARM.value + self.JointType.QPOS.value
                        ]
                        + value
                    )
                elif key.startswith(self.ControlParts.RIGHT_ARM.value):
                    value = (
                        qpos_chunks[
                            self.ControlParts.RIGHT_ARM.value
                            + self.JointType.QPOS.value
                        ]
                        + value
                    )
            elif self.EndEffector.GRIPPER.value in key:
                value = self.gripper_closed_value + (1.0 - value) * (
                    self.gripper_open_value - self.gripper_closed_value
                )
            decoded_parts.append(value)

        if not decoded_parts:
            raise ValueError(
                f"No action keys could be decoded from model outputs: {self.vla_model.output}"
            )
        return torch.cat(decoded_parts, dim=-1).to(self.device)

    def _expand_env_action(
        self, action: torch.Tensor, observation: TensorDict
    ) -> torch.Tensor:
        expanded_parts: list[torch.Tensor] = []
        offset = 0
        for key in self.action_key_order:
            width = len(self.indices_generator.get([key]))
            value = action[:, offset : offset + width]
            offset += width

            if (
                self._runtime_robot is not None
                and self.EndEffector.GRIPPER.value in key
                and value.shape[-1] == 1
            ):
                value = self.EefNormalizer.denormalize_cobotmagic_gripper(
                    value, key, robot=self._runtime_robot
                )
                value = (
                    value.to(self.device, dtype=action.dtype)
                    if isinstance(value, torch.Tensor)
                    else torch.as_tensor(value, device=self.device, dtype=action.dtype)
                )
                if value.dim() == 1:
                    value = value.unsqueeze(0)
                control_part = key.replace(self.EndEffector.GRIPPER.value, "")
                target_dim = len(
                    self._runtime_robot.get_joint_ids(control_part, remove_mimic=False)
                )
                if target_dim > value.shape[-1]:
                    value = value.repeat(1, target_dim)

            expanded_parts.append(value)

        return torch.cat(expanded_parts, dim=-1).to(self.device)

    def _action_stats(
        self,
        mean_action: torch.Tensor,
        deterministic: bool,
        provided_action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp().expand(mean_action.shape[0], -1)
        dist = Normal(mean_action, std)
        if provided_action is not None:
            action = provided_action
        elif deterministic:
            action = mean_action
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return action, log_prob, entropy

    @torch.no_grad()
    def forward(
        self, tensordict: TensorDict, deterministic: bool = False
    ) -> TensorDict:
        observation = tensordict["observation"]
        batch, critic_input, context = self._build_policy_context(
            observation, update_history=True
        )
        trajectory = self._predict_chunk_actions(batch)
        mean_action = self._decode_first_action(trajectory, observation)
        action, log_prob, _ = self._action_stats(mean_action, deterministic)
        tensordict["action"] = action
        tensordict["env_action"] = self._expand_env_action(action, observation)
        tensordict["sample_log_prob"] = log_prob
        tensordict["value"] = self.value_head(critic_input)
        tensordict["policy_context"] = context
        tensordict["loc"] = mean_action
        tensordict["scale"] = (
            self.log_std.clamp(self.log_std_min, self.log_std_max)
            .exp()
            .expand_as(mean_action)
        )
        return tensordict

    @torch.no_grad()
    def get_value(self, tensordict: TensorDict) -> TensorDict:
        _, critic_input, _ = self._build_policy_context(
            tensordict["observation"], update_history=False
        )
        tensordict["value"] = self.value_head(critic_input)
        return tensordict

    def evaluate_actions(self, tensordict: TensorDict) -> TensorDict:
        observation = tensordict["observation"]
        actions = tensordict["action"]
        context = tensordict.get("policy_context", None)
        batch, critic_input, _ = self._build_policy_context(
            observation, update_history=False, cached_context=context
        )
        trajectory = self._predict_chunk_actions(batch)
        mean_action = self._decode_first_action(trajectory, observation)
        _, log_prob, entropy = self._action_stats(
            mean_action, deterministic=False, provided_action=actions
        )
        tensordict["sample_log_prob"] = log_prob
        tensordict["entropy"] = entropy
        tensordict["value"] = self.value_head(critic_input)
        return tensordict


def load_vla_model(
    model_path: str,
    model_class: Optional[str] = None,
    model_config: Optional[dict] = None,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Load a pretrained DexForceVLA-compatible model."""
    workspace_root = Path(__file__).resolve().parents[5]
    dexechain_root = workspace_root / "embodichain"
    if str(workspace_root) not in sys.path:
        sys.path.append(str(workspace_root))
    if str(dexechain_root) not in sys.path:
        sys.path.append(str(dexechain_root))

    model_config = {} if model_config is None else dict(model_config)
    torch_dtype_name = model_config.pop("torch_dtype", "float32")
    weight_dtype = getattr(torch, torch_dtype_name)

    if model_class:
        module_name, class_name = model_class.rsplit(".", 1)
        module = __import__(module_name, fromlist=[class_name])
        model_cls = getattr(module, class_name)
        return model_cls.from_pretrained(model_path, dtype=weight_dtype).to(device)

    from dexechain.agents.dexforce_vla.models.dexforcevla_runner import (  # pyright: ignore[reportMissingImports]
        DexForceVLA,
    )

    return DexForceVLA.from_pretrained(model_path, dtype=weight_dtype).to(device)


def build_vla_policy(
    policy_block: dict,
    action_dim: int,
    device: torch.device,
) -> VLAPolicy:
    """Build a VLAPolicy from configuration."""
    vla_config = policy_block.get("vla_config")
    if vla_config is None:
        raise ValueError("VLA policy requires 'vla_config' in policy block")

    model_path = vla_config.get("model_path")
    if model_path is None:
        raise ValueError("VLA config requires 'model_path'")

    vla_model = load_vla_model(
        model_path=model_path,
        model_class=vla_config.get("model_class"),
        model_config=dict(vla_config.get("model_config", {})),
        device=device,
    )
    return VLAPolicy(
        action_dim=action_dim,
        device=device,
        vla_model=vla_model,
        instruction=vla_config.get("instruction", "Stack the bowls."),
        inference_horizon=int(vla_config.get("inference_horizon", 32)),
        action_std_init=float(vla_config.get("action_std_init", 0.02)),
        robot_type=vla_config.get("robot_type", "CobotMagic"),
        gripper_open_value=float(vla_config.get("gripper_open_value", 0.05)),
        gripper_closed_value=float(vla_config.get("gripper_closed_value", 0.0)),
        action_key_order=vla_config.get("action_key_order"),
    )
