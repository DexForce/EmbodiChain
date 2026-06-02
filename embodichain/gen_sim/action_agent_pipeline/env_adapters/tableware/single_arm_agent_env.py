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

from embodichain.gen_sim.action_agent_pipeline.env_adapters.tableware.base_agent_env import (
    BaseAgentEnv,
)

__all__ = ["SingleArmAgentEnv"]


class SingleArmAgentEnv(BaseAgentEnv):
    """Agent mixin for single-arm robots exposed through the right-arm graph slot."""

    def get_states(self):
        # TODO: only support num_env = 1 for now
        self.init_qpos = self.robot.get_qpos().squeeze(0)

        self.single_arm_name = getattr(self, "single_arm_name", "right_arm")
        self.single_eef_name = getattr(self, "single_eef_name", "right_eef")

        self.left_arm_joints = []
        self.left_eef_joints = []
        self.right_arm_joints = list(
            self.robot.get_joint_ids(name=self.single_arm_name)
        )
        self.right_eef_joints = list(
            self.robot.get_joint_ids(name=self.single_eef_name)
        )

        self.left_arm_init_qpos = self.init_qpos.new_empty(0)
        self.left_arm_init_xpos = None
        self.left_arm_base_pose = None

        self.right_arm_init_qpos = self.init_qpos[self.right_arm_joints]
        self.right_arm_init_xpos = self.robot.compute_fk(
            self.right_arm_init_qpos, name=self.single_arm_name, to_matrix=True
        ).squeeze(0)
        self.right_arm_base_pose = self.robot.get_control_part_base_pose(
            self.single_arm_name, to_matrix=True
        ).squeeze(0)

        self.left_arm_current_qpos = self.left_arm_init_qpos
        self.right_arm_current_qpos = self.right_arm_init_qpos
        self.left_arm_current_xpos = self.left_arm_init_xpos
        self.right_arm_current_xpos = self.right_arm_init_xpos

        self.open_state = torch.as_tensor(
            getattr(self, "gripper_open_state", [0.05]),
            dtype=self.init_qpos.dtype,
        ).flatten()
        self.close_state = torch.as_tensor(
            getattr(self, "gripper_close_state", [0.0]),
            dtype=self.init_qpos.dtype,
        ).flatten()
        self.left_arm_current_gripper_state = self.open_state.new_empty(0)
        self.right_arm_current_gripper_state = self.open_state

        self.update_obj_info()

    def get_arm_ik(self, target_xpos, is_left, qpos_seed=None):
        if is_left:
            raise ValueError(
                "SingleArmAgentEnv only supports the right-arm graph slot."
            )
        ret, qpos = self.robot.compute_ik(
            name=self.single_arm_name, pose=target_xpos, joint_seed=qpos_seed
        )
        return ret.all().item(), qpos.squeeze(0)

    def get_arm_fk(self, qpos, is_left):
        if is_left:
            raise ValueError(
                "SingleArmAgentEnv only supports the right-arm graph slot."
            )
        xpos = self.robot.compute_fk(
            name=self.single_arm_name, qpos=torch.as_tensor(qpos), to_matrix=True
        )
        return xpos.squeeze(0)

    def create_demo_action_list(
        self, regenerate=False, recovery=False, *args, **kwargs
    ):
        graph_file_path, compile_kwargs, _ = self.generate_graph_for_actions(
            regenerate=regenerate, recovery=recovery
        )
        public_atomic_kwargs = {
            "use_public_atomic_actions": True,
            "require_public_atomic_actions": False,
            "use_public_grasp_semantics": False,
            "use_public_grasp_action": False,
            "use_public_place_action": True,
            "allow_public_grasp_annotation": False,
            "force_public_grasp_reannotate": False,
        }
        for key in public_atomic_kwargs:
            if key in kwargs:
                public_atomic_kwargs[key] = kwargs[key]
        compile_kwargs.update(public_atomic_kwargs)
        compile_kwargs["interactive_error_injection"] = kwargs.get(
            "interactive_error_injection", False
        )
        return self.compile_agent.act(graph_file_path, **compile_kwargs)
