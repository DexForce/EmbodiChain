# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

from __future__ import annotations

from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.utils.registration import register_env

__all__ = ["RearrangementEnv"]


@register_env("Rearrangement-v3", max_episode_steps=600)
class RearrangementEnv(EmbodiedEnv):
    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        super().__init__(cfg, **kwargs)

        action_config = kwargs.get("action_config", None)
        if action_config is not None:
            self.action_config = action_config

    def is_task_success(self) -> bool:
        fork = self.sim.get_rigid_object("fork")
        spoon = self.sim.get_rigid_object("spoon")
        plate = self.sim.get_rigid_object("plate")
        plate_pose = plate.get_local_pose(to_matrix=True)
        # TODO: now only for 1 env
        (
            spoon_place_target_x,
            spoon_place_target_y,
            spoon_place_target_z,
        ) = self.affordance_datas["spoon_place_pose"][:3, 3]
        (
            fork_place_target_x,
            fork_place_target_y,
            fork_place_target_z,
        ) = self.affordance_datas["fork_place_pose"][:3, 3]

        spoon_pose = spoon.get_local_pose(to_matrix=True)
        spoon_x, spoon_y, spoon_z = spoon_pose[0, :3, 3]

        fork_pose = fork.get_local_pose(to_matrix=True)
        fork_x, fork_y, fork_z = fork_pose[0, :3, 3]

        tolerance = self.metadata.get("success_params", {}).get("tolerance", 0.02)

        # spoon and fork should with the x y range of tolerance related to plate.
        return ~(
            abs(spoon_x - spoon_place_target_x) > tolerance
            or abs(spoon_y - spoon_place_target_y) > tolerance
            or abs(fork_x - fork_place_target_x) > tolerance
            or abs(fork_y - fork_place_target_y) > tolerance
        )
