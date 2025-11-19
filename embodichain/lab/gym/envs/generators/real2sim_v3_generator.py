# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------
import os
import torch
import numpy as np
from typing import List, Dict, Union

from embodichain import embodichain_dir
from embodichain.utils.utility import save_json
from embodichain.utils.logger import log_info, log_error, log_warning
from embodichain.data.enum import ControlParts, JointType, EndEffector
from embodichain.lab.gym.envs.generators.reality_parser import (
    RealityParser,
    CANONICAL_ID,
)
from embodichain.lab.gym.envs.managers.randomization import get_random_pose
from embodichain.lab.gym.utils.registration import register_env_function
from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
from embodichain.lab.gym.utils.gym_utils import config_to_cfg

from embodichain.lab.gym.envs.action_bank.configurable_action import (
    attach_action_bank,
    attach_node_and_edge,
    GeneralActionBank,
    ActionBankMimic,
    ActionBank,
    get_func_tag,
)


def create_demo_action_list(env, mimic: bool = True, vis=False, *args, **kwargs):
    env.action_banks: List[ActionBank]
    random_id = np.random.choice(len(env.action_banks))

    if mimic:
        # NOTE: hard code for bottle object
        fix_obj_name = ["cup"]
        for obj_cfg in env.cfg.rigid_object:
            if obj_cfg.uid in fix_obj_name:
                continue
            obj = env.sim.get_asset(obj_cfg.uid)
            new_obj_pose = get_random_pose(
                init_pos=obj.get_local_pose(to_matrix=True)[:, :3, 3],
                init_rot=obj.get_local_pose(to_matrix=True)[:, :3, :3],
                position_range=[[-0.03, -0.03, 0.0], [0.03, 0.03, 0.0]],
                relative_position=True,
            )
            obj.set_local_pose(new_obj_pose)
        env.sim.update(step=100)

        new_bank = ActionBankMimic(env.action_banks).mimic(id=random_id)
        modified_path = os.path.join(
            os.path.dirname(embodichain_dir),
            "test_configs",
            "action_config_modified.json",
        )
        save_json(modified_path, new_bank.conf)
    else:
        new_bank = env.action_banks[random_id]
    graph_compose, tasks_data, taskkey2index = new_bank.parse_network(
        get_func_tag("node").functions[new_bank.__class__.__name__],
        get_func_tag("edge").functions[new_bank.__class__.__name__],
        vis_graph=vis,
    )
    package = new_bank.gantt(tasks_data, taskkey2index, vis=vis)
    ret = new_bank.create_action_list(
        env,
        graph_compose,
        package,
        trajectory=env.trajectory,
        canonical_trajectory=env.canonical_trajectory,
    )

    if ret is None:
        log_warning(
            f"Failed to create action list for Env: {type(env).__name__}, Task Type: {env.metadata['task_type']}."
        )
        return None

    all_dim = env.robot.get_qpos().shape[-1]
    total_traj_num = ret[list(ret.keys())[0]].shape[-1]

    left_arm_joints = env.robot.get_joint_ids(
        name=ControlParts.LEFT_ARM.value
    )  # TODO: hardcode
    right_arm_joints = env.robot.get_joint_ids(
        name=ControlParts.RIGHT_ARM.value
    )  # TODO: hardcode
    # TODO: 这里需要修改，因为qpos_new是全0的，所以需要修改
    qpos_new = np.zeros((total_traj_num, all_dim), dtype=np.float32)
    qpos_new[:, left_arm_joints] = ret[
        ControlParts.LEFT_ARM.value + JointType.QPOS.value
    ].T
    qpos_new[:, right_arm_joints] = ret[
        ControlParts.RIGHT_ARM.value + JointType.QPOS.value
    ].T
    qpos_new = qpos_new.astype(np.float32)

    log_info(f"Total generated trajectory number: {total_traj_num}.", color="green")

    ee_state_list_left = ret[
        ControlParts.LEFT_EEF.value + EndEffector.DEXTROUSHAND.value
    ].T
    ee_state_list_right = ret[
        ControlParts.RIGHT_EEF.value + EndEffector.DEXTROUSHAND.value
    ].T

    # FIXME: TODO: A specific robot cfg for mapping
    from embodichain.lab.gym.utils.misc import map_ee_state_to_env_actions

    qpos_new = map_ee_state_to_env_actions(
        env.robot,
        np.hstack((np.array(ee_state_list_left), np.array(ee_state_list_right))),
        qpos_new,
    )

    actions = []
    for i in range(total_traj_num):
        actions.append(qpos_new[None, i])
    return actions


class Real2SimV3EnvGenerator:
    """Real2Sim environment generator based on V3 (EmbodiedEnv).

    Features:
    - Dynamically creates a class that inherits from V3 `EmbodiedEnv` using the
      parsed results from `RealityParser`.
    - Registers the environment ID with `RealityParser.get_env_name()` (e.g.,
      `pour_water_single_real2sim`).
    - Registers `trajectory`, `canonical_trajectory`, `action_config`, and
      `functions_dict` as default kwargs for environment construction.
    """

    @staticmethod
    def build_env(reality_config: Union[Dict, RealityParser]) -> type:
        rp = (
            RealityParser(reality_config)
            if not isinstance(reality_config, RealityParser)
            else reality_config
        )

        env_class_name = rp.get_task_name() + "Real2SimEnvV3"

        # Construct a new class derived from V3 EmbodiedEnv and attach a dummy create_demo_action_list
        real2sim_env_v3 = type(
            env_class_name,
            (EmbodiedEnv,),
            {
                "create_demo_action_list": create_demo_action_list,
                "trajectory": rp.get_all_trajectory(),
                "canonical_trajectory": rp.get_trajectory(CANONICAL_ID),
            },
        )

        # Simplify to placeholders without action bank (can be enabled downstream if needed):
        list_action_config, list_functions_dict = rp.get_action_config_v3()
        for action_config, functions_dict in zip(
            list_action_config, list_functions_dict
        ):
            bank = attach_node_and_edge(GeneralActionBank, functions_dict)
            attach_action_bank(real2sim_env_v3, bank, action_config=action_config)

        gym_conf = rp.compose_gym_config_v3()
        config_for_cfg = {
            "id": rp.get_env_name(),
            "max_episodes": gym_conf.get("max_episodes", 1),
            "env": gym_conf.get("env", {}),
            "robot": rp.get_robot_config(),
            "sensor": gym_conf.get("sensor", []),
            "light": gym_conf.get("light", []),
            "background": gym_conf.get("background", []),
            "rigid_object": gym_conf.get("rigid_object", []),
            "rigid_object_group": gym_conf.get("rigid_object_group", []),
            "articulation": gym_conf.get("articulation", []),
        }
        cfg = config_to_cfg(config_for_cfg)
        register_env_function(
            real2sim_env_v3,
            rp.get_env_name(),
            cfg=cfg,
            action_config=action_config,
            functions_dict=functions_dict,
        )

        return real2sim_env_v3
