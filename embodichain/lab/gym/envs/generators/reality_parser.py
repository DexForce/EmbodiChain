from embodichain.utils.utility import load_json
from embodichain.lab.gym.envs.generators.config_template import (
    ConfigTemplate,
)
from embodichain.lab.gym.envs.action_bank.configurable_action import (
    generate_trajectory_qpos,
    to_affordance_node_func,
    get_init_affordance,
)
from embodichain.lab.gym.envs.managers import SceneEntityCfg
import numpy as np
from typing import Dict, Tuple, List
import os
from copy import deepcopy
from embodichain.utils.logger import log_warning, log_info
from functools import partial

CANONICAL_ID = "0"


def get_edge_by_name(edge_dict: Dict, name: str, tag: str = "src") -> List[str]:
    ret = []
    for scope in edge_dict:
        for edge in edge_dict[scope]:
            for key, val in edge.items():
                if val[tag] == name:
                    ret.append(key)
    if len(ret) == 0:
        log_warning(f"No edge found with {tag} name {name}.")
    return ret


def simple_mapping(scope_names):
    from embodichain.data.enum import ControlParts, JointType, EndEffector
    import re

    # Dynamically generate mapping rules for ControlParts
    scope_mapping = {
        ControlParts.LEFT_ARM.value: f"{ControlParts.LEFT_ARM.value}{JointType.QPOS.value}",
        ControlParts.RIGHT_ARM.value: f"{ControlParts.RIGHT_ARM.value}{JointType.QPOS.value}",
        ControlParts.LEFT_EEF.value: f"{ControlParts.LEFT_EEF.value}{EndEffector.DEXTROUSHAND.value}",
        ControlParts.RIGHT_EEF.value: f"{ControlParts.RIGHT_EEF.value}{EndEffector.DEXTROUSHAND.value}",
    }

    def get_scope_key(name: str) -> str:
        control_parts_values = "|".join(re.escape(part.value) for part in ControlParts)
        pattern = rf"^({control_parts_values})"

        match = re.match(pattern, name)
        if match:
            return match.group(1)
        return name

    return {
        scope_name: scope_mapping.get(get_scope_key(scope_name), scope_name)
        for scope_name in scope_names
    }


def real2sim_affordance(
    cfg_list: List,
    len_trajectory: int,
    scope: str,
    canonical_cfg_list: List,
    object_config: Dict = {},
    id: str = "",
    gather_index: List[int] = [],
) -> Tuple[Dict, Dict]:
    # NOTE: need to keep consistent with modify_action_config_node
    functions = {}
    nodes = []

    canonical_cfg_dict = {}
    [canonical_cfg_dict.update(cfg) for cfg in canonical_cfg_list]
    canonical_cfg_dict = {
        val["affordance_name"]: val for key, val in canonical_cfg_dict.items()
    }
    object_config = {val["name"]: val for _, val in object_config.items()}
    tl = len_trajectory - 1
    for cfg in cfg_list:
        for key, val in cfg.items():
            functions[to_affordance_node_func(val["affordance_name"])] = partial(
                generate_trajectory_qpos,
                agent_uid=scope,
                affordance_name=val["affordance_name"],
            )
            temp = {}

            cononical_val = canonical_cfg_dict[val["affordance_name"]]
            if val.get("mimicable", False):
                assert (
                    val.get("slaver", "") != ""
                ), f"Affordance {val['affordance_name']} is mimicable but slaver is not specified."
                slaver = val.get("slaver", "")
                canonical_pose = object_config[slaver]["pose"]
            else:
                canonical_pose = []
                slaver = ""
            assert val["timestep"] != 0, "Timestep cannot be 0."
            assert val["timestep"] != tl, "Timestep cannot be {}.".format(tl)

            temp[val["affordance_name"]] = {
                "name": to_affordance_node_func(val["affordance_name"]),
                "mimicable": val.get("mimicable", False),
                "kwargs": {
                    "trajectory_index": val["timestep"],
                    "canonical_trajectory_index": cononical_val["timestep"],
                    "canonical_pose": canonical_pose,
                    "slaver": slaver,
                    "trajectory_id": id,
                    "gather_index": gather_index,
                },
            }
            nodes.append(temp)

    init_qpos = get_init_affordance(scope)
    functions["generate_{}".format(init_qpos)] = partial(
        generate_trajectory_qpos,
        agent_uid=scope,
        affordance_name=init_qpos,
    )
    temp = {}
    temp[init_qpos] = {
        "name": "generate_{}".format(init_qpos),
        "kwargs": {
            "trajectory_index": 0,
            "trajectory_id": id,
            "gather_index": gather_index,
        },
    }
    nodes.append(temp)

    end_qpos = get_init_affordance(scope, "end")
    functions["generate_{}".format(end_qpos)] = partial(
        generate_trajectory_qpos,
        agent_uid=scope,
        affordance_name=end_qpos,
    )
    temp = {}
    temp[end_qpos] = {
        "name": "generate_{}".format(end_qpos),
        "kwargs": {
            "trajectory_index": tl,
            "trajectory_id": id,
            "gather_index": gather_index,
        },
    }
    nodes.append(temp)

    return nodes, functions


def real2sim_affordance_v3(
    cfg_list: List,
    len_trajectory: int,
    scope: str,
    canonical_cfg_list: List,
    object_config: Dict = {},
    id: str = "",
    gather_index: List[int] = [],
) -> Tuple[Dict, Dict]:
    # NOTE: need to keep consistent with modify_action_config_node
    functions = {}
    nodes = []

    canonical_cfg_dict = {}
    [canonical_cfg_dict.update(cfg) for cfg in canonical_cfg_list]
    canonical_cfg_dict = {
        val["affordance_name"]: val for key, val in canonical_cfg_dict.items()
    }
    object_config = {val["uid"]: val for val in object_config}
    tl = len_trajectory - 1
    for cfg in cfg_list:
        for key, val in cfg.items():
            functions[to_affordance_node_func(val["affordance_name"])] = partial(
                generate_trajectory_qpos,
                agent_uid=scope,
                affordance_name=val["affordance_name"],
            )
            temp = {}

            cononical_val = canonical_cfg_dict[val["affordance_name"]]
            if val.get("mimicable", False):
                assert (
                    val.get("slaver", "") != ""
                ), f"Affordance {val['affordance_name']} is mimicable but slaver is not specified."
                slaver = val.get("slaver", "")
                canonical_pose = object_config[slaver]["init_local_pose"]
            else:
                canonical_pose = []
                slaver = ""
            assert val["timestep"] != 0, "Timestep cannot be 0."
            assert val["timestep"] != tl, "Timestep cannot be {}.".format(tl)

            temp[val["affordance_name"]] = {
                "name": to_affordance_node_func(val["affordance_name"]),
                "mimicable": val.get("mimicable", False),
                "kwargs": {
                    "trajectory_index": val["timestep"],
                    "canonical_trajectory_index": cononical_val["timestep"],
                    "canonical_pose": canonical_pose,
                    "slaver": slaver,
                    "trajectory_id": id,
                    "gather_index": gather_index,
                },
            }
            nodes.append(temp)

    init_qpos = get_init_affordance(scope)
    functions["generate_{}".format(init_qpos)] = partial(
        generate_trajectory_qpos,
        agent_uid=scope,
        affordance_name=init_qpos,
    )
    temp = {}
    temp[init_qpos] = {
        "name": "generate_{}".format(init_qpos),
        "kwargs": {
            "trajectory_index": 0,
            "trajectory_id": id,
            "gather_index": gather_index,
        },
    }
    nodes.append(temp)

    end_qpos = get_init_affordance(scope, "end")
    functions["generate_{}".format(end_qpos)] = partial(
        generate_trajectory_qpos,
        agent_uid=scope,
        affordance_name=end_qpos,
    )
    temp = {}
    temp[end_qpos] = {
        "name": "generate_{}".format(end_qpos),
        "kwargs": {
            "trajectory_index": tl,
            "trajectory_id": id,
            "gather_index": gather_index,
        },
    }
    nodes.append(temp)

    return nodes, functions


def auto_link(
    cfg_list: List,
    len_trajectory: int,
    scope: str,
    sample_ratio: float,
    id: str,
    gather_index: List[int],
) -> Tuple[Dict, Dict]:
    from embodichain.data.enum import ControlParts

    timesteps = {}
    basket = {}
    for cfg in cfg_list:
        for key, val in cfg.items():
            timesteps[val["timestep"]] = key
            basket[val["timestep"]] = val
    end_timestep = len_trajectory - 1

    basket[0] = {"affordance_name": get_init_affordance(scope)}

    if ControlParts.LEFT_EEF.value in scope or ControlParts.RIGHT_EEF.value in scope:
        timestep_order = list(timesteps.keys())
    else:
        timestep_order = list(timesteps.keys()) + [end_timestep]
        basket[end_timestep] = {
            "affordance_name": get_init_affordance(scope, "end"),
            "trajectory": {
                "name": "load_trajectory",
                "kwargs": {"trajectory_id": id, "gather_index": gather_index},
            },
        }
    indices = np.argsort(timestep_order)

    edges = []
    for t, idx in enumerate(list(indices)):
        if t == 0:
            duration = timestep_order[idx]
            start_t = 0
            end_t = timestep_order[idx]
        else:
            duration = timestep_order[idx] - timestep_order[indices[t - 1]]
            start_t = timestep_order[indices[t - 1]]
            end_t = timestep_order[idx]

        temp = {}
        if (
            ControlParts.LEFT_EEF.value in scope
            or ControlParts.RIGHT_EEF.value in scope
        ):
            kwargs = deepcopy(basket[end_t]["trajectory"]["kwargs"])
            kwargs.update({"trajectory_id": id, "gather_index": gather_index})
            temp[
                "{}_to_{}".format(
                    basket[start_t]["affordance_name"],
                    basket[end_t]["affordance_name"],
                )
            ] = {
                "src": basket[start_t]["affordance_name"],
                "sink": basket[end_t]["affordance_name"],
                "name": basket[end_t]["trajectory"]["name"],
                "duration": basket[end_t]["duration"],
                "kwargs": kwargs,
            }
        else:

            temp[
                "{}_to_{}".format(
                    basket[start_t]["affordance_name"],
                    basket[end_t]["affordance_name"],
                )
            ] = {
                "src": basket[start_t]["affordance_name"],
                "sink": basket[end_t]["affordance_name"],
                "name": basket[end_t]["trajectory"]["name"],
                "duration": int(duration * sample_ratio),
                "fill_type": basket[end_t].get("fill_type", "still"),
                "kwargs": {
                    "agent_uid": scope,
                    "keypose_names": [
                        basket[start_t]["affordance_name"],
                        basket[end_t]["affordance_name"],
                    ],
                    "keypose_timesteps": [start_t, end_t],
                    "raw_duration": duration,
                    "trajectory_id": id,
                    "gather_index": gather_index,
                },
            }
        edges.append(temp)
    return edges, {}


class RealityParser:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = load_json(config_path)

    def get_object_config(
        self,
    ) -> Dict:
        ret = {"obj_list": self.config.get("objects", {})}
        for _, val in ret["obj_list"].items():
            val["mesh_file"] = os.path.join(
                os.path.dirname(self.config_path), val["mesh_file"]
            )
        return ret

    def get_robot_name(
        self,
    ) -> str:
        return self.config["robot"]["agent"]["robot_type"]

    def get_task_type(
        self,
    ) -> str:
        return self.config["task"]["name"]

    def get_robot_config(
        self,
    ) -> Dict:
        return self.config.get("robot", {})

    def get_perturbation_config(self) -> Dict:
        robot = self.get_robot_config()
        user_config = robot.get("robot_action", {}).get("perturbation_config", {})

        if not user_config:
            return {
                "enable": True,
                "default": {
                    "enable": True,
                    "translation": {
                        "xy_range": [-0.03, 0.03, -0.03, 0.03],
                        "z_range": [0.0, 0.0],
                    },
                    "rotation": {"z_angle_range": [0.0, 0.0]},
                },
            }

        perturbation_config = deepcopy(user_config)

        if "enable" not in perturbation_config:
            perturbation_config["enable"] = True

        return perturbation_config

    def get_trajectory_path(self, id: str) -> str:
        return self.config["task"]["data"][id]["trajectory"]["path"]

    def get_all_trajectory(
        self,
    ) -> Dict[str, np.ndarray]:
        action_raw_configs = self.config["task"]
        trajectory_dict = {}
        for id, action_raw_config in action_raw_configs["data"].items():
            trajectory_dict[id] = self.get_trajectory(id)
        return trajectory_dict

    def get_trajectory(self, id: str) -> np.ndarray:
        trajectory_path = self.get_trajectory_path(id)
        assert os.path.exists(
            trajectory_path
        ), f"Trajectory file {trajectory_path} does not exist."
        if trajectory_path.endswith(".hdf5"):
            import h5py

            with h5py.File(trajectory_path, "r") as f:
                trajectory = f["observations/qpos"][:]
        elif trajectory_path.endswith(".npy"):
            trajectory = np.load(trajectory_path)
        else:
            raise ValueError(
                f"Unsupported trajectory file format: {trajectory_path}. Supported formats are .hdf5 and .npy"
            )
        trajectory = np.array(trajectory)
        return trajectory

    def compose_gym_config(
        self,
    ) -> Dict:
        gym_config = {}
        gym_config["id"] = self.get_env_name()
        gym_config["max_episodes"] = self.config["dataset"]["max_episodes"]
        gym_config["dir_path"] = self.config["dataset"].get("dir_path", None)
        gym_config["task_type"] = {}
        gym_config["task_type"][self.get_env_name()] = {}
        gym_config["task_type"][self.get_env_name()]["num_trajectory"] = self.config[
            "dataset"
        ]["num_trajectory"]

        robot = self.get_robot_config()
        robot["robot_action"]["perturbation_config"] = self.get_perturbation_config()

        # NOTE: maybe need re-index
        # NOTE: assume initial state is the mean of all initial states
        # all_init_state = np.concatenate(
        #     [self.get_trajectory(id)[0:1] for id in self.config["task"]["data"].keys()],
        #     0,
        # )
        robot["agent"]["agent_home_joint"] = self.config["robot"]["agent"][
            "agent_home_joint"
        ]
        gym_config["task_type"][self.get_env_name()]["robot_name"] = {
            robot["agent"]["robot_type"]: robot
        }

        gym_config["task_type"][self.get_env_name()][
            "dataset"
        ] = ConfigTemplate.get_dataset_config()

        is_preview = self.config["dataset"].get("preview_flag", False)
        gym_config["task_type"][self.get_env_name()]["dataset"][
            "preview_flag"
        ] = is_preview

        domain_randomization_template = ConfigTemplate.get_domain_randomization_config()
        if "domain_randomization" in self.config:
            domain_randomization_template["enable"] = self.config[
                "domain_randomization"
            ].get("enable", domain_randomization_template["enable"])

        if is_preview:
            log_info("Preview mode enabled, domain randomization will be disabled.")
            domain_randomization_template["enable"] = False

        gym_config["task_type"][self.get_env_name()][
            "domain_randomization"
        ] = domain_randomization_template

        gym_config["task_type"][self.get_env_name()][
            "success_params"
        ] = ConfigTemplate.get_success_params_config()

        gym_config["sensor"] = self.config["sensor"]
        gym_config["scene"] = self.config["scene"]
        gym_config["record"] = ConfigTemplate.get_record_config()
        return gym_config

    def compose_gym_config_v3(
        self,
    ) -> Dict:
        gym_config: Dict = {}
        gym_config["id"] = self.config.get("id") or self.get_env_name()
        gym_config["max_episodes"] = self.config.get("max_episodes", 1)
        gym_config["robot"] = self.config.get("robot", {})
        gym_config["sensor"] = self.config.get("sensor", [])

        if "light" in self.config and "direct" in self.config["light"]:
            gym_config["light"] = {"direct": self.config["light"]["direct"]}
        else:
            gym_config["light"] = {"direct": []}

        gym_config["rigid_object"] = self.config.get("rigid_object", [])
        gym_config["background"] = self.config.get("background", [])

        env_config = self.config.get("env", {})
        dataset_config = env_config.get("dataset", {})
        env_dataset = {
            "dir_path": dataset_config.get("dir_path", None),
            "num_trajectory": dataset_config.get("num_trajectory", 1),
        }

        robot_meta = dataset_config.get("robot_meta", {})
        env_dataset["robot_meta"] = robot_meta

        instruction = dataset_config.get("instruction", {})
        if not instruction:
            instruction = {"lang": ""}
        env_dataset["instruction"] = instruction

        gym_config["env"] = {
            "dataset": env_dataset,
            "sim_steps_per_control": env_config.get("sim_steps_per_control", 4),
        }

        if "observations" in env_config:
            gym_config["env"]["observations"] = env_config["observations"]

        if "events" in env_config:
            gym_config["env"]["events"] = env_config["events"]

        if "success_params" in env_config:
            gym_config["env"]["success_params"] = env_config["success_params"]

        return gym_config

    def get_env_name(
        self,
    ) -> str:
        return self.get_task_name(camel=False) + "_real2sim"

    def get_task_name(self, camel: bool = True):
        from embodichain.utils.utility import snake_to_camel

        if camel:
            return snake_to_camel(self.config["task"]["name"])
        else:
            return self.config["task"]["name"]

    def get_action_config_v3(
        self,
    ) -> Tuple[List[Dict], List[Dict]]:
        action_raw_configs = self.config["task"]
        list_action_config, list_action_bank_functions = [], []

        canonical_cfg = action_raw_configs["data"][CANONICAL_ID]
        canonical_trajectory = self.get_trajectory(CANONICAL_ID)

        for id, action_raw_config in action_raw_configs["data"].items():
            action_config = ConfigTemplate.get_action_config()
            scope_index = action_raw_config["trajectory"]["scope"]
            sample_ratio = action_raw_config["trajectory"].get("sample_ratio", 0.1)

            scope_name = deepcopy(list(scope_index.keys()))
            new_scope_map = simple_mapping(scope_name)
            new_scope_index = {}
            for scope_name_i in scope_name:
                new_scope_index[new_scope_map[scope_name_i]] = scope_index[scope_name_i]

            for name in scope_name:
                action_config["scope"][new_scope_map[name]] = ConfigTemplate.get_scope()
                action_config["scope"][new_scope_map[name]]["dim"] = [
                    len(scope_index[name])
                ]
                action_config["scope"][new_scope_map[name]]["init"][
                    "init_node_name"
                ] = get_init_affordance(new_scope_map[name])

            action_config["trajectory_path"] = action_raw_config["trajectory"]["path"]
            trajectory = self.get_trajectory(id)
            action_bank_functions = {"node": {}, "edge": {}}
            for name_, cfg in action_raw_config["node"].items():
                name = new_scope_map[name_]
                (action_config["node"][name], functions,) = real2sim_affordance_v3(
                    cfg,
                    len(trajectory[:, new_scope_index[name]]),
                    name,
                    canonical_cfg["node"][name_],
                    self.config.get("rigid_object", {}),
                    id=id,
                    gather_index=new_scope_index[name],
                )
                action_bank_functions["node"].update(functions)

            for name_, cfg in action_raw_config["node"].items():
                name = new_scope_map[name_]
                action_config["edge"][name], functions = auto_link(
                    cfg,
                    len(trajectory[:, new_scope_index[name]]),
                    name,
                    sample_ratio,
                    id=id,
                    gather_index=new_scope_index[name],
                )
                action_bank_functions["edge"].update(functions)

            sync_info = {}
            for start_node_name, cfg in action_raw_config.get("sync", {}).items():
                for edge_i in get_edge_by_name(
                    action_config["edge"], start_node_name, "src"
                ):
                    sync_info[edge_i] = {}
                    sync_info[edge_i]["depend_tasks"] = []
                    for end_node_name in cfg.get("depend_tasks", []):
                        for edge_j in get_edge_by_name(
                            action_config["edge"], end_node_name, "sink"
                        ):
                            sync_info[edge_i]["depend_tasks"].append(edge_j)

            action_config["sync"] = sync_info
            if "misc" in action_raw_config:
                action_config["misc"] = action_raw_config["misc"]
            list_action_config.append(action_config)
            list_action_bank_functions.append(action_bank_functions)
        return list_action_config, list_action_bank_functions

    def get_action_config(
        self,
    ) -> Tuple[List[Dict], List[Dict]]:
        action_raw_configs = self.config["task"]
        list_action_config, list_action_bank_functions = [], []

        canonical_cfg = action_raw_configs["data"][CANONICAL_ID]
        canonical_trajectory = self.get_trajectory(CANONICAL_ID)

        for id, action_raw_config in action_raw_configs["data"].items():
            action_config = ConfigTemplate.get_action_config()
            scope_index = action_raw_config["trajectory"]["scope"]
            sample_ratio = action_raw_config["trajectory"].get("sample_ratio", 0.1)

            scope_name = deepcopy(list(scope_index.keys()))
            new_scope_map = simple_mapping(scope_name)
            new_scope_index = {}
            for scope_name_i in scope_name:
                new_scope_index[new_scope_map[scope_name_i]] = scope_index[scope_name_i]

            for name in scope_name:
                action_config["scope"][new_scope_map[name]] = ConfigTemplate.get_scope()
                action_config["scope"][new_scope_map[name]]["dim"] = [
                    len(scope_index[name])
                ]
                action_config["scope"][new_scope_map[name]]["init"][
                    "init_node_name"
                ] = get_init_affordance(new_scope_map[name])

            action_config["trajectory_path"] = action_raw_config["trajectory"]["path"]
            trajectory = self.get_trajectory(id)
            action_bank_functions = {"node": {}, "edge": {}}
            for name_, cfg in action_raw_config["node"].items():
                name = new_scope_map[name_]
                (action_config["node"][name], functions,) = real2sim_affordance(
                    cfg,
                    len(trajectory[:, new_scope_index[name]]),
                    name,
                    canonical_cfg["node"][name_],
                    self.config.get("objects", {}),
                    id=id,
                    gather_index=new_scope_index[name],
                )
                action_bank_functions["node"].update(functions)

            for name_, cfg in action_raw_config["node"].items():
                name = new_scope_map[name_]
                action_config["edge"][name], functions = auto_link(
                    cfg,
                    len(trajectory[:, new_scope_index[name]]),
                    name,
                    sample_ratio,
                    id=id,
                    gather_index=new_scope_index[name],
                )
                action_bank_functions["edge"].update(functions)

            sync_info = {}
            for start_node_name, cfg in action_raw_config.get("sync", {}).items():
                for edge_i in get_edge_by_name(
                    action_config["edge"], start_node_name, "src"
                ):
                    sync_info[edge_i] = {}
                    sync_info[edge_i]["depend_tasks"] = []
                    for end_node_name in cfg.get("depend_tasks", []):
                        for edge_j in get_edge_by_name(
                            action_config["edge"], end_node_name, "sink"
                        ):
                            sync_info[edge_i]["depend_tasks"].append(edge_j)

            action_config["sync"] = sync_info
            if "misc" in action_raw_config:
                action_config["misc"] = action_raw_config["misc"]
            list_action_config.append(action_config)
            list_action_bank_functions.append(action_bank_functions)
        return list_action_config, list_action_bank_functions
