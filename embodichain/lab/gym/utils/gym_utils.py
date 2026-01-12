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

from typing import Dict, Any, List, Tuple, Union, Sequence
from gymnasium import spaces
from copy import deepcopy

from embodichain.lab.sim.types import Device, Array
from embodichain.lab.sim.objects import Robot
from embodichain.utils.module_utils import find_function_from_modules
from embodichain.utils.utility import get_class_instance
from dexsim.utility import log_debug, log_error


def get_dtype_bounds(dtype: np.dtype):
    """Gets the min and max values of a given numpy type"""
    if np.issubdtype(dtype, np.floating):
        info = np.finfo(dtype)
        return info.min, info.max
    elif np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return info.min, info.max
    elif np.issubdtype(dtype, np.bool_):
        return 0, 1
    else:
        raise TypeError(dtype)


def convert_observation_to_space(
    observation: Any, prefix: str = "", unbatched: bool = False
) -> spaces.Space:
    """Convert observation to OpenAI gym observation space (recursively).
    Modified from `gym.envs.mujoco_env`
    """
    if isinstance(observation, (dict)):
        # CATUION: Explicitly create a list of key-value tuples
        # Otherwise, spaces.Dict will sort keys if a dict is provided
        space = spaces.Dict(
            [
                (
                    k,
                    convert_observation_to_space(
                        v, prefix + "/" + k, unbatched=unbatched
                    ),
                )
                for k, v in observation.items()
            ]
        )
    elif isinstance(observation, (list, tuple)):
        array = np.array(observation)
        dtype = array.dtype
        space = spaces.Box(-np.inf, np.inf, shape=array.shape, dtype=dtype)
    elif isinstance(observation, np.ndarray):
        if unbatched:
            shape = observation.shape[1:]
        else:
            shape = observation.shape
        dtype = observation.dtype
        low, high = get_dtype_bounds(dtype)
        if np.issubdtype(dtype, np.floating):
            low, high = -np.inf, np.inf
        space = spaces.Box(low, high, shape=shape, dtype=dtype)
    elif isinstance(observation, (float, np.float32, np.float64)):
        log_debug(f"The observation ({prefix}) is a (float) scalar")
        space = spaces.Box(-np.inf, np.inf, shape=[1], dtype=np.float32)
    elif isinstance(observation, (int, np.int32, np.int64)):
        log_debug(f"The observation ({prefix}) is a (integer) scalar")
        space = spaces.Box(-np.inf, np.inf, shape=[1], dtype=int)
    elif isinstance(observation, (bool, np.bool_)):
        log_debug(f"The observation ({prefix}) is a (bool) scalar")
        space = spaces.Box(0, 1, shape=[1], dtype=np.bool_)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


def _batch(array: Union[np.ndarray, Sequence]):
    if isinstance(array, (dict)):
        return {k: _batch(v) for k, v in array.items()}
    if isinstance(array, str):
        return array
    if isinstance(array, np.ndarray):
        if array.shape == ():
            return array.reshape(1, 1)
        return array[None, :]
    if isinstance(array, list):
        if len(array) == 1:
            return [array]
    if (
        isinstance(array, float)
        or isinstance(array, int)
        or isinstance(array, bool)
        or isinstance(array, np.bool_)
    ):
        return np.array([[array]])
    return array


def batch(*args: Tuple[Union[np.ndarray, Dict]]):
    """Adds one dimension in front of everything. If given a dictionary, every leaf in the dictionary
    has a new dimension. If given a tuple, returns the same tuple with each element batched
    """
    x = [_batch(x) for x in args]
    if len(args) == 1:
        return x[0]
    return tuple(x)


def to_tensor(array: Array, device: Device | None = None):
    """
    Maps any given sequence to a torch tensor on the CPU/GPU. If physx gpu is not enabled then we use CPU, otherwise GPU, unless specified
    by the device argument

    Args:
        array: The data to map to a tensor
        device: The device to put the tensor on. By default this is None and to_tensor will put the device on the GPU if physx is enabled
            and CPU otherwise

    """
    if isinstance(array, (dict)):
        return {k: to_tensor(v) for k, v in array.items()}
    if torch.cuda.is_available():
        if isinstance(array, np.ndarray):
            if array.dtype == np.uint16:
                array = array.astype(np.int32)
            ret = torch.from_numpy(array)
            if ret.dtype == torch.float64:
                ret = ret.float()
        elif isinstance(array, torch.Tensor):
            ret = array
        else:
            ret = torch.tensor(array)
        if device is None:
            if ret.device.type == "cpu":
                return ret.cuda()
            # keep same device if already on GPU
            return ret
        else:
            return ret.to(device)
    else:
        if isinstance(array, np.ndarray):
            if array.dtype == np.uint16:
                array = array.astype(np.int32)
            if array.dtype == np.uint32:
                array = array.astype(np.int64)
            ret = torch.from_numpy(array)
            if ret.dtype == torch.float64:
                ret = ret.float()
        elif isinstance(array, list) and isinstance(array[0], np.ndarray):
            ret = torch.from_numpy(np.array(array))
            if ret.dtype == torch.float64:
                ret = ret.float()
        elif np.iterable(array):
            ret = torch.Tensor(array)
        else:
            ret = torch.Tensor(array)
        if device is None:
            return ret
        else:
            return ret.to(device)


def to_cpu_tensor(array: Array):
    """
    Maps any given sequence to a torch tensor on the CPU.
    """
    if isinstance(array, (dict)):
        return {k: to_tensor(v) for k, v in array.items()}
    if isinstance(array, np.ndarray):
        ret = torch.from_numpy(array)
        if ret.dtype == torch.float64:
            ret = ret.float()
        return ret
    elif isinstance(array, torch.Tensor):
        return array.cpu()
    else:
        return torch.tensor(array).cpu()


def flatten_state_dict(
    state_dict: dict, use_torch=False, device: Device = None
) -> Array:
    """Flatten a dictionary containing states recursively. Expects all data to be either torch or numpy

    Args:
        state_dict: a dictionary containing scalars or 1-dim vectors.
        use_torch (bool): Whether to convert the data to torch tensors.

    Raises:
        AssertionError: If a value of @state_dict is an ndarray with ndim > 2.

    Returns:
        np.ndarray | torch.Tensor: flattened states.

    Notes:
        The input is recommended to be ordered (e.g. dict).
        However, since python 3.7, dictionary order is guaranteed to be insertion order.
    """
    states = []

    for key, value in state_dict.items():
        if isinstance(value, dict):
            state = flatten_state_dict(value, use_torch=use_torch)
            if state.size == 0:
                state = None
            if use_torch:
                state = to_tensor(state)
        elif isinstance(value, (tuple, list)):
            state = None if len(value) == 0 else value
            if use_torch:
                state = to_tensor(state)
        elif isinstance(value, (bool, np.bool_, int, np.int32, np.int64)):
            # x = np.array(1) > 0 is np.bool_ instead of ndarray
            state = int(value)
            if use_torch:
                state = to_tensor(state)
        elif isinstance(value, (float, np.float32, np.float64)):
            state = np.float32(value)
            if use_torch:
                state = to_tensor(state)
        elif isinstance(value, np.ndarray):
            if value.ndim > 2:
                raise AssertionError(
                    "The dimension of {} should not be more than 2.".format(key)
                )
            state = value if value.size > 0 else None
            if use_torch:
                state = to_tensor(state)

        elif isinstance(value, torch.Tensor):
            state = value
            if len(state.shape) == 1:
                state = state[:, None]
        else:
            raise TypeError("Unsupported type: {}".format(type(value)))
        if state is not None:
            states.append(state)

    if use_torch:
        if len(states) == 0:
            return torch.empty(0, device=device)
        else:
            return torch.hstack(states)
    else:
        if len(states) == 0:
            return np.empty(0)
        else:
            return np.hstack(states)


def clip_and_scale_action(
    action: Union[np.ndarray, torch.Tensor], low: float, high: float
):
    """Clip action to [-1, 1] and scale according to a range [low, high]."""
    if isinstance(action, np.ndarray):
        action = np.clip(action, -1, 1)
    elif isinstance(action, torch.Tensor):
        action = torch.clip(action, -1, 1)
    else:
        log_error("Unsupported type: {}".format(type(action)))
    return 0.5 * (high + low) + 0.5 * (high - low) * action


def dict_array_to_torch_inplace(
    data: Dict[str, Any], device: Union[str, torch.device] = "cpu"
) -> None:
    """
    Convert arrays in a dictionary to torch tensors in-place.

    Args:
        data (Dict[str, Any]): Dictionary to modify in-place
        device (Union[str, torch.device]): Device to place the tensors on
    """
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            item: torch.Tensor = torch.from_numpy(value).to(device)
            if len(item.shape) == 1:
                item.unsqueeze_(0)
            data[key] = item
        elif isinstance(value, dict):
            dict_array_to_torch_inplace(value, device)


def cat_tensor_with_ids(
    tensors: List[torch.Tensor], ids: List[List[int]], dim: int
) -> torch.Tensor:
    """
    Concatenate tensors along a new dimension specified by `dim`, using the provided `ids` to index into the tensors.

    Args:
        tensors (List[torch.Tensor]): List of tensors to concatenate.
        ids (List[List[int]]): List of lists, where each inner list contains the indices to select from the corresponding tensor.
        dim (int): The dimension along which to concatenate the tensors.

    Returns:
        torch.Tensor: The concatenated tensor.
    """
    out = torch.zeros(
        (tensors[0].shape[0], dim), dtype=tensors[0].dtype, device=tensors[0].device
    )

    for i, tensor in enumerate(tensors):
        out[:, ids[i]] = tensor

    return out


def config_to_rl_cfg(config: dict) -> "RLEnvCfg":
    """Parse gym-style configuration dict into an RL-ready config object."""

    from embodichain.lab.gym.envs.rl_env_cfg import RLEnvCfg

    # Use config_to_cfg to parse shared fields
    env_cfg = config_to_cfg(config)
    # Convert to RLEnvCfg if needed
    if not isinstance(env_cfg, RLEnvCfg):
        env_cfg = RLEnvCfg.from_dict(env_cfg.__dict__)
    # RL-specific fields
    env_cfg.env_id = config.get("id")
    env_cfg.num_envs = config["env"].get("num_envs", env_cfg.num_envs)
    env_cfg.extensions = deepcopy(config.get("env", {}).get("extensions", {}))
    # Add any RL-specific parsing here
    return env_cfg


def config_to_cfg(config: dict) -> "EmbodiedEnvCfg":
    """Parser configuration file into cfgs for env initialization.

    Args:
        config (dict): The configuration dictionary containing robot, sensor, light, background, and interactive objects.

    Returns:
        EmbodiedEnvCfg: A configuration object for initializing the environment.
    """

    from embodichain.lab.sim.cfg import (
        RobotCfg,
        RigidObjectCfg,
        RigidObjectGroupCfg,
        ArticulationCfg,
        LightCfg,
    )
    from embodichain.lab.gym.envs import EmbodiedEnvCfg
    from embodichain.lab.sim.sensors import SensorCfg
    from embodichain.lab.gym.envs.managers import (
        SceneEntityCfg,
        EventCfg,
        ObservationCfg,
        DatasetFunctorCfg,
    )
    from embodichain.utils import configclass
    from embodichain.data import get_data_path

    @configclass
    class ComponentCfg:
        """Configuration for env events.

        This class is used to define various events that can occur in the environment,
        """

        pass

    env_cfg = EmbodiedEnvCfg()

    # check all necessary keys
    required_keys = ["id", "max_episodes", "env", "robot"]
    for key in required_keys:
        if key not in config:
            log_error(f"Missing required config key: {key}")

    # parser robot config
    # TODO: support multiple robots cfg initialization from config, eg, cobotmagic, dexforce_w1, etc.
    if "robot_type" in config["robot"]:
        robot_cfg = get_class_instance(
            "embodichain.lab.sim.robots",
            config["robot"]["robot_type"] + "Cfg",
        )
        config["robot"].pop("robot_type")
        robot_cfg = robot_cfg.from_dict(config["robot"])
    else:
        robot_cfg = RobotCfg.from_dict(config["robot"])

    env_cfg.robot = robot_cfg

    # parser sensor config
    env_cfg.sensor = [SensorCfg.from_dict(s) for s in config.get("sensor", [])]

    # parser light config
    if "light" in config:
        env_cfg.light = EmbodiedEnvCfg.EnvLightCfg()
        env_cfg.light.direct = [
            LightCfg.from_dict(l) for l in config["light"].get("direct", [])
        ]

    # parser background objects config
    if "background" in config:
        for obj_dict in config["background"]:
            shape_type = obj_dict["shape"]["shape_type"]
            if shape_type == "Mesh":
                obj_dict["shape"]["fpath"] = get_data_path(obj_dict["shape"]["fpath"])
            # Set to static object if not specified.
            obj_dict["body_type"] = (
                "static" if "body_type" not in obj_dict else obj_dict["body_type"]
            )
            cfg = RigidObjectCfg.from_dict(obj_dict)
            env_cfg.background.append(cfg)

    # parser scene objects config
    if "rigid_object" in config:
        for obj_dict in config["rigid_object"]:
            shape_type = obj_dict["shape"]["shape_type"]
            if shape_type == "Mesh":
                obj_dict["shape"]["fpath"] = get_data_path(obj_dict["shape"]["fpath"])
            cfg = RigidObjectCfg.from_dict(obj_dict)
            env_cfg.rigid_object.append(cfg)

    if "rigid_object_group" in config:
        for obj_dict in config["rigid_object_group"]:
            if "folder_path" in obj_dict:
                obj_dict["folder_path"] = get_data_path(obj_dict["folder_path"])
            for rigid_obj in obj_dict["rigid_objects"].values():
                shape_type = rigid_obj["shape"]["shape_type"]
                if shape_type == "Mesh" and "fpath" in rigid_obj["shape"]:
                    rigid_obj["shape"]["fpath"] = get_data_path(
                        rigid_obj["shape"]["fpath"]
                    )
            cfg = RigidObjectGroupCfg.from_dict(obj_dict)
            env_cfg.rigid_object_group.append(cfg)

    if "articulation" in config:
        for obj_dict in config["articulation"]:
            obj_dict["fpath"] = get_data_path(obj_dict["fpath"])
            cfg = ArticulationCfg.from_dict(obj_dict)
            env_cfg.articulation.append(cfg)

    env_cfg.sim_steps_per_control = config["env"].get("sim_steps_per_control", 4)

    # load dataset config
    env_cfg.dataset = ComponentCfg()
    if "dataset" in config["env"]:
        # Define modules to search for dataset functions
        dataset_modules = [
            "embodichain.lab.gym.envs.managers.datasets",
        ]

        for dataset_name, dataset_params in config["env"]["dataset"].items():
            dataset_params_modified = deepcopy(dataset_params)

            # Find the function from multiple modules using the utility function
            dataset_func = find_function_from_modules(
                dataset_params["func"],
                dataset_modules,
                raise_if_not_found=True,
            )

            from embodichain.lab.gym.envs.managers import DatasetFunctorCfg

            dataset = DatasetFunctorCfg(
                func=dataset_func,
                mode=dataset_params_modified["mode"],
                params=dataset_params_modified["params"],
            )

            setattr(env_cfg.dataset, dataset_name, dataset)

    # TODO: support more env events, eg, grasp pose generation, mesh preprocessing, etc.

    env_cfg.events = ComponentCfg()
    if "events" in config["env"]:
        # Define modules to search for event functions
        event_modules = [
            "embodichain.lab.gym.envs.managers.randomization",
            "embodichain.lab.gym.envs.managers.record",
            "embodichain.lab.gym.envs.managers.events",
        ]

        # parser env events config
        for event_name, event_params in config["env"]["events"].items():
            event_params_modified = deepcopy(event_params)
            if "entity_cfg" in event_params["params"]:
                entity_cfg = SceneEntityCfg(
                    **event_params_modified["params"]["entity_cfg"]
                )
                event_params_modified["params"]["entity_cfg"] = entity_cfg

            # Find the function from multiple modules using the utility function
            event_func = find_function_from_modules(
                event_params["func"], event_modules, raise_if_not_found=True
            )
            interval_step = event_params_modified.get("interval_step", 10)

            event = EventCfg(
                func=event_func,
                mode=event_params_modified["mode"],
                params=event_params_modified["params"],
                interval_step=interval_step,
            )
            setattr(env_cfg.events, event_name, event)

    env_cfg.observations = ComponentCfg()
    if "observations" in config["env"]:
        # Define modules to search for observation functions
        observation_modules = [
            "embodichain.lab.gym.envs.managers.observations",
        ]

        for obs_name, obs_params in config["env"]["observations"].items():
            obs_params_modified = deepcopy(obs_params)

            if "entity_cfg" in obs_params["params"]:
                entity_cfg = SceneEntityCfg(
                    **obs_params_modified["params"]["entity_cfg"]
                )
                obs_params_modified["params"]["entity_cfg"] = entity_cfg

            # Find the function from multiple modules using the utility function
            obs_func = find_function_from_modules(
                obs_params["func"],
                observation_modules,
                raise_if_not_found=True,
            )

            observation = ObservationCfg(
                func=obs_func,
                mode=obs_params_modified["mode"],
                name=obs_params_modified["name"],
                params=obs_params_modified["params"],
            )

            setattr(env_cfg.observations, obs_name, observation)

    return env_cfg


def map_qpos_to_eef_pose(
    robot: Robot, qpos: torch.Tensor, control_parts: List[str]
) -> Dict[str, torch.Tensor]:
    """Map qpos to end-effector pose.

    Note:
        The computed eef pose will be in the base frame of the control part.

    Args:
        robot (Robot): The robot instance.
        qpos (torch.Tensor): The qpos tensor of shape (N, num_joints).
        control_parts (List[str]): List of control part names.
        to_dict (bool): Whether to return the result as a dictionary.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing the end-effector poses for each control part.
    """

    eef_pose_dict = {}
    for i, name in enumerate(control_parts):
        eef_pose = torch.zeros(
            (qpos.shape[0], 9), dtype=torch.float32, device=qpos.device
        )

        # TODO: need to be configurable.
        control_ids = robot.get_joint_ids(name)
        current_qpos = qpos[:, control_ids]
        part_eef_pose = (
            robot.pk_serial_chain[name]
            .forward_kinematics(current_qpos, end_only=True)
            .get_matrix()
        )

        eef_pose[:, :3] = part_eef_pose[:, :3, 3]
        eef_pose[:, 3:6] = part_eef_pose[:, :3, 0]
        eef_pose[:, 6:9] = part_eef_pose[:, :3, 1]

        eef_pose_dict[name] = eef_pose

    return eef_pose_dict


def fetch_data_from_dict(
    data_dict: Dict[str, Union[Any, Dict[str, Any]]], name: str
) -> Any:
    """Fetch data from a nested dictionary using a '/' separated key.

    Args:
        data_dict (Dict[str, Union[Any, Dict[str, Any]]]): The nested dictionary to fetch data from.
        name (str): The '/' separated key string.

    Returns:
        Any: The fetched data.

    Raises:
        KeyError: If the specified key does not exist in the dictionary.
    """
    keys = name.split("/")
    current_data = data_dict

    for key in keys:
        if key in current_data:
            current_data = current_data[key]
        else:
            raise KeyError(f"Key '{key}' not found in the dictionary.")

    return current_data


def assign_data_to_dict(
    data_dict: Dict[str, Union[Any, Dict[str, Any]]], name: str, value: Any
) -> None:
    """Assign data to a nested dictionary using a '/' separated key.
    Missing intermediate dictionaries will be created automatically.

    Args:
        data_dict (Dict[str, Union[Any, Dict[str, Any]]]): The nested dictionary to assign data to.
        name (str): The '/' separated key string.
        value (Any): The value to assign.
    """
    keys = name.split("/")
    current_data = data_dict

    for key in keys[:-1]:
        if key not in current_data or not isinstance(current_data[key], dict):
            current_data[key] = {}  # create intermediate dict if missing
        current_data = current_data[key]

    last_key = keys[-1]
    current_data[last_key] = value
