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

import os
import torch
import numpy as np
import gymnasium as gym

from dataclasses import MISSING
from typing import Dict, Union, Sequence, Tuple, Any, List

from embodichain.lab.sim.cfg import (
    RobotCfg,
    RigidObjectCfg,
    RigidObjectGroupCfg,
    ArticulationCfg,
    LightCfg,
)
from embodichain.lab.gym.envs.action_bank.configurable_action import (
    get_func_tag,
)
from embodichain.lab.gym.envs.action_bank.configurable_action import (
    ActionBank,
)
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.sensors import BaseSensor, SensorCfg
from embodichain.lab.sim.types import EnvObs, EnvAction
from embodichain.lab.gym.envs import BaseEnv, EnvCfg
from embodichain.lab.gym.envs.managers import (
    EventManager,
    ObservationManager,
)
from embodichain.lab.gym.utils.registration import register_env
from embodichain.utils import configclass, logger


__all__ = ["EmbodiedEnvCfg", "EmbodiedEnv"]


@configclass
class EmbodiedEnvCfg(EnvCfg):
    """Configuration class for the Embodied Environment. Inherits from EnvCfg and can be extended
    with additional parameters if needed.
    """

    @configclass
    class EnvLightCfg:
        direct: List[LightCfg] = []

        # TODO: support more types of indirect light in the future.
        # indirect: Dict[str, Any] | None = None

    robot: RobotCfg = MISSING

    sensor: List[SensorCfg] = []

    light: EnvLightCfg = EnvLightCfg()

    background: List[RigidObjectCfg] = []

    rigid_object: List[RigidObjectCfg] = []

    rigid_object_group: List[RigidObjectGroupCfg] = []

    articulation: List[ArticulationCfg] = []

    events: Union[object, None] = None
    """Event settings. Defaults to None, in which case no events are applied through the event manager.

    Please refer to the :class:`embodichain.lab.gym.managers.EventManager` class for more details.
    """

    observations: Union[object, None] = None
    """Observation settings. Defaults to None, in which case no additional observations are applied through
    the observation manager.

    Please refer to the :class:`embodichain.lab.gym.managers.ObservationManager` class for more details.
    """

    # TODO: This would be changed to a more generic data pipeline configuration.
    dataset: Union[Dict[str, Any], None] = None
    """Data pipeline configuration. Defaults to None.
    """

    # Some helper attributes
    filter_visual_rand: bool = False
    """Whether to filter out visual randomization 
    
    This is useful when we want to disable visual randomization for debug motion and physics issues.
    """


@register_env("EmbodiedEnv-v1")
class EmbodiedEnv(BaseEnv):
    """Embodied AI environment that is used to simulate the Embodied AI tasks.

    Core simulation components for Embodied AI environments.
    - sensor: The sensors used to perceive the environment, which could be attached to the agent or the environment.
    - robot: The robot which will be used to interact with the environment.
    - light: The lights in the environment, which could be used to illuminate the environment.
        - indirect: the indirect light sources, such as ambient light, IBL, etc.
            The indirect light sources are used for global illumination which affects the entire scene.
        - direct: The direct light sources, such as point light, spot light, etc.
            The direct light sources are used for local illumination which mainly affects the arena in the scene.
    - background: Kinematic or Static rigid objects, such as obstacles or landmarks.
    - rigid_object: Dynamic objects that can be interacted with.
    - rigid_object_group: Groups of rigid objects that can be interacted with.
    - deformable_object(TODO: supported in the future): Deformable volumes or surfaces (cloth) that can be interacted with.
    - articulation: Articulated objects that can be manipulated, such as doors, drawers, etc.
    - event manager: The event manager is used to manage the events in the environment, such as randomization,
        perturbation, etc.
    - observation manager: The observation manager is used to manage the observations in the environment,
        such as depth, segmentation, etc.
    - action bank: The action bank is used to manage the actions in the environment, such as action composition, action graph, etc.
    - affordance_datas: The affordance data that can be used to store the intermediate results or information
    """

    def __init__(self, cfg: EmbodiedEnvCfg, **kwargs):
        self.affordance_datas = {}
        self.action_bank = None

        extensions = getattr(cfg, "extensions", {}) or {}

        defaults = {
            "obs_mode": "state",
            "episode_length": 50,
            "joint_limits": 0.5,
            "action_scale": 0.1,
        }

        for name, default in defaults.items():
            value = extensions.get(name, getattr(cfg, name, default))
            setattr(cfg, name, value)
            setattr(self, name, getattr(cfg, name))

        super().__init__(cfg, **kwargs)

    def _init_sim_state(self, **kwargs):
        """Initialize the simulation state at the beginning of scene creation."""

        self._apply_functor_filter()

        # create event manager
        self.cfg: EmbodiedEnvCfg
        if self.cfg.events:
            self.event_manager = EventManager(self.cfg.events, self)

            # perform events at the start of the simulation
            if "startup" in self.event_manager.available_modes:
                self.event_manager.apply(mode="startup")

        if self.cfg.observations:
            self.observation_manager = ObservationManager(self.cfg.observations, self)

        # TODO: A workaround for handling dataset saving, which need history data of obs-action pairs.
        # We may improve this by implementing a data manager to handle data saving and online streaming.
        if self.cfg.dataset is not None:
            self.metadata["dataset"] = self.cfg.dataset
            self.episode_obs_list = []
            self.episode_action_list = []

            self.curr_episode = 0

    def _apply_functor_filter(self) -> None:
        """Apply functor filters to the environment components based on configuration.

        This method is used to filter out certain components of the environment, such as visual randomization,
        based on the configuration settings. For example, if `filter_visual_rand` is set to True in the configuration,
        all visual randomization functors will be removed from the event manager.
        """
        from embodichain.utils.module_utils import get_all_exported_items_from_module
        from embodichain.lab.gym.envs.managers.cfg import EventCfg

        functors_to_remove = get_all_exported_items_from_module(
            "embodichain.lab.gym.envs.managers.randomization.rendering"
        )
        if self.cfg.filter_visual_rand and self.cfg.events:
            # Iterate through all attributes of the events object
            for attr_name in dir(self.cfg.events):
                attr = getattr(self.cfg.events, attr_name)
                if isinstance(attr, EventCfg):
                    if attr.func.__name__ in functors_to_remove:
                        logger.log_info(
                            f"Filtering out visual randomization functor: {attr.func.__name__}"
                        )
                        setattr(self.cfg.events, attr_name, None)

    def _init_action_bank(
        self, action_bank_cls: ActionBank, action_config: Dict[str, Any]
    ):
        """
        Initialize action bank and parse action graph structure.

        Args:
            action_bank_cls: The ActionBank class for this environment.
            action_config: The configuration dict for the action bank.
        """
        self.action_bank = action_bank_cls(action_config)
        misc_cfg = action_config.get("misc", {})
        try:
            this_class_name = self.action_bank.__class__.__name__
            node_func = {}
            edge_func = {}
            for class_name in [this_class_name, ActionBank.__name__]:
                node_func.update(get_func_tag("node").functions.get(class_name, {}))
                edge_func.update(get_func_tag("edge").functions.get(class_name, {}))
        except KeyError as e:
            raise KeyError(
                f"Function tag for {e} not found in action bank function registry."
            )

        self.graph_compose, jobs_data, jobkey2index = self.action_bank.parse_network(
            node_functions=node_func, edge_functions=edge_func, vis_graph=False
        )
        self.packages = self.action_bank.gantt(
            tasks_data=jobs_data, taskkey2index=jobkey2index, vis=False
        )

    def set_affordance(self, key: str, value: Any):
        """
        Set an affordance value by key.

        Args:
            key (str): The affordance key.
            value (Any): The affordance value.
        """
        self.affordance_datas[key] = value

    def get_affordance(self, key: str, default: Any = None):
        """
        Get an affordance value by key.

        Args:
            key (str): The affordance key.
            default (Any, optional): Default value if key not found.

        Returns:
            Any: The affordance value or default.
        """
        return self.affordance_datas.get(key, default)

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> Tuple[EnvObs, Dict]:
        obs, info = super().reset(seed=seed, options=options)

        if hasattr(self, "episode_obs_list"):
            self.episode_obs_list = [obs]
            self.episode_action_list = []

        return obs, info

    def step(
        self, action: EnvAction, **kwargs
    ) -> Tuple[EnvObs, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        # TODO: Maybe add action preprocessing manager and its functors.
        obs, reward, done, truncated, info = super().step(action, **kwargs)

        if hasattr(self, "episode_action_list"):

            self.episode_obs_list.append(obs)
            self.episode_action_list.append(action)

        return obs, reward, done, truncated, info

    def _extend_obs(self, obs: EnvObs, **kwargs) -> EnvObs:
        if self.observation_manager:
            obs = self.observation_manager.compute(obs)
        return obs

    def _prepare_scene(self, **kwargs) -> None:
        self._setup_lights()
        self._setup_background()
        self._setup_interactive_objects()

    def _update_sim_state(self, **kwargs) -> None:
        """Perform the simulation step and apply events if configured.

        The events manager applies its functors after physics simulation and rendering,
        and before the observation and reward computation (if applicable).
        """
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval")

    def _initialize_episode(
        self, env_ids: Sequence[int] | None = None, **kwargs
    ) -> None:
        # apply events such as randomization for environments that need a reset
        if self.cfg.events:
            if "reset" in self.event_manager.available_modes:
                self.event_manager.apply(mode="reset", env_ids=env_ids)

    def _step_action(self, action: EnvAction) -> EnvAction:
        """Set action control command into simulation.

        Args:
            action: The action applied to the robot agent.

        Returns:
            The action return.
        """
        # TODO: Support data structure action input such as struct.
        qpos = action

        self.robot.set_qpos(qpos=qpos)

        return action

    def _setup_robot(self, **kwargs) -> Robot:
        """Setup the robot in the environment.

        Currently, only joint position control is supported. Would be extended to support joint velocity and torque
            control in the future.

        Returns:
            Robot: The robot instance added to the scene.
        """
        if self.cfg.robot is None:
            logger.error("Robot configuration is not provided.")

        # Initialize the robot based on the configuration.
        robot: Robot = self.sim.add_robot(self.cfg.robot)

        robot.build_pk_serial_chain()

        # TODO: we may need control parts to group actual controlled joints ids.
        # In this way, the action pass to env should be a dict or struct to store the
        # joint ids as well.
        qpos_limits = robot.body_data.qpos_limits[0].cpu().numpy()
        self.single_action_space = gym.spaces.Box(
            low=qpos_limits[:, 0], high=qpos_limits[:, 1], dtype=np.float32
        )
        return robot

    def _setup_sensors(self, **kwargs) -> Dict[str, BaseSensor]:
        """Setup the sensors in the environment.

        Returns:
            Dict[str, BaseSensor]: A dictionary mapping sensor UIDs to sensor instances.
        """

        # TODO: support sensor attachment to the robot.

        sensors = {}
        for cfg in self.cfg.sensor:
            sensor = self.sim.add_sensor(cfg)
            sensors[cfg.uid] = sensor
        return sensors

    def _setup_lights(self) -> None:
        """Setup the lights in the environment."""
        for cfg in self.cfg.light.direct:
            self.sim.add_light(cfg=cfg)

    def _setup_background(self) -> None:
        """Setup the static rigid objects in the environment."""
        for cfg in self.cfg.background:
            if cfg.body_type == "dynamic":
                logger.log_error(
                    f"Background object must be kinematic or static rigid object."
                )
            self.sim.add_rigid_object(cfg=cfg)

    def _setup_interactive_objects(self) -> None:
        """Setup the interactive objects in the environment."""

        for cfg in self.cfg.articulation:
            self.sim.add_articulation(cfg=cfg)

        for cfg in self.cfg.rigid_object:
            if cfg.body_type != "dynamic":
                logger.log_error(
                    f"Interactive rigid object must be dynamic rigid object."
                )
            self.sim.add_rigid_object(cfg=cfg)

        for cfg in self.cfg.rigid_object_group:
            self.sim.add_rigid_object_group(cfg=cfg)

    def preview_sensor_data(
        self, name: str, data_type: str = "color", env_ids: int = 0, method: str = "plt"
    ) -> None:
        """Preview the sensor data by matplotlib

        Note:
            Currently only support RGB image preview.

        Args:
            name (str): name of the sensor to preview.
            data_type (str): type of the sensor data to preview.
            env_ids (int): index of the arena to preview. Defaults to 0.
            method (str): method to preview the sensor data. Currently support "plt" and "cv2". Defaults to "plt".
        """
        # TODO: this function need to be improved to support more sensor types and data types.

        sensor = self.get_sensor(name=name)

        if data_type not in sensor.SUPPORTED_DATA_TYPES:
            logger.error(
                f"Data type '{data_type}' not supported by sensor '{name}'. Supported types: {sensor.SUPPORTED_DATA_TYPES}"
            )

        sensor.update()

        data = sensor.get_data()

        # TODO: maybe put the preview (visualization) method to the sensor class.
        if sensor.cfg.sensor_type == "StereoCamera":
            view = data[data_type][env_ids].cpu().numpy()
            view_right = data[f"{data_type}_right"][env_ids].cpu().numpy()
            view = np.concatenate((view, view_right), axis=1)
        else:
            view = data[data_type][env_ids].cpu().numpy()

        if method == "cv2":
            import cv2

            cv2.imshow(
                f"sensor_data_{data_type}", cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
            )
            cv2.waitKey(0)
        elif method == "plt":
            from matplotlib import pyplot as plt

            plt.imshow(view)
            plt.savefig(f"sensor_data_{data_type}.png")

    def create_demo_action_list(self, *args, **kwargs) -> Sequence[EnvAction] | None:
        """Create a demonstration action list for the environment.

        This function should be implemented in subclasses to generate a sequence of actions
        that demonstrate a specific task or behavior within the environment.

        Returns:
            Sequence[EnvAction] | None: A list of actions if a demonstration is available, otherwise None.
        """
        raise NotImplementedError(
            "The method 'create_demo_action_list' must be implemented in subclasses."
        )

    def to_dataset(
        self,
        repo_id: str,
        fps: int = 30,
        use_videos: bool = True,
        push_to_hub: bool = False,
        image_writer_threads: int = 4,
        image_writer_processes: int = 0,
    ) -> str | None:
        """Convert the recorded episode data to LeRobot dataset format.

        Args:
            repo_id (str): Repository ID for LeRobot dataset (e.g., "username/dataset_name").
            fps (int): Frames per second for video encoding. Defaults to 30.
            use_videos (bool): Whether to encode images as videos. Defaults to True.
            push_to_hub (bool): Whether to push to Hugging Face Hub. Defaults to False.
            image_writer_threads (int): Number of threads for image writing. Defaults to 4.
            image_writer_processes (int): Number of processes for image writing. Defaults to 0.

        Returns:
            str | None: The path to the saved dataset, or None if failed.
        """
        if not hasattr(self, "episode_obs_list") or not hasattr(
            self, "episode_action_list"
        ):
            logger.log_error(
                "Episode data not available. Make sure dataset configuration is set in the environment config."
            )
            return None

        if len(self.episode_obs_list) == 0:
            logger.log_error(
                "No episode data to save. Episode observation list is empty."
            )
            return None

        try:
            # Import the handler - use try-except to catch import errors
            from embodichain.data.handler.lerobot_data_handler import (
                save_to_lerobot_format,
            )
        except ImportError as e:
            logger.log_error(f"Failed to import lerobot_data_handler: {e}")
            logger.log_error(
                "Make sure all dependencies are installed: pip install lerobot"
            )
            return None
        except Exception as e:
            logger.log_error(f"Unexpected error importing lerobot_data_handler: {e}")
            return None

        # Prepare obs_list and action_list
        # Remove the last observation as it doesn't have a corresponding action
        obs_list = (
            self.episode_obs_list[:-1]
            if len(self.episode_obs_list) > len(self.episode_action_list)
            else self.episode_obs_list
        )
        action_list = self.episode_action_list

        logger.log_info(
            f"Saving episode with {len(obs_list)} frames to LeRobot format..."
        )

        # Save to LeRobot format
        dataset_path = save_to_lerobot_format(
            env=self,
            obs_list=obs_list,
            action_list=action_list,
            repo_id=repo_id,
            fps=fps,
            use_videos=use_videos,
            push_to_hub=push_to_hub,
            image_writer_threads=image_writer_threads,
            image_writer_processes=image_writer_processes,
        )

        if dataset_path:
            logger.log_info(
                f"Successfully saved episode {self.curr_episode} to {dataset_path}"
            )
            self.curr_episode += 1

        return dataset_path

    def is_task_success(self, **kwargs) -> torch.Tensor:
        """Determine if the task is successfully completed. This is mainly used in the data generation process
        of the imitation learning.

        Args:
            **kwargs: Additional arguments for task-specific success criteria.

        Returns:
            torch.Tensor: A boolean tensor indicating success for each environment in the batch.
        """

        return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
