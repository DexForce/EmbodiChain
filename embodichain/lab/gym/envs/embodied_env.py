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

from math import log
import os
import torch
import numpy as np
import gymnasium as gym

from dataclasses import MISSING
from typing import Dict, Union, Sequence, Tuple, Any, List, Optional
from tensordict import TensorDict

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
    RewardManager,
    ActionManager,
    DatasetManager,
)
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.gym.utils.gym_utils import (
    init_rollout_buffer_from_gym_space,
)
from embodichain.utils import configclass, logger


__all__ = ["EmbodiedEnvCfg", "EmbodiedEnv"]


@configclass
class EmbodiedEnvCfg(EnvCfg):
    """Configuration for Embodied AI environments.

    `EmbodiedEnvCfg` extends `EnvCfg` with high-level scene, robot, sensor,
    object and manager declarations used to build modular embodied environments.
    The configuration is intended to be declarative: the environment and its
    managers (events, observations, rewards, dataset) are assembled from the
    provided config fields with minimal additional code.

    Typical usage: declare robots, sensors, lights, rigid objects/articulations,
    and manager configurations. Additional task-specific parameters can be
    supplied via the `extensions` dict and will be bound to the environment
    instance as attributes during initialization.

    Key fields
    - **robot**: `RobotCfg` (required) — the agent definition (URDF/MJCF, initial
        state, control mode, etc.).
    - **control_parts**: Optional[List[str]] — named robot parts to control. If
        `None`, all controllable joints are used.
    - **active_joint_ids**: List[int] — explicit joint indices to use for
        control (alternative to `control_parts`).
    - **sensor**: List[`SensorCfg`] — sensors attached to the robot or scene
        (cameras, depth, segmentation, force sensors, ...).
    - **light**: `EnvLightCfg` — lighting configuration (direct lights now,
        indirect/IBL planned for future releases).
    - **background**, **rigid_object**, **rigid_object_group**, **articulation**:
        scene object lists for static/kinematic props, dynamic objects, grouped
        object pools, and articulated mechanisms respectively.
    - **events**: Optional manager config — event functors for startup/reset/
        periodic randomization and scripted behaviors.
    - **observations**, **rewards**, **dataset**: Optional manager configs to
        compose observation transforms, reward functors, and dataset/recorder
        settings (auto-saving on episode completion).
    - **extensions**: Optional[Dict[str, Any]] — arbitrary task-specific key/value
        pairs (e.g. `success_threshold`, `control_frequency`) that are automatically
        set on the config *and* bound to the environment instance.
    - **filter_visual_rand** / **filter_dataset_saving**: booleans to disable
        visual randomization or dataset saving for debugging purposes.
    - **init_rollout_buffer**: bool — when true (or when a dataset manager is
        present and dataset saving is enabled) the environment will initialize a
        rollout buffer matching the observation/action spaces for episode
        recording.

    See `EmbodiedEnv` for usage patterns and the project documentation
    for full examples showing how to declare environments from these configs.
    """

    @configclass
    class EnvLightCfg:
        direct: List[LightCfg] = []

        # TODO: support more types of indirect light in the future.
        indirect: dict[str, Any] | None = None

    robot: RobotCfg = MISSING

    control_parts: list[str] | None = None
    """List of robot parts to control. If None, all controllable joints will be used. 
    This is useful when we want to control only a subset of the robot joints for certain tasks or demonstrations.
    """

    active_joint_ids: List[int] = []
    """List of active joint IDs for control. User also can directly specify the active joint IDs instead of control \
    parts. This is useful when the control parts are not well defined or we want to have more fine-grained control.
    """

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

    rewards: Union[object, None] = None
    """Reward settings. Defaults to None, in which case no reward computation is performed through
    the reward manager.

    Please refer to the :class:`embodichain.lab.gym.managers.RewardManager` class for more details.
    """

    dataset: Union[object, None] = None
    """Dataset settings. Defaults to None, in which case no dataset collection is performed.

    Please refer to the :class:`embodichain.lab.gym.managers.DatasetManager` class for more details.
    """

    actions: Union[object, None] = None
    """Action manager settings. Defaults to None, in which case no action preprocessing is applied.

    When configured, the ActionManager preprocesses raw policy actions (e.g., delta_qpos, eef_pose)
    into robot control format.

    Please refer to the :class:`embodichain.lab.gym.envs.managers.ActionManager` class for more details.
    """

    extensions: Union[Dict[str, Any], None] = None
    """Extension parameters for task-specific configurations.

    This field can be used to pass additional parameters that are specific to certain
    environments or tasks without modifying the base configuration class. For example:
    - success_threshold: Task-specific success distance threshold
    - vr_joint_mapping: VR joint mapping for teleoperation
    - control_frequency: Control frequency for VR teleoperation

    Note: Action configuration (e.g., delta_qpos, scale) should use the ``actions``
    field and ActionManager, not extensions.
    """

    # Some helper attributes
    filter_visual_rand: bool = False
    """Whether to filter out visual randomization 
    
    This is useful when we want to disable visual randomization for debug motion and physics issues.
    """

    filter_dataset_saving: bool = False
    """Whether to filter out dataset saving
    
    This is useful when we want to disable dataset saving for debug motion and physics issues.
    If no dataset manager is configured, this flag will have no effect.
    """

    init_rollout_buffer: bool = False
    """Whether to initialize the rollout buffer in the environment.

    If filter_dataset_saving is False and a dataset manager is configured, the rollout buffer will be initialized by default
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

        for name, value in extensions.items():
            setattr(cfg, name, value)
            setattr(self, name, value)

        self.event_manager: EventManager | None = None
        self.observation_manager: ObservationManager | None = None
        self.reward_manager: RewardManager | None = None
        self.action_manager: ActionManager | None = None
        self.dataset_manager: DatasetManager | None = None

        super().__init__(cfg, **kwargs)

        if self.cfg.dataset and not self.cfg.filter_dataset_saving:
            self.dataset_manager = DatasetManager(self.cfg.dataset, self)
            self.cfg.init_rollout_buffer = True

        # Rollout buffer for episode data collection.
        # The shape of the buffer is (num_envs, max_episode_steps, *data_shape) for each key.
        # The default key in the buffer are:
        # - obs: the observation returned by the environment.
        # - action: the action applied to the environment.
        # - reward: the reward returned by the environment.
        # TODO: we may add more keys and make the buffer extensible in the future.
        # This buffer should also be support initialized from outside of the environment.
        # For example, a shared rollout buffer initialized in model training process and passed to the environment for data collection.
        self.rollout_buffer: TensorDict | None = None
        self._max_rollout_steps = 0
        self._rollout_buffer_mode: str | None = None
        if self.cfg.init_rollout_buffer:
            self.rollout_buffer = init_rollout_buffer_from_gym_space(
                obs_space=self.observation_space,
                action_space=self.action_space,
                max_episode_steps=self.max_episode_steps,
                num_envs=self.num_envs,
                device=self.device,
            )
            self._max_rollout_steps = self.rollout_buffer.shape[1]
            self._rollout_buffer_mode = "episode"

        self.current_rollout_step = 0

        self.episode_success_status: torch.Tensor = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def set_rollout_buffer(self, rollout_buffer: TensorDict) -> None:
        """Set the rollout buffer for episode data collection.

        This function can be used to set the rollout buffer from outside of the environment,
        such as a shared rollout buffer initialized in model training process and passed to the environment for data collection.

        Args:
            rollout_buffer (TensorDict): The rollout buffer to be set. The shape of the buffer should be (num_envs, max_episode_steps, *data_shape) for each key.
        """
        if len(rollout_buffer.shape) != 2:
            logger.log_error(
                f"Invalid rollout buffer shape: {rollout_buffer.shape}. The expected shape is (num_envs, max_episode_steps) for each key."
            )
        self.rollout_buffer = rollout_buffer
        self._max_rollout_steps = self.rollout_buffer.shape[1]
        self.current_rollout_step = 0
        self._rollout_buffer_mode = self._infer_rollout_buffer_mode(rollout_buffer)

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

        if self.cfg.rewards:
            self.reward_manager = RewardManager(self.cfg.rewards, self)

        if self.cfg.actions:
            self.action_manager = ActionManager(self.cfg.actions, self)
            # Override action space to match ActionManager output dim (e.g. EefPoseTerm uses 6/7D)
            self.single_action_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.action_manager.total_action_dim,),
                dtype=np.float32,
            )

    def _apply_functor_filter(self) -> None:
        """Apply functor filters to the environment components based on configuration.

        This method is used to filter out certain components of the environment, such as visual randomization,
        based on the configuration settings. For example, if `filter_visual_rand` is set to True in the configuration,
        all visual randomization functors will be removed from the event manager.
        """
        from embodichain.utils.module_utils import get_all_exported_items_from_module
        from embodichain.lab.gym.envs.managers.cfg import EventCfg

        functors_to_remove = get_all_exported_items_from_module(
            "embodichain.lab.gym.envs.managers.randomization.visual"
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

    def _hook_after_sim_step(
        self,
        obs: EnvObs,
        action: EnvAction,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        info: Dict,
        **kwargs,
    ):
        # TODO: We may make the data collection customizable for rollout buffer.
        if self.rollout_buffer is not None:
            if self.current_rollout_step < self._max_rollout_steps:
                if self._rollout_buffer_mode == "external_rl":
                    self._write_external_rl_rollout_step(
                        obs=obs,
                        rewards=rewards,
                        dones=dones,
                        terminateds=kwargs.get("terminateds"),
                        truncateds=kwargs.get("truncateds"),
                    )
                else:
                    self._write_episode_rollout_step(
                        obs=obs,
                        action=action,
                        rewards=rewards,
                    )
                self.current_rollout_step += 1
            else:
                logger.log_warning(
                    f"Current rollout step {self.current_rollout_step} exceeds max rollout steps {self._max_rollout_steps}. \
                        Data will not be recorded in the rollout buffer."
                )

        # Update success status for all environments where episode is done
        if "success" in info:
            # info["success"] should be a tensor or array of shape (num_envs,)
            self.episode_success_status[dones] = info["success"][dones]

    def _extend_obs(self, obs: EnvObs, **kwargs) -> EnvObs:
        if self.observation_manager:
            obs = self.observation_manager.compute(obs)
        return obs

    def _extend_reward(
        self,
        rewards: torch.Tensor,
        obs: EnvObs,
        action: EnvAction,
        info: Dict[str, Any],
        **kwargs,
    ) -> torch.Tensor:
        if self.reward_manager:
            extra_rewards, reward_info = self.reward_manager.compute(
                obs=obs, action=action, info=info
            )
            info["rewards"] = reward_info
            # Add manager terms to base reward from get_reward() so task reward is kept
            rewards = rewards + extra_rewards
        return rewards

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
        logger.log_debug(f"Initializing episode for env_ids: {env_ids}", color="blue")
        save_data = kwargs.get("save_data", True)

        # Determine which environments to process
        env_ids_to_process = list(range(self.num_envs)) if env_ids is None else env_ids

        # Save dataset before clearing buffers for environments that are being reset
        if save_data and self.dataset_manager:
            if "save" in self.dataset_manager.available_modes:

                # Filter to only save successful episodes
                successful_env_ids = self.episode_success_status | self._task_success

                if successful_env_ids.any():

                    self.dataset_manager.apply(
                        mode="save",
                        env_ids=successful_env_ids.nonzero(as_tuple=True)[0],
                    )

        # Clear episode buffers and reset success status for environments being reset
        if (
            self.rollout_buffer is not None
            and self._rollout_buffer_mode != "external_rl"
        ):
            self.current_rollout_step = 0

        self.episode_success_status[env_ids_to_process] = False

        # apply events such as randomization for environments that need a reset
        if self.cfg.events:
            if "reset" in self.event_manager.available_modes:
                self.event_manager.apply(mode="reset", env_ids=env_ids)

        # reset reward manager for environments that need a reset
        if self.cfg.rewards:
            self.reward_manager.reset(env_ids=env_ids)

    def _infer_rollout_buffer_mode(self, rollout_buffer: TensorDict) -> str:
        """Infer whether the rollout buffer is env-owned episode data or external RL data."""
        if "next" in rollout_buffer.keys() and "observation" in rollout_buffer.keys():
            return "external_rl"
        return "episode"

    def _write_episode_rollout_step(
        self,
        obs: EnvObs,
        action: EnvAction,
        rewards: torch.Tensor,
    ) -> None:
        """Write one step into the legacy episode recording rollout buffer."""
        buffer_device = self.rollout_buffer.device
        self.rollout_buffer["obs"][:, self.current_rollout_step, ...].copy_(
            obs.to(buffer_device), non_blocking=True
        )
        if isinstance(action, TensorDict):
            action_to_store = (
                action["qpos"]
                if "qpos" in action
                else (action["qvel"] if "qvel" in action else action["qf"])
            )
        elif isinstance(action, torch.Tensor):
            action_to_store = action
        else:
            logger.log_warning(
                f"Unexpected action type {type(action)} in _hook_after_sim_step; "
                "skipping action storage in rollout buffer."
            )
            action_to_store = None
        if action_to_store is not None:
            self.rollout_buffer["actions"][:, self.current_rollout_step, ...].copy_(
                action_to_store.to(buffer_device), non_blocking=True
            )
        self.rollout_buffer["rewards"][:, self.current_rollout_step].copy_(
            rewards.to(buffer_device), non_blocking=True
        )

    def _write_external_rl_rollout_step(
        self,
        obs: EnvObs,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        terminateds: torch.Tensor | None,
        truncateds: torch.Tensor | None,
    ) -> None:
        """Write environment-side fields into an externally managed RL rollout buffer."""
        from embodichain.agents.rl.utils import flatten_dict_observation

        buffer_device = self.rollout_buffer.device
        obs_to_store = (
            flatten_dict_observation(obs) if isinstance(obs, TensorDict) else obs
        )
        self.rollout_buffer["next", "observation"][:, self.current_rollout_step].copy_(
            obs_to_store.to(buffer_device), non_blocking=True
        )
        self.rollout_buffer["next", "reward"][:, self.current_rollout_step].copy_(
            rewards.to(buffer_device), non_blocking=True
        )
        self.rollout_buffer["next", "done"][:, self.current_rollout_step].copy_(
            dones.to(buffer_device), non_blocking=True
        )

        terminateds = (
            terminateds
            if terminateds is not None
            else torch.zeros_like(dones, dtype=torch.bool)
        )
        truncateds = (
            truncateds
            if truncateds is not None
            else torch.zeros_like(dones, dtype=torch.bool)
        )
        self.rollout_buffer["next", "terminated"][:, self.current_rollout_step].copy_(
            terminateds.to(buffer_device), non_blocking=True
        )
        self.rollout_buffer["next", "truncated"][:, self.current_rollout_step].copy_(
            truncateds.to(buffer_device), non_blocking=True
        )

    def _step_action(self, action: EnvAction) -> EnvAction:
        """Set action control command into simulation.

        Supports multiple action formats:
        1. torch.Tensor: Interpreted as qpos (joint positions)
        2. Dict with keys:
           - "qpos": Joint positions
           - "qvel": Joint velocities
           - "qf": Joint forces/torques

        Args:
            action: The action applied to the robot agent.

        Returns:
            The action return.
        """
        if isinstance(action, TensorDict):
            # Support multiple control modes simultaneously
            if "qpos" in action:
                self.robot.set_qpos(
                    qpos=action["qpos"], joint_ids=self.active_joint_ids
                )
            if "qvel" in action:
                self.robot.set_qvel(
                    qvel=action["qvel"], joint_ids=self.active_joint_ids
                )
            if "qf" in action:
                self.robot.set_qf(qf=action["qf"], joint_ids=self.active_joint_ids)
        elif isinstance(action, torch.Tensor):
            self.robot.set_qpos(qpos=action, joint_ids=self.active_joint_ids)
        else:
            logger.log_error(f"Unsupported action type: {type(action)}")

        return action

    def compute_task_state(
        self, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Compute task-specific state: success, fail, and metrics.

        Override this method in subclass to define task-specific logic for RL tasks.

        Returns:
            Tuple of (success, fail, metrics):
                - success: Boolean tensor of shape (num_envs,)
                - fail: Boolean tensor of shape (num_envs,)
                - metrics: Dict of metric tensors
        """
        success = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        fail = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        metrics: Dict[str, Any] = {}
        return success, fail, metrics

    def get_info(self, **kwargs) -> Dict[str, Any]:
        """Get environment info dictionary.

        Calls compute_task_state() to get task-specific success/fail/metrics when
        available. Subclasses should override compute_task_state() for RL tasks.

        Returns:
            Info dictionary with success, fail, elapsed_steps, metrics
        """
        success, fail, metrics = self.compute_task_state(**kwargs)
        info: Dict[str, Any] = {
            "success": success,
            "fail": fail,
            "elapsed_steps": self._elapsed_steps,
            "metrics": metrics,
        }
        return info

    def evaluate(self, **kwargs) -> Dict[str, Any]:
        """Evaluate the environment state.

        Returns:
            Evaluation dictionary with success and metrics
        """
        info = self.get_info(**kwargs)
        eval_dict: Dict[str, Any] = {
            "success": info["success"][0].item(),
        }
        if "metrics" in info:
            for key, value in info["metrics"].items():
                eval_dict[key] = value
        return eval_dict

    def _preprocess_action(self, action: EnvAction) -> EnvAction:
        """Delegate to ActionManager when configured."""
        if self.action_manager is not None:
            return self.action_manager.process_action(action)
        return super()._preprocess_action(action)

    def _setup_robot(self, **kwargs) -> Robot:
        """Setup the robot in the environment.

        Currently, only joint position control is supported. Would be extended to support joint velocity and torque
            control in the future.

        Returns:
            Robot: The robot instance added to the scene.
        """
        if self.cfg.robot is None:
            logger.log_error("Robot configuration is not provided.")

        # Initialize the robot based on the configuration.
        robot: Robot = self.sim.add_robot(self.cfg.robot)

        # Setup active joints for robot to control.
        if self.cfg.control_parts:
            if len(self.cfg.active_joint_ids) > 0:
                logger.log_error(
                    f"Both control_parts and active_joint_ids are specified in the configuration. Please specify only one of them."
                )

            # Check env control parts are valid
            for part_name in self.cfg.control_parts:
                if part_name not in robot.control_parts:
                    logger.log_error(
                        f"Invalid control part: {part_name}. The supported control parts are: {robot.control_parts}"
                    )

            for part_name in self.cfg.control_parts:
                self.active_joint_ids.extend(
                    robot.get_joint_ids(name=part_name, remove_mimic=True)
                )
        elif self.cfg.active_joint_ids:
            # Check env active joint ids are valid
            for joint_id in self.cfg.active_joint_ids:
                if joint_id not in robot.active_joint_ids:
                    logger.log_error(
                        f"Invalid active joint id: {joint_id}. The supported active joint ids are: {robot.active_joint_ids}"
                    )
            self.active_joint_ids = self.cfg.active_joint_ids
        else:
            # Use all joints of the robot.
            self.active_joint_ids = list(range(robot.dof))

        robot.build_pk_serial_chain()

        qpos_limits = (
            robot.body_data.qpos_limits[0, self.active_joint_ids].cpu().numpy()
        )
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
        # Set direct lights.
        for cfg in self.cfg.light.direct:
            self.sim.add_light(cfg=cfg)

        # Set indirect lights.
        if self.cfg.light.indirect is not None:
            if "emission_light" in self.cfg.light.indirect:
                self.sim.set_emission_light(**self.cfg.light.indirect["emission_light"])

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
        self,
        name: str,
        data_type: str = "color",
        env_ids: int = 0,
        method: str = "cv2",
        save: bool = False,
    ) -> None:
        """Preview the sensor data by matplotlib

        Note:
            Currently only support RGB image preview.

        Args:
            name (str): name of the sensor to preview.
            data_type (str): type of the sensor data to preview.
            env_ids (int): index of the arena to preview. Defaults to 0.
            method (str): method to preview the sensor data. Currently support "plt" and "cv2". Defaults to "cv2".
            save (bool): whether to save the preview image. Defaults to False.
        """
        # TODO: this function need to be improved to support more sensor types and data types.

        sensor = self.get_sensor(name=name)

        if data_type not in sensor.SUPPORTED_DATA_TYPES:
            logger.log_error(
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

            if save:
                cv2.imwrite(
                    f"sensor_data_{data_type}.png",
                    cv2.cvtColor(view, cv2.COLOR_RGB2BGR),
                )
            else:
                window_name = f"sensor_data_{data_type}"
                height, width = view.shape[:2]
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, width, height)
                cv2.imshow(window_name, cv2.cvtColor(view, cv2.COLOR_RGB2BGR))
                cv2.waitKey(0)
                cv2.destroyWindow(window_name)

        elif method == "plt":
            from matplotlib import pyplot as plt

            plt.imshow(view)
            if save:
                plt.savefig(f"sensor_data_{data_type}.png")
                plt.close()
            else:
                plt.show()

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

    def close(self) -> None:
        """Close the environment and release resources."""
        # Finalize dataset if present
        if self.dataset_manager:
            self.dataset_manager.finalize()

        self.sim.destroy()
