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
from numbers import Integral

import torch
import numpy as np
import gymnasium as gym

from typing import Dict, List, Union, Tuple, Any, Sequence
from functools import cached_property
from tensordict import TensorDict

from embodichain.lab.sim.types import EnvObs, EnvAction
from embodichain.lab.sim import SimulationManagerCfg, SimulationManager
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.sensors import BaseSensor, Camera
from embodichain.lab.gym.utils import gym_utils
from embodichain.lab.gym.utils.profiler import EnvProfilerCfg
from embodichain.utils import configclass
from embodichain.utils import logger, set_seed

__all__ = ["BaseEnv", "EnvCfg"]


@configclass
class EnvCfg:
    """Configuration for an Robot Learning Environment."""

    num_envs: int = 1
    """The number of sub environments (arena in dexsim context) to be simulated in parallel."""

    sim_cfg: SimulationManagerCfg = SimulationManagerCfg()
    """Simulation configuration for the environment."""

    seed: int | None = None
    """The seed for the random number generator. Defaults to -1, in which case the seed is not set.

    Note:
      The seed is set at the beginning of the environment initialization. This ensures that the environment
      creation is deterministic and behaves similarly across different runs.
    """

    sim_steps_per_control: int = 4
    """Number of simulation steps per control (env) step.

    For instance, if the simulation dt is 0.01s and the control dt is 0.1s, then the `sim_steps_per_control` is 10.
    This means that the control action is updated every 10 simulation steps.
    """

    target_control_frequency_hz: float | None = None
    """Optional requested control frequency in hertz.

    When set, the environment resolves this value to an integer
    :attr:`sim_steps_per_control` using the configured physics timestep and
    takes precedence over the directly configured step count. The requested
    frequency must be exactly representable; the physics timestep is never
    changed and the frequency is never silently approximated.
    """

    ignore_terminations: bool = False
    """Whether to ignore terminations when deciding when to auto reset. Terminations can be caused by
    the task reaching a success or fail state as defined in a task's evaluation function. 

    If set to False, meaning there is early stop in episode rollouts. 
    If set to True, this would generally for situations where you may want to model a task as infinite horizon where a task
    stops only due to the timelimit.
    """

    max_episode_steps: int = 300
    """The maximum number of steps per episode. If set to -1, there is no limit on the episode length, and the episode will
    only end when the task is successfully completed or failed.
    """

    profiler: EnvProfilerCfg | None = None
    """Optional profiler for reset/step wall-time breakdown. ``None`` keeps
    the profiler disabled unless one is configured directly on ``sim_cfg``.
    See :class:`EnvProfilerCfg` for the available options."""


class BaseEnv(gym.Env):
    """Base environment for robot learning.

    Args:
        cfg (EnvCfg): The environment configuration.
        **kwargs: Additional keyword arguments.
    """

    # placeholder contains any meta information about the environment.
    metadata: Dict = {}

    # The simulator manager instance.
    sim: SimulationManager = None

    # The robot agent instance.
    robot: Robot = None

    active_joint_ids: List[int] = []

    # The sensors used in the environment.
    sensors: Dict[str, BaseSensor] = {}

    # The action space is determined by the robot agent and the task the environment is used for.
    action_space: gym.spaces.Space = None
    # The observation space is determined by the sensors used in the environment and the task the environment is used for.
    observation_space: gym.spaces.Space = None

    single_action_space: gym.spaces.Space = None
    single_observation_space: gym.spaces.Space = None

    def __init__(
        self,
        cfg: EnvCfg,
        **kwargs,
    ):
        self.cfg = cfg

        # the number of envs to be simulated in parallel.
        self._num_envs = self.cfg.num_envs

        if self.cfg.sim_cfg is None:
            self.sim_cfg = SimulationManagerCfg(headless=True)
        else:
            self.sim_cfg = self.cfg.sim_cfg
            self.sim_cfg.num_envs = self._num_envs

        # Preserve EnvCfg.profiler as the environment-facing configuration
        # entry point while letting SimulationManager own the profiler. A
        # profiler configured directly on sim_cfg is also supported when the
        # legacy env-level field is left unset.
        if self.cfg.profiler is not None:
            self.sim_cfg.profiler = self.cfg.profiler

        if self.cfg.seed is not None:
            self.cfg.seed = set_seed(self.cfg.seed)
        else:
            logger.log_info(f"No seed is set for the environment.")

        self._configure_timing()

        self._setup_scene(**kwargs)

        # Keep the established env._profiler API while sharing the single
        # profiler instance owned by SimulationManager.
        self._profiler = self.sim.profiler

        # TODO: To be removed.
        if self.device.type == "cuda":
            self.sim.init_gpu_physics()

        if not self.sim_cfg.headless:
            self.sim.open_window()

        self._elapsed_steps = torch.zeros(
            self._num_envs, dtype=torch.int32, device=self.sim_cfg.sim_device
        )

        # -1 means no limit on episode length, and the episode will only end when the task is successfully completed or failed.
        self.max_episode_steps = (
            self.cfg.max_episode_steps if self.cfg.max_episode_steps > 0 else 2**31 - 1
        )

        self._task_success = torch.zeros(
            self._num_envs, dtype=torch.bool, device=self.device
        )
        # The UIDs of objects that are detached from automatic reset.
        self._detached_uids_for_reset: List[str] = []

        self._init_sim_state(**kwargs)

        self.sim.capture_visualization_safely(force=True)

        self._init_raw_obs: Dict = self.get_obs(**kwargs)

        logger.log_info("[INFO]: Initialized environment:")
        logger.log_info(f"\tEnvironment device    : {self.sim.device}")
        logger.log_info(f"\tNumber of environments: {self._num_envs}")
        logger.log_info(f"\tEnvironment seed      : {self.cfg.seed}")
        logger.log_info(f"\tPhysics dt            : {self.physics_dt}")
        logger.log_info(f"\tEnvironment dt        : {self.step_dt}")
        logger.log_info(f"\tControl frequency     : {self.control_frequency_hz} Hz")

    def _configure_timing(self) -> None:
        """Validate and expose the environment's simulation-derived timing."""
        physics_dt = self.sim_cfg.physics_dt
        try:
            physics_dt_value = float(physics_dt)
        except (TypeError, ValueError):
            physics_dt_value = math.nan
        if (
            isinstance(physics_dt, bool)
            or not math.isfinite(physics_dt_value)
            or physics_dt_value <= 0.0
        ):
            raise ValueError(
                f"physics_dt must be a finite positive number, got {physics_dt!r}."
            )

        target_frequency = self.cfg.target_control_frequency_hz
        if target_frequency is not None:
            try:
                target_frequency_value = float(target_frequency)
            except (TypeError, ValueError):
                target_frequency_value = math.nan
            if (
                isinstance(target_frequency, bool)
                or not math.isfinite(target_frequency_value)
                or target_frequency_value <= 0.0
            ):
                raise ValueError(
                    "target_control_frequency_hz must be a finite positive number, "
                    f"got {target_frequency!r}."
                )

            ideal_steps = 1.0 / (physics_dt_value * target_frequency_value)
            resolved_steps = max(1, round(ideal_steps))
            if not math.isclose(
                ideal_steps, float(resolved_steps), rel_tol=0.0, abs_tol=1e-9
            ):
                achievable_frequency = 1.0 / (physics_dt_value * resolved_steps)
                raise ValueError(
                    f"target_control_frequency_hz={target_frequency!r} cannot be "
                    f"represented exactly with physics_dt={physics_dt!r}. The nearest "
                    f"integer sim_steps_per_control is {resolved_steps}, which gives "
                    f"{achievable_frequency:g} Hz. Set sim_steps_per_control explicitly "
                    "or choose an exactly representable target frequency."
                )
            self.cfg.sim_steps_per_control = resolved_steps

        sim_steps = self.cfg.sim_steps_per_control
        if isinstance(sim_steps, bool) or not isinstance(sim_steps, Integral):
            raise ValueError(
                "sim_steps_per_control must be a positive integer, "
                f"got {sim_steps!r}."
            )
        if sim_steps <= 0:
            raise ValueError(
                "sim_steps_per_control must be a positive integer, "
                f"got {sim_steps!r}."
            )

        # Backwards-compatible aliases. Unlike the previous integer division,
        # these values preserve the exact rate implied by the simulation.
        self.sim_freq = self.physics_frequency_hz
        self.control_freq = self.control_frequency_hz

        # Gym consumers (for example video recorders) should observe the same
        # cadence as environment and dataset steps.
        self.metadata = dict(self.metadata)
        self.metadata["render_fps"] = self.control_frequency_hz

    @property
    def num_envs(self) -> int:
        """Return the number of environments simulated in parallel."""
        return self._num_envs

    @property
    def physics_dt(self) -> float:
        """Return the duration of one physics simulation step.

        Returns:
            Physics simulation step duration in seconds.
        """
        return float(self.sim_cfg.physics_dt)

    @property
    def step_dt(self) -> float:
        """Return the duration of one environment control step.

        Returns:
            Environment control step duration in seconds.
        """
        return self.physics_dt * int(self.cfg.sim_steps_per_control)

    @property
    def physics_frequency_hz(self) -> float:
        """Return the physics simulation frequency.

        Returns:
            Physics simulation frequency in hertz.
        """
        return 1.0 / self.physics_dt

    @property
    def control_frequency_hz(self) -> float:
        """Return the environment control frequency.

        Returns:
            Environment control frequency in hertz.
        """
        return 1.0 / self.step_dt

    @property
    def device(self) -> torch.device:
        """Return the device used by the environment."""
        return self.sim.device

    @cached_property
    def single_observation_space(self) -> gym.spaces.Space:
        return gym_utils.convert_observation_to_space(
            self._init_raw_obs, unbatched=True
        )

    @cached_property
    def observation_space(self) -> gym.spaces.Space:
        return gym_utils.convert_observation_to_space(
            self._init_raw_obs, unbatched=False
        )

    @cached_property
    def flattened_observation_space(self) -> gym.spaces.Box:
        """Flattened observation space for RL training.

        Returns a Box space by computing total dimensions from nested dict observations.
        This is needed because RL algorithms (PPO, SAC, etc.) require flat vector inputs.
        """
        from embodichain.learning.rl.utils.helper import flatten_dict_observation

        flattened_obs = flatten_dict_observation(self._init_raw_obs)
        total_dim = flattened_obs.shape[-1]
        return gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32
        )

    @cached_property
    def action_space(self) -> gym.spaces.Space:
        return gym.vector.utils.batch_space(self.single_action_space, n=self.num_envs)

    @property
    def elapsed_steps(self) -> Union[int, torch.Tensor]:
        return self._elapsed_steps

    @property
    def has_sensors(self) -> bool:
        """Return whether the environment has sensors."""
        return len(self.sensors) > 0

    def get_sensor(self, name: str, **kwargs) -> BaseSensor:
        """Get the sensor instance by name.

        Args:
            name: The name of the sensor.
            kwargs: Additional keyword arguments.

        Returns:
            The sensor instance.
        """
        if name not in self.sensors:
            logger.log_error(
                f"Sensor '{name}' not found in the environment. Available sensors: {list(self.sensors.keys())}"
            )

        return self.sensors[name]

    def add_camera_group_id(self, group_id: int) -> None:
        """Add a camera group ID for rendering.

        Args:
            group_id: The camera group ID to be added.
        """
        if not hasattr(self, "_camera_group_ids"):
            self._camera_group_ids: List[int] = []
        self._camera_group_ids.append(group_id)

    def _setup_scene(self, **kwargs):
        # Init sim manager.
        # we want to open gui window when the scene is setup, so init sim manager in headless mode first.
        headless = self.sim_cfg.headless
        self.sim_cfg.headless = True
        self.sim = SimulationManager(self.sim_cfg)
        self.sim_cfg.headless = headless

        logger.log_info(
            f"Initializing {self.num_envs} environments on {self.sim_cfg.sim_device}."
        )

        self.robot = self._setup_robot(**kwargs)
        if len(self.active_joint_ids) == 0:
            self.active_joint_ids = self.robot.active_joint_ids

        if self.robot is None:
            logger.log_error(
                f"The robot instance must be initialized in :meth:`_setup_robot` function."
            )
        if self.single_action_space is None:
            logger.log_error(
                f":attr:`single_action_space` must be defined in the :meth:`_setup_robot` function."
            )

        self._prepare_scene(**kwargs)

        self.sensors = self._setup_sensors(**kwargs)

        # Setup camera groups for rendering.
        self._camera_group_ids: List[int] = []
        for sensor in self.sensors.values():
            if isinstance(sensor, Camera):
                self._camera_group_ids.append(sensor.group_id)

    def _setup_robot(self, **kwargs) -> Robot:
        """Load the robot agent, setup the controller and action space.

        Note:
            1. The fuction must return the robot instance.
            2. The self.single_action_space should be defined.
        """

        # TODO: single_action_space may be configured in config?
        pass

    def _prepare_scene(self, **kwargs) -> None:
        """Prepare the scene assets into the environment.

        This function can be customized to performed different scene creation ways, such as loading from file.
        """
        pass

    def _setup_sensors(self, **kwargs) -> Dict[str, BaseSensor]:
        """Setup the sensors used in the environment.

        The sensors to be setup could be binding to the robot or the environment.

        Note:
            If the function is overridden, it must return a dictionary of sensors with the sensor name as the key
                and the sensor instance as the value.
        """
        return {}

    def _init_sim_state(self, **kwargs):
        """Initialize the simulation state at the beginning of scene creation."""
        pass

    def _update_sim_state(self, **kwargs):
        """Update the simulation state at each step.

        The function is called internally by the environment in :meth:`step` after update the physics simulation.

        Note:
            Currently, the interface is designed to perform randomization of lighting, textures at each simulation step.

        Args:
            **kwargs: Additional keyword arguments to be passed to the :meth:`_update_sim_state` function.
        """
        # TODO: Add randomization event here.
        pass

    def _hook_after_sim_step(
        self,
        obs: EnvObs,
        action: EnvAction,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        info: Dict,
        **kwargs,
    ) -> None:
        """Hook function called after each simulation step.

        Args:
            obs: The observation dictionary.
            action: The action taken by the agent.
            rewards: The reward tensor for the current step.
            dones: A tensor indicating which environments are done.
            info: A dictionary containing additional information.
            **kwargs: Additional keyword arguments to be passed to the :meth:`_hook_after_sim_step` function.
        """
        pass

    def _initialize_episode(self, env_ids: Sequence[int] | None = None, **kwargs):
        """Initialize the simulation assets before each episode. Randomization can be performed at this stage.

        Args:
            env_ids: The environment IDs to be initialized. If None, all environments are initialized.
                This is useful for vectorized environments to reset only the specified environments.
            **kwargs: Additional keyword arguments to be passed to the :meth:`_initialize_episode` function.
        """
        pass

    def _get_sensor_obs(self, **kwargs) -> TensorDict[str, any]:
        """Get the sensor observation from the environment.

        Args:
            **kwargs: Additional keyword arguments to be passed to the :meth:`_get_sensor_obs` function.

        Returns:
            The sensor observation dictionary.
        """
        obs = TensorDict({}, batch_size=[self.num_envs], device=self.device)

        fetch_only = True
        with self._profiler.section("render_camera_group"):
            self.sim.render_camera_group(self._camera_group_ids)

        with self._profiler.section("sensor_fetch"):
            for sensor_name, sensor in self.sensors.items():
                with self._profiler.section(f"sensor_update.{sensor_name}"):
                    sensor.update(fetch_only=fetch_only)
                with self._profiler.section(f"sensor_get_data.{sensor_name}"):
                    obs[sensor_name] = sensor.get_data()
        return obs

    def _extend_obs(self, obs: EnvObs, **kwargs) -> EnvObs:
        """Extend the observation dictionary.

        Overwrite this function to extend or modify extra observation to the existing keys (robot, sensor, extra).

        Args:
            obs: The observation dictionary.
            **kwargs: Additional keyword arguments to be passed to the :meth:`_extend_obs` function.

        Returns:
            The extended observation dictionary.
        """
        return obs

    def get_obs(self, **kwargs) -> EnvObs:
        """Get the observation from the robot agent and the environment.

        The default observation are:
            - robot: the robot proprioception.
            - sensor (optional): the sensor readings.
            - extra (optional): any extra information.

        Args:
            **kwargs: Additional keyword arguments to be passed to the :meth:`_get_sensor_obs` functions.

        Returns:
            The observation dictionary.
        """

        with self._profiler.section("proprio"):
            obs = TensorDict(
                dict(robot=self.robot.get_proprioception()[:, self.active_joint_ids]),
                batch_size=[self.num_envs],
                device=self.device,
            )

        with self._profiler.section("sensor"):
            sensor_obs = self._get_sensor_obs(**kwargs)
        if len(sensor_obs.keys()) > 0:
            obs["sensor"] = sensor_obs

        with self._profiler.section("extend"):
            obs = self._extend_obs(obs=obs, **kwargs)

        return obs

    def evaluate(self, **kwargs) -> Dict[str, Any]:
        """
        Evaluate whether the environment is currently in a success state by returning a dictionary with a "success" key or
        a failure state via a "fail" key

        This function may also return additional data that has been computed (e.g. is the robot grasping some object) that may be
        reused when generating observations and rewards.

        By default if not overridden, this function returns an empty dictionary

        Args:
            **kwargs: Additional keyword arguments to be passed to the :meth:`evaluate` function.

        Returns:
            The evaluation dictionary.
        """
        return dict()

    def get_info(self, **kwargs) -> TensorDict[str, Any]:
        """Get info about the current environment state, include elapsed steps, success, fail, etc.

        The returned info dictionary must contain at the success and fail status of the current step.

        Args:
            **kwargs: Additional keyword arguments to be passed to the :meth:`get_info` function.

        Returns:
            The info dictionary.
        """
        info = TensorDict(
            dict(elapsed_steps=self._elapsed_steps),
            batch_size=[self.num_envs],
            device=self.device,
        )

        evaluate = self.evaluate(**kwargs)
        if evaluate:
            info.update(evaluate)
        return info

    def check_truncated(self, obs: EnvObs, info: TensorDict[str, Any]) -> torch.Tensor:
        """Check if the episode is truncated.

        Args:
            obs: The observation from the environment.
            info: The info dictionary.

        Returns:
            A boolean tensor indicating truncation for each environment in the batch.
        """
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _extend_reward(
        self,
        rewards: torch.Tensor,
        obs: EnvObs,
        action: EnvAction,
        info: Dict[str, Any],
        **kwargs,
    ) -> torch.Tensor:
        """Extend the reward computation.

        Overwrite this function to extend or modify the reward computation.

        Args:
            rewards: The base reward tensor.
            obs: The observation from the environment.
            action: The action applied to the robot agent.
            info: The info dictionary.
            **kwargs: Additional keyword arguments.

        Returns:
            The extended reward tensor.
        """
        return rewards

    def get_reward(
        self,
        obs: EnvObs,
        action: EnvAction,
        info: Dict[str, Any],
    ) -> float:
        """Get the reward for the current step.

        Each SimulationManager env must implement its own get_reward function to define the reward function for the task, If the
        env is considered for RL/IL training.

        Args:
            obs: The observation from the environment.
            action: The action applied to the robot agent.
            info: The info dictionary.

        Returns:
            The reward for the current step.
        """

        rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        return rewards

    def is_task_success(self, **kwargs) -> torch.Tensor:
        """
        Determine if the task is successfully completed. This is mainly used in the data generation process
        of the imitation learning.

        Args:
            **kwargs: Additional arguments for task-specific success criteria.

        Returns:
            torch.Tensor: A boolean tensor indicating success for each environment in the batch.
        """

        return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def _preprocess_action(self, action: EnvAction) -> EnvAction:
        """Preprocess action before sending to robot.

        Override this method to add custom preprocessing like:
        - Action scaling
        - Coordinate transformation (e.g., EEF pose to joint positions)
        - Action space conversion

        Args:
            action: Raw action from policy

        Returns:
            Preprocessed action
        """
        return action

    def _postprocess_action(self, action: EnvAction) -> EnvAction:
        """Postprocess action after applying to robot.

        Post processing is usually used to modify the action after it has been applied to the robot,
        performing normalization, noise addition, or any other modifications that need to be applied
        for policy learning or evaluation purposes.

        Args:
            action: Action after preprocessing and robot control command generation

        Returns:
            Final action to be applied in the simulation
        """
        return action

    def _step_action(self, action: EnvAction) -> EnvAction:
        """Set action control command into simulation.

        Args:
            action: The action applied to the robot agent.

        Returns:
            The action return.
        """
        pass

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> Tuple[EnvObs, Dict]:
        """Reset the SimulationManager environment and return the observation and info.

        Args:
            seed: The seed for the random number generator. Defaults to None, in which case the seed is not set.
            options: Additional options for resetting the environment. This can include:

        Returns:
            A tuple containing the observations and infos.
        """
        if seed is not None:
            torch.manual_seed(seed)

        if options is None:
            options = dict()

        with self._profiler.section("reset", is_root=True):
            reset_ids = options.get(
                "reset_ids",
                torch.arange(self.num_envs, dtype=torch.int32, device=self.device),
            )

            # Save task success status before resetting objects
            with self._profiler.section("is_task_success"):
                self._task_success = self.is_task_success()

            with self._profiler.section("reset_objects_state"):
                self.sim.reset_objects_state(
                    env_ids=reset_ids, excluded_uids=self._detached_uids_for_reset
                )

            # Reset hook for user to perform any custom reset logic.
            with self._profiler.section("initialize_episode"):
                self._initialize_episode(reset_ids, **options)
            self._elapsed_steps[reset_ids] = 0

            self.sim.capture_visualization_safely(force=True)

            with self._profiler.section("get_obs"):
                obs = self.get_obs(**options)
            with self._profiler.section("get_info"):
                info = self.get_info(**options)

        return obs, info

    def step(
        self, action: EnvAction, **kwargs
    ) -> Tuple[EnvObs, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the environment with the given action.

        Args:
            action: The action applied to the robot agent.

        Returns:
            A tuple contraining the observation, reward, terminated, truncated, and info dictionary.
        """

        with self._profiler.section("step", is_root=True):
            with self._profiler.section("preprocess_action"):
                action = self._preprocess_action(action=action)
            with self._profiler.section("step_action"):
                action = self._step_action(action=action)

            with self._profiler.section("sim_update"):
                self.sim.update(self.physics_dt, self.cfg.sim_steps_per_control)
            with self._profiler.section("update_sim_state"):
                self._update_sim_state(**kwargs)

            with self._profiler.section("get_obs"):
                obs = self.get_obs(**kwargs)
            with self._profiler.section("get_info"):
                info = self.get_info(**kwargs)
            with self._profiler.section("reward"):
                rewards = self.get_reward(obs=obs, action=action, info=info)
                rewards = self._extend_reward(
                    rewards=rewards, obs=obs, action=action, info=info
                )

            # Apply postprocessing to the action after all computations are done.
            with self._profiler.section("postprocess_action"):
                action = self._postprocess_action(action=action)

            self._elapsed_steps += 1

            terminateds = torch.logical_or(
                info.get(
                    "success",
                    torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                ),
                info.get(
                    "fail",
                    torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                ),
            )
            truncateds = self.check_truncated(obs=obs, info=info)
            truncateds = truncateds | (self._elapsed_steps >= self.max_episode_steps)

            if self.cfg.ignore_terminations:
                terminateds[:] = False

            dones = terminateds | truncateds

            with self._profiler.section("hook_after"):
                self._hook_after_sim_step(
                    obs=obs,
                    action=action,
                    rewards=rewards,
                    dones=dones,
                    info=info,
                    terminateds=terminateds,
                    truncateds=truncateds,
                    **kwargs,
                )

            if not getattr(self, "_replay_no_auto_reset", False):
                reset_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                if len(reset_env_ids) > 0:
                    with self._profiler.section("auto_reset"):
                        obs, _ = self.reset(options={"reset_ids": reset_env_ids})

        return obs, rewards, terminateds, truncateds, info

    def add_detached_uids_for_reset(self, uids: List[str]) -> None:
        """Add the UIDs of objects that are detached from automatic reset.

        Args:
            uids: The list of UIDs to be detached from automatic reset.
        """
        self._detached_uids_for_reset.extend(uids)

    def close(self) -> None:
        """Close the environment and release resources."""
        # Report before sim.destroy(): destroy() exits the process without
        # returning to Python, so the report must be flushed first.
        self._profiler.report()
        self.sim.destroy()
