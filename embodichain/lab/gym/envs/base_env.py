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

import torch
import gymnasium as gym

from typing import Dict, List, Union, Tuple, Any, Sequence
from functools import cached_property

from embodichain.lab.sim.types import EnvObs, EnvAction
from embodichain.lab.sim import SimulationManagerCfg, SimulationManager
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.sensors import BaseSensor
from embodichain.lab.gym.utils import gym_utils
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

    ignore_terminations: bool = False
    """Whether to ignore terminations when deciding when to auto reset. Terminations can be caused by
    the task reaching a success or fail state as defined in a task's evaluation function. 

    If set to False, meaning there is early stop in episode rollouts. 
    If set to True, this would generally for situations where you may want to model a task as infinite horizon where a task
    stops only due to the timelimit.
    """


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

    # TODO: May be support multiple robots in the future.
    # The robot agent instance.
    robot: Robot = None

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

        if self.cfg.seed is not None:
            self.cfg.seed = set_seed(self.cfg.seed)
        else:
            logger.log_info(f"No seed is set for the environment.")

        self.sim_freq = int(1 / self.sim_cfg.physics_dt)
        self.control_freq = self.sim_freq // self.cfg.sim_steps_per_control

        self._setup_scene(**kwargs)

        # TODO: To be removed.
        if self.device.type == "cuda":
            self.sim.init_gpu_physics()

        if not self.sim_cfg.headless:
            self.sim.open_window()

        self._elapsed_steps = torch.zeros(
            self._num_envs, dtype=torch.int32, device=self.sim_cfg.sim_device
        )

        self._init_sim_state(**kwargs)

        self._init_raw_obs: Dict = self.get_obs(**kwargs)

        logger.log_info("[INFO]: Initialized environment:")
        logger.log_info(f"\tEnvironment device    : {self.sim.device}")
        logger.log_info(f"\tNumber of environments: {self._num_envs}")
        logger.log_info(f"\tEnvironment seed      : {self.cfg.seed}")
        logger.log_info(f"\tPhysics dt            : {self.sim_cfg.physics_dt}")
        logger.log_info(
            f"\tEnvironment dt        : {self.sim_cfg.physics_dt * self.cfg.sim_steps_per_control}"
        )

    @property
    def num_envs(self) -> int:
        """Return the number of environments simulated in parallel."""
        return self._num_envs

    @property
    def device(self) -> torch.device:
        """Return the device used by the environment."""
        return self.sim.device

    @cached_property
    def single_observation_space(self) -> gym.spaces.Space:
        if self.num_envs == 1:
            return gym_utils.convert_observation_to_space(self._init_raw_obs)
        else:
            return gym_utils.convert_observation_to_space(
                self._init_raw_obs, unbatched=True
            )

    @cached_property
    def observation_space(self) -> gym.spaces.Space:
        if self.num_envs == 1:
            return self.single_observation_space
        else:
            return gym.vector.utils.batch_space(
                self.single_observation_space, n=self.num_envs
            )

    @cached_property
    def action_space(self) -> gym.spaces.Space:
        if self.num_envs == 1:
            return self.single_action_space
        else:
            return gym.vector.utils.batch_space(
                self.single_action_space, n=self.num_envs
            )

    @property
    def elapsed_steps(self) -> Union[int, torch.Tensor]:
        return self._elapsed_steps

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

    def _initialize_episode(self, env_ids: Sequence[int] | None = None, **kwargs):
        """Initialize the simulation assets before each episode. Randomization can be performed at this stage.

        Args:
            env_ids: The environment IDs to be initialized. If None, all environments are initialized.
                This is useful for vectorized environments to reset only the specified environments.
            **kwargs: Additional keyword arguments to be passed to the :meth:`_initialize_episode` function.
        """
        pass

    def _get_sensor_obs(self, **kwargs) -> Dict[str, any]:
        """Get the sensor observation from the environment.

        Args:
            **kwargs: Additional keyword arguments to be passed to the :meth:`_get_sensor_obs` function.

        Returns:
            The sensor observation dictionary.
        """
        obs = {}

        fetch_only = False
        if self.sim.is_rt_enabled:
            fetch_only = True
            self.sim.render_camera_group()

        for sensor_name, sensor in self.sensors.items():
            sensor.update(fetch_only=fetch_only)
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

        Note:
            If self.num_envs == 1, return the observation in single_observation_space format.
            If self.num_envs > 1, return the observation in observation_space format.

        Args:
            **kwargs: Additional keyword arguments to be passed to the :meth:`_get_sensor_obs` functions.

        Returns:
            The observation dictionary.
        """
        obs = None

        obs = dict(robot=self.robot.get_proprioception())

        sensor_obs = self._get_sensor_obs(**kwargs)
        if sensor_obs:
            obs["sensor"] = sensor_obs

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

    def get_info(self, **kwargs) -> Dict[str, Any]:
        """Get info about the current environment state, include elapsed steps, success, fail, etc.

        The returned info dictionary must contain at the success and fail status of the current step.

        Args:
            **kwargs: Additional keyword arguments to be passed to the :meth:`get_info` function.

        Returns:
            The info dictionary.
        """
        info = dict(elapsed_steps=self._elapsed_steps)

        info.update(self.evaluate(**kwargs))
        return info

    def check_truncated(self, obs: EnvObs, info: Dict[str, Any]) -> torch.Tensor:
        """Check if the episode is truncated.

        Args:
            obs: The observation from the environment.
            info: The info dictionary.

        Returns:
            A boolean tensor indicating truncation for each environment in the batch.
        """
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

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

        return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

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

        reset_ids = options.get(
            "reset_ids",
            torch.arange(self.num_envs, dtype=torch.int32, device=self.device),
        )
        self.sim.reset_objects_state(env_ids=reset_ids)
        self._elapsed_steps[reset_ids] = 0

        # Reset hook for user to perform any custom reset logic.
        self._initialize_episode(reset_ids, **options)

        return self.get_obs(**options), self.get_info(**options)

    def step(
        self, action: EnvAction, **kwargs
    ) -> Tuple[EnvObs, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the environment with the given action.

        Args:
            action: The action applied to the robot agent.

        Returns:
            A tuple contraining the observation, reward, terminated, truncated, and info dictionary.
        """
        self._elapsed_steps += 1

        # TODO: may be add hook for action preprocessing.
        action = self._step_action(action=action)
        self.sim.update(self.sim_cfg.physics_dt, self.cfg.sim_steps_per_control)
        self._update_sim_state(**kwargs)

        obs = self.get_obs(**kwargs)
        info = self.get_info(**kwargs)
        rewards = self.get_reward(obs=obs, action=action, info=info)

        terminateds = torch.logical_or(
            info.get(
                "success",
                torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            ),
            info.get(
                "fail", torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            ),
        )
        truncateds = self.check_truncated(obs=obs, info=info)
        if self.cfg.ignore_terminations:
            terminateds[:] = False

        dones = torch.logical_or(terminateds, truncateds)
        reset_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            obs, _ = self.reset(options={"reset_ids": reset_env_ids})

        # TODO: may be add hook for observation postprocessing.

        return obs, rewards, terminateds, truncateds, info

    def close(self) -> None:
        """Close the environment and release resources."""
        self.sim.destroy()
