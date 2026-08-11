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

from collections.abc import Mapping
from math import log
from functools import wraps
from datetime import datetime
import os
import threading
import torch
import numpy as np
import gymnasium as gym

from dataclasses import MISSING
from typing import (
    TYPE_CHECKING,
    Dict,
    Union,
    Sequence,
    Tuple,
    Any,
    Iterable,
    List,
    Optional,
)
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
from embodichain.lab.gym.envs.demo import (
    DEMO_SCHEMA_VERSION,
    DemoEpisodeResult,
    DemoSegment,
    DemoSegmentResult,
    ProcessedEnvAction,
)
from embodichain.lab.gym.envs.managers import (
    EventManager,
    ObservationManager,
    RewardManager,
    ActionManager,
    DatasetManager,
)
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.gym.utils.gym_utils import (
    build_trajectory_buffer,
    init_rollout_buffer_from_gym_space,
)
from embodichain.lab.gym.utils.trajectory_state import capture_trajectory_state
from embodichain.utils import configclass, logger
from embodichain.data import get_data_path
from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATA_ROOT

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.expert_program import (
        CompiledProgram,
        ExpertProgramCfg,
    )
    from embodichain.lab.gym.envs.expert_program.bridge import AtomicDemoBridge

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

    record_trajectory: bool = False
    """Whether to record per-object states and pre-process actions.

    Each saved row is a causal ``(state_t, action_t)`` pair, matching expert
    trajectory frame alignment. Uses a per-env step counter so async parallel
    environments are supported.
    """

    trajectory_uids: list[str] | None = None
    """Optional allow-list of non-robot object uids to record. If None, all rigid
    objects and articulations are recorded. The robot is always recorded."""

    trajectory_save_dir: str | None = None
    """Directory for auto-saved trajectories. Defaults to
    ``<EMBODICHAIN_DEFAULT_DATA_ROOT>/trajectories/{run_id}/``."""

    trajectory_auto_save: bool = True
    """If True (and record_trajectory is True), auto-save each env's trajectory to
    ``trajectory_save_dir`` at episode end and on close()."""

    expert_program: ExpertProgramCfg | None = None
    """Optional declarative Expert Program used to generate demo segments.

    The program remains inert until :meth:`EmbodiedEnv.create_demo_segments`
    requests an explicit environment compiler and bridge through the dedicated
    hooks. No live provider, planner, or callable is stored in this config.
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

    @classmethod
    def __init_subclass__(cls, **kwargs):
        """Automatically wrap subclass demo-action builders with shape checks.

        Any subclass overriding ``create_demo_action_list`` will be wrapped so its
        returned action sequence is validated and, when possible, converted to the
        environment action dimension.
        """
        super().__init_subclass__(**kwargs)
        method = cls.__dict__.get("create_demo_action_list")
        if method is None or getattr(method, "_demo_action_shape_wrapped", False):
            return

        @wraps(method)
        def wrapped_create_demo_action_list(self, *args, **kwargs):
            action_list = method(self, *args, **kwargs)
            return self._normalize_demo_action_list(action_list)

        wrapped_create_demo_action_list._demo_action_shape_wrapped = True
        setattr(cls, "create_demo_action_list", wrapped_create_demo_action_list)

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

        dataset_terms = getattr(self.cfg.dataset, "__dict__", self.cfg.dataset)
        if dataset_terms and not self.cfg.filter_dataset_saving:
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
            self._rollout_buffer_mode = "expert"

        # Dedicated per-env trajectory buffer (states + actions). Decoupled from
        # rollout_buffer so async parallel envs and ActionManager are supported.
        self._traj_buffer: TensorDict | None = None
        self._traj_steps: torch.Tensor | None = None
        self._traj_raw_action: EnvAction | None = None
        self._traj_save_count = 0
        self._traj_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.cfg.record_trajectory:
            self._traj_buffer = build_trajectory_buffer(
                env=self,
                max_steps=self.max_episode_steps,
                num_envs=self.num_envs,
                device=self.device,
                uids=self.cfg.trajectory_uids,
                action_space=self.action_space,
            )
            self._traj_steps = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )

        self.rollout_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._demo_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.current_rollout_step = 0

        # Segment recording is intentionally separate from task planning. The
        # common demo executor updates this context while the regular rollout
        # writer turns it into per-frame annotations.
        self._demo_episode_index = 0
        self._demo_active_segment_id = 0
        self._demo_active_segment_ids = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._demo_active_mask = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._demo_segment_participants = self._demo_active_mask.clone()
        self._demo_active_segment_start_steps = self._demo_steps.clone()
        self._demo_active_rollout_start_steps = self.rollout_steps.clone()
        self._demo_episode_metadata: list[dict[str, Any]] = [
            self._new_demo_episode_metadata(env_id) for env_id in range(self.num_envs)
        ]

        self.episode_success_status: torch.Tensor = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._closed = False
        self._close_error: BaseException | None = None
        self._close_lock = threading.RLock()

        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self._seed_recording_state(self._init_raw_obs, all_env_ids)

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[EnvObs, Dict]:
        """Reset environments and seed pre-action recording state.

        Expert frames must pair the observation before an action with that
        action. The base reset computes the authoritative post-reset
        observation, so recording is seeded only after it returns.

        Args:
            seed: Optional random seed forwarded to :class:`BaseEnv`.
            options: Reset options. ``reset_ids`` may select only some vector
                environment rows.

        Returns:
            The reset observation and info dictionary.
        """
        obs, info = super().reset(seed=seed, options=options)
        if options is None or "reset_ids" not in options:
            reset_ids = torch.arange(self.num_envs, device=self.device)
        else:
            reset_ids = torch.as_tensor(
                options["reset_ids"], dtype=torch.long, device=self.device
            ).reshape(-1)
        self._seed_recording_state(obs, reset_ids)
        return obs, info

    def _seed_recording_state(self, obs: EnvObs, env_ids: torch.Tensor) -> None:
        """Seed all enabled recorders from the current environment state."""
        self._seed_expert_observations(obs, env_ids)
        self._seed_trajectory_states(env_ids)

    def _seed_expert_observations(self, obs: EnvObs, env_ids: torch.Tensor) -> None:
        """Copy current observations into pending expert-transition slots."""
        if (
            self.rollout_buffer is None
            or getattr(self, "_rollout_buffer_mode", "expert") == "rl"
            or env_ids.numel() == 0
        ):
            return
        eligible = self.rollout_steps[env_ids] < self._max_rollout_steps
        env_ids = env_ids[eligible]
        if env_ids.numel() == 0:
            return
        step_ids = self.rollout_steps[env_ids]
        buffer_device = self.rollout_buffer.device
        buffer_env_ids = env_ids.to(buffer_device)
        buffer_step_ids = step_ids.to(buffer_device)
        obs_device = getattr(obs, "device", None) or self.device
        obs_env_ids = env_ids.to(obs_device)
        self.rollout_buffer["obs"][buffer_env_ids, buffer_step_ids] = obs[
            obs_env_ids
        ].to(buffer_device)

    def _seed_trajectory_states(self, env_ids: torch.Tensor) -> None:
        """Copy current states into pending trajectory-transition slots."""
        if self._traj_buffer is None or env_ids.numel() == 0:
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        eligible = self._traj_steps[env_ids] < self._traj_buffer.shape[1]
        env_ids = env_ids[eligible]
        if env_ids.numel() == 0:
            return
        step_ids = self._traj_steps[env_ids]
        capture_trajectory_state(self, self._traj_buffer["states"], env_ids, step_ids)

    def set_rollout_buffer(self, rollout_buffer: TensorDict) -> None:
        """Set the rollout buffer for episode data collection.

        This function can be used to set the rollout buffer from outside of the environment,
        such as a shared rollout buffer initialized in model training process and passed to the environment for data collection.

        Args:
            rollout_buffer (TensorDict): The rollout buffer to be set. RL
                rollouts use a uniform `[num_envs, time + 1]` layout so all
                fields share the same batch shape; the last slot of
                transition-only fields is reserved as padding. Expert buffers
                keep the legacy `[num_envs, time]` batch layout.
        """
        if rollout_buffer.shape[0] != self.num_envs:
            raise ValueError(
                "Rollout buffer rows must match env.num_envs: "
                f"got {rollout_buffer.shape[0]} rows for {self.num_envs} envs."
            )

        self.rollout_buffer = rollout_buffer
        self._rollout_buffer_mode = self._infer_rollout_buffer_mode(rollout_buffer)
        if self._rollout_buffer_mode == "rl":
            batch_size = self.rollout_buffer.batch_size
            if len(batch_size) != 2:
                message = (
                    f"Invalid RL rollout buffer batch size: {batch_size}. "
                    "Expected a 2D batch layout [num_envs, time + 1] for RL rollouts."
                )
                logger.log_error(message)
                raise ValueError(message)
            self._max_rollout_steps = batch_size[1] - 1
        else:
            if len(rollout_buffer.shape) != 2:
                logger.log_error(
                    f"Invalid rollout buffer shape: {rollout_buffer.shape}. The expected shape is (num_envs, max_episode_steps) for each key."
                )
            self._max_rollout_steps = self.rollout_buffer.shape[1]
        self.rollout_steps.zero_()
        self.current_rollout_step = 0
        if self._rollout_buffer_mode != "rl":
            env_ids = torch.arange(self.num_envs, device=self.device)
            self._clear_expert_rollout_rows(env_ids)
            self._seed_expert_observations(self.get_obs(), env_ids)

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
            # Override action space to match ActionManager output dim.
            self.single_action_space = self.action_manager.single_action_space

    def _apply_functor_filter(self) -> None:
        """Apply functor filters to the environment components based on configuration.

        This method is used to filter out certain components of the environment, such as visual randomization,
        based on the configuration settings. For example, if `filter_visual_rand` is set to True in the configuration,
        all visual randomization functors will be removed from the event manager.
        """
        from embodichain.utils.module_utils import get_all_exported_items_from_module
        from embodichain.lab.gym.envs.managers.cfg import EventCfg

        functors_to_remove = {
            name
            for name in get_all_exported_items_from_module(
                "embodichain.lab.gym.envs.managers.randomization.visual"
            )
            if name.startswith("randomize_")
        }
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
            with self._profiler.section("rollout_write"):
                if self._rollout_buffer_mode == "rl":
                    if self.current_rollout_step < self._max_rollout_steps:
                        self._write_rl_rollout_step(
                            obs=obs,
                            rewards=rewards,
                            dones=dones,
                            terminateds=kwargs.get("terminateds"),
                            truncateds=kwargs.get("truncateds"),
                        )
                    else:
                        logger.log_warning(
                            f"Current rollout step {self.current_rollout_step} exceeds max rollout steps {self._max_rollout_steps}. "
                            "Data will not be recorded in the rollout buffer."
                        )
                    self.current_rollout_step += 1
                else:
                    self._write_episode_rollout_step(
                        obs=obs,
                        action=action,
                        rewards=rewards,
                        terminateds=kwargs.get("terminateds"),
                        truncateds=kwargs.get("truncateds"),
                    )

        demo_steps = getattr(self, "_demo_steps", None)
        if demo_steps is not None:
            active_mask = getattr(self, "_demo_active_mask", None)
            if active_mask is None:
                demo_steps += 1
            else:
                demo_steps += active_mask.to(demo_steps.device, dtype=demo_steps.dtype)

        with self._profiler.section("trajectory_write"):
            self._write_trajectory_step()

        self._update_episode_success_status(info, dones)

    def _update_episode_success_status(
        self, info: Dict[str, Any], dones: torch.Tensor
    ) -> None:
        """Update terminal success without mutating already frozen demo rows."""
        if "success" not in info:
            return
        update_mask = dones.to(
            device=self.episode_success_status.device, dtype=torch.bool
        )
        if getattr(self, "_demo_no_auto_reset", False):
            active_mask = getattr(self, "_demo_active_mask", None)
            if active_mask is not None:
                update_mask = update_mask & active_mask.to(update_mask.device)
        success = torch.as_tensor(
            info["success"],
            dtype=torch.bool,
            device=self.episode_success_status.device,
        )
        self.episode_success_status[update_mask] = success[update_mask]

    def _extend_obs(self, obs: EnvObs, **kwargs) -> EnvObs:
        if self.observation_manager:
            with self._profiler.section("obs_compute"):
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
            with self._profiler.section("reward_compute"):
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
                with self._profiler.section("event_interval"):
                    self.event_manager.apply(mode="interval")

    def _initialize_episode(
        self, env_ids: Sequence[int] | None = None, **kwargs
    ) -> None:
        logger.log_debug(f"Initializing episode for env_ids: {env_ids}", color="blue")
        save_data = kwargs.get("save_data", True)

        # Determine which environments to process
        status_device = self.episode_success_status.device
        if env_ids is None:
            env_ids_to_process = torch.arange(self.num_envs, device=status_device)
        elif isinstance(env_ids, torch.Tensor):
            env_ids_to_process = env_ids.to(device=status_device, dtype=torch.long)
        else:
            env_ids_to_process = torch.tensor(
                list(env_ids), device=status_device, dtype=torch.long
            )

        # Save dataset before clearing buffers for environments that are being reset
        if save_data and self.dataset_manager:
            if "save" in self.dataset_manager.available_modes:

                if self.dataset_manager.save_failed_episodes:
                    env_ids_to_save = env_ids_to_process
                else:
                    successful_envs = self.episode_success_status | self._task_success
                    env_ids_to_save = env_ids_to_process[
                        successful_envs[env_ids_to_process]
                    ]

                if env_ids_to_save.numel() > 0:
                    with self._profiler.section("dataset_save"):
                        self.dataset_manager.apply(
                            mode="save",
                            env_ids=env_ids_to_save,
                        )

        # Save recorded camera data before resetting
        if self.cfg.events and self.event_manager is not None:
            from embodichain.lab.gym.envs.managers.record import record_camera_data

            with self._profiler.section("record_camera_save"):
                for mode_cfgs in self.event_manager._mode_functor_cfgs.values():
                    for functor_cfg in mode_cfgs:
                        if isinstance(functor_cfg.func, record_camera_data):
                            if save_data:
                                functor_cfg.func.save_and_clear(
                                    env_ids=env_ids_to_process
                                )
                            else:
                                functor_cfg.func.discard_and_clear(
                                    env_ids=env_ids_to_process
                                )

        # Auto-save + reset the per-env trajectory buffer for environments being
        # reset. Use getattr so this no-ops on envs/subclasses that don't allocate
        # a _traj_buffer (e.g. unit-test stubs of _initialize_episode).
        _traj_buffer = getattr(self, "_traj_buffer", None)
        if (
            save_data
            and _traj_buffer is not None
            and getattr(self.cfg, "trajectory_auto_save", False)
        ):
            with self._profiler.section("trajectory_save"):
                for env_id in env_ids_to_process.tolist():
                    self._save_trajectory_for_env(env_id)

        _traj_steps = getattr(self, "_traj_steps", None)
        if _traj_steps is not None:
            _traj_steps[env_ids_to_process] = 0

        # Clear episode buffers only after every recorder has consumed them.
        if self.rollout_buffer is not None and self._rollout_buffer_mode != "rl":
            self._clear_expert_rollout_rows(env_ids_to_process)
            rollout_steps = getattr(self, "rollout_steps", None)
            if rollout_steps is not None:
                rollout_ids = env_ids_to_process.to(rollout_steps.device)
                rollout_steps[rollout_ids] = 0
                self.current_rollout_step = int(rollout_steps.max().item())

        episode_metadata = getattr(self, "_demo_episode_metadata", None)
        if episode_metadata is not None:
            for env_id in env_ids_to_process.cpu().tolist():
                episode_metadata[env_id] = self._new_demo_episode_metadata(env_id)
        active_segment_ids = getattr(self, "_demo_active_segment_ids", None)
        if active_segment_ids is not None:
            demo_ids = env_ids_to_process.to(active_segment_ids.device)
            active_segment_ids[demo_ids] = 0
            self._demo_active_mask[demo_ids] = True
            self._demo_segment_participants[demo_ids] = False
            self._demo_active_segment_start_steps[demo_ids] = 0
            self._demo_active_rollout_start_steps[demo_ids] = 0
            self._demo_steps[demo_ids] = 0

        self.episode_success_status[env_ids_to_process] = False

        # apply events such as randomization for environments that need a reset
        if self.cfg.events:
            if "reset" in self.event_manager.available_modes:
                with self._profiler.section("event_reset"):
                    self.event_manager.apply(mode="reset", env_ids=env_ids)

        # reset observation manager for environments that need a reset
        # This clears any cached data in observation functors (e.g., physics attributes)
        if self.cfg.observations:
            with self._profiler.section("obs_reset"):
                self.observation_manager.reset(env_ids=env_ids)

        # reset reward manager for environments that need a reset
        if self.cfg.rewards:
            with self._profiler.section("reward_reset"):
                self.reward_manager.reset(env_ids=env_ids)

        # Dataset saving can be disabled while the dataset configuration remains
        # present.  In that mode no DatasetManager is created in __init__, so
        # reset must not dereference the optional manager.
        if self.cfg.dataset and self.dataset_manager is not None:
            with self._profiler.section("dataset_reset"):
                self.dataset_manager.reset(env_ids=env_ids)

    def _clear_expert_rollout_rows(self, env_ids: torch.Tensor) -> None:
        """Invalidate selected expert-buffer rows without clearing large frames."""
        if self.rollout_buffer is None or len(env_ids) == 0:
            return
        buffer_ids = env_ids.to(self.rollout_buffer.device, dtype=torch.long)
        if "valid" not in self.rollout_buffer.keys():
            # Preserve schema-v1 behavior for external buffers that have no
            # validity mask and therefore cannot hide a stale tail.
            for key in self.rollout_buffer.keys(include_nested=True, leaves_only=True):
                self.rollout_buffer[key][buffer_ids] = 0
            return

        for key in (
            "valid",
            "segment_start",
            "segment_end",
            "terminated",
            "truncated",
        ):
            if key in self.rollout_buffer.keys():
                self.rollout_buffer[key][buffer_ids] = False
        for key in ("episode_step", "segment_id", "segment_step"):
            if key in self.rollout_buffer.keys():
                self.rollout_buffer[key][buffer_ids] = -1

    def _new_demo_episode_metadata(self, env_id: int) -> dict[str, Any]:
        """Create an empty metadata record for one environment row."""
        return {
            "schema_version": DEMO_SCHEMA_VERSION,
            "episode_index": int(getattr(self, "_demo_episode_index", 0)),
            "env_id": env_id,
            "length": 0,
            "completed": False,
            "success": False,
            "terminated": False,
            "truncated": False,
            "terminal_reason": "unknown",
            "segments": [],
        }

    def _begin_demo_episode_recording(self, episode_index: int = 0) -> None:
        """Start annotation metadata for a new demonstration episode."""
        self._demo_episode_index = episode_index
        self._demo_active_segment_id = 0
        self._demo_active_segment_ids.zero_()
        self._demo_active_mask.fill_(True)
        self._demo_segment_participants.fill_(False)
        self._demo_active_segment_start_steps = self._demo_steps.clone()
        self._demo_active_rollout_start_steps = self.rollout_steps.clone()
        self._demo_episode_metadata = [
            self._new_demo_episode_metadata(env_id) for env_id in range(self.num_envs)
        ]

    def _begin_demo_segment_recording(
        self, segment_id: int, segment: DemoSegment
    ) -> None:
        """Set the segment context used by subsequent rollout writes."""
        self._demo_active_segment_id = segment_id
        self._demo_segment_participants = self._demo_active_mask.clone()
        self._demo_active_segment_ids[self._demo_segment_participants] = segment_id
        self._demo_active_segment_start_steps = self._demo_steps.clone()
        self._demo_active_rollout_start_steps = self.rollout_steps.clone()

    def _set_demo_active_mask(self, active_mask: Sequence[bool]) -> None:
        """Set rows that still participate in the current demo episode.

        The executor updates this sticky mask after each terminal transition.
        Recording hooks use it to freeze completed rows while unfinished rows
        continue on the shared vector-environment clock.

        Args:
            active_mask: One liveness flag per parallel environment.

        Raises:
            ValueError: If the mask length does not match ``num_envs``.
        """
        mask = torch.as_tensor(
            active_mask, dtype=torch.bool, device=self._demo_active_mask.device
        ).reshape(-1)
        if mask.numel() != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} demo activity flags, got {mask.numel()}."
            )
        self._demo_active_mask.copy_(mask)

    def _end_demo_segment_recording(self, result: DemoSegmentResult) -> None:
        """Close the active segment span and mark its final valid frame."""
        for env_id in range(self.num_envs):
            is_participant = (
                result.active[env_id]
                if result.active
                else bool(self._demo_segment_participants[env_id])
            )
            if not is_participant:
                continue
            start = int(self._demo_active_segment_start_steps[env_id].item())
            end = int(self._demo_steps[env_id].item())
            rollout_start = int(self._demo_active_rollout_start_steps[env_id].item())
            rollout_end = int(self.rollout_steps[env_id].item())
            if (
                self.rollout_buffer is not None
                and "segment_end" in self.rollout_buffer.keys()
                and rollout_end > rollout_start
            ):
                self.rollout_buffer["segment_end"][env_id, rollout_end - 1] = True

            metadata = result.to_metadata(env_id if result.start_steps else None)
            metadata["start_step"] = start
            metadata["end_step"] = end
            self._demo_episode_metadata[env_id]["segments"].append(metadata)

    def _end_demo_episode_recording(self, result: DemoEpisodeResult) -> None:
        """Finalize per-environment metadata after demonstration execution."""
        for env_id in range(self.num_envs):
            metadata = self._demo_episode_metadata[env_id]
            length = (
                result.lengths[env_id]
                if result.lengths
                else int(self._demo_steps[env_id].item())
            )
            completed = (
                result.completed_by_env[env_id]
                if result.completed_by_env
                else result.completed
            )
            terminal_reason = (
                result.terminal_reasons[env_id]
                if result.terminal_reasons
                else result.terminal_reason
            )
            metadata.update(
                {
                    "episode_index": result.episode_index,
                    "length": length,
                    "completed": completed,
                    "success": result.success[env_id],
                    "terminated": result.terminated[env_id],
                    "truncated": result.truncated[env_id],
                    "terminal_reason": terminal_reason,
                }
            )

    def get_demo_episode_metadata(self, env_id: int) -> dict[str, Any]:
        """Return segment-aware metadata for one buffered episode.

        Legacy collection paths that do not use the common executor are
        represented as one segment spanning every valid frame.

        Args:
            env_id: Parallel environment row.

        Returns:
            A JSON-compatible metadata dictionary.
        """
        metadata = dict(self._demo_episode_metadata[env_id])
        metadata["segments"] = [
            dict(segment)
            for segment in self._demo_episode_metadata[env_id].get("segments", [])
        ]
        rollout_steps = getattr(self, "rollout_steps", None)
        length = int(
            (
                rollout_steps[env_id]
                if rollout_steps is not None and self.rollout_buffer is not None
                else self._demo_steps[env_id]
            ).item()
        )
        metadata["length"] = length
        if length > 0 and not metadata["segments"]:
            env_metadata = getattr(self, "metadata", {})
            dataset_metadata = (
                env_metadata.get("dataset", {})
                if isinstance(env_metadata, Mapping)
                else {}
            )
            instruction_cfg = (
                dataset_metadata.get("instruction")
                if isinstance(dataset_metadata, Mapping)
                else None
            )
            instruction = (
                instruction_cfg.get("lang")
                if isinstance(instruction_cfg, Mapping)
                else instruction_cfg
            )
            instruction = str(instruction) if instruction else "unknown_task"
            success_status = getattr(self, "episode_success_status", None)
            task_success = getattr(self, "_task_success", None)
            success = bool(
                (success_status is not None and success_status[env_id])
                or (task_success is not None and task_success[env_id])
            )
            terminated = bool(
                self.rollout_buffer is not None
                and "terminated" in self.rollout_buffer.keys()
                and self.rollout_buffer["terminated"][env_id, length - 1]
            )
            truncated = bool(
                self.rollout_buffer is not None
                and "truncated" in self.rollout_buffer.keys()
                and self.rollout_buffer["truncated"][env_id, length - 1]
            )
            if truncated:
                success = False
                terminal_reason = "truncated"
            elif success:
                terminal_reason = "success"
            elif terminated:
                terminal_reason = "failure"
            else:
                terminal_reason = "task_incomplete"
            metadata.update(
                {
                    "completed": success,
                    "success": success,
                    "terminated": terminated,
                    "truncated": truncated,
                    "terminal_reason": terminal_reason,
                }
            )
            metadata["segments"] = [
                {
                    "segment_id": 0,
                    "name": "legacy",
                    "start_step": 0,
                    "end_step": length,
                    "success": success,
                    "target_uid": None,
                    "instruction": instruction,
                    "failure_reason": None if success else terminal_reason,
                    "metadata": {},
                }
            ]
        return metadata

    def _infer_rollout_buffer_mode(self, rollout_buffer: TensorDict) -> str:
        """Infer whether the rollout buffer is expert recording or RL training data."""
        if {
            "obs",
            "action",
            "reward",
            "done",
            "value",
            "terminated",
            "truncated",
        }.issubset(set(rollout_buffer.keys())):
            return "rl"
        return "expert"

    def _write_episode_rollout_step(
        self,
        obs: EnvObs,
        action: EnvAction,
        rewards: torch.Tensor,
        terminateds: torch.Tensor | None = None,
        truncateds: torch.Tensor | None = None,
    ) -> None:
        """Complete one causally aligned expert transition per active env.

        The observation at each cursor is seeded before its action is applied.
        This method fills transition fields at that cursor, then seeds the
        returned post-action observation as the next pending pre-action state.
        """
        buffer_device = self.rollout_buffer.device
        active = self.rollout_steps < self._max_rollout_steps
        demo_active = getattr(self, "_demo_active_mask", None)
        if demo_active is not None:
            active &= demo_active.to(active.device)
        env_ids = active.nonzero(as_tuple=False).squeeze(-1)
        if env_ids.numel() == 0:
            logger.log_warning(
                "No active expert rollout row can accept another frame; "
                "new frames are dropped."
            )
            return

        step_ids = self.rollout_steps[env_ids]
        buffer_env_ids = env_ids.to(buffer_device)
        buffer_step_ids = step_ids.to(buffer_device)

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
            self.rollout_buffer["actions"][buffer_env_ids, buffer_step_ids] = (
                action_to_store[env_ids].to(buffer_device)
            )
        self.rollout_buffer["rewards"][buffer_env_ids, buffer_step_ids] = rewards[
            env_ids
        ].to(buffer_device)

        segment_start_steps = getattr(
            self,
            "_demo_active_rollout_start_steps",
            self._demo_active_segment_start_steps,
        )[env_ids]
        segment_steps = step_ids - segment_start_steps
        buffer_keys = set(self.rollout_buffer.keys())
        if "valid" in buffer_keys:
            self.rollout_buffer["valid"][buffer_env_ids, buffer_step_ids] = True
        if "episode_step" in buffer_keys:
            self.rollout_buffer["episode_step"][
                buffer_env_ids, buffer_step_ids
            ] = buffer_step_ids
        if "segment_id" in buffer_keys:
            active_segment_ids = getattr(self, "_demo_active_segment_ids", None)
            segment_ids = (
                active_segment_ids[env_ids].to(buffer_device)
                if active_segment_ids is not None
                else self._demo_active_segment_id
            )
            self.rollout_buffer["segment_id"][
                buffer_env_ids, buffer_step_ids
            ] = segment_ids
        if "segment_step" in buffer_keys:
            self.rollout_buffer["segment_step"][buffer_env_ids, buffer_step_ids] = (
                segment_steps.to(buffer_device)
            )
        if "segment_start" in buffer_keys:
            self.rollout_buffer["segment_start"][buffer_env_ids, buffer_step_ids] = (
                segment_steps.to(buffer_device) == 0
            )

        if terminateds is None:
            terminateds = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        if truncateds is None:
            truncateds = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        terminal = terminateds.to(self.device) | truncateds.to(self.device)
        if "terminated" in buffer_keys:
            self.rollout_buffer["terminated"][buffer_env_ids, buffer_step_ids] = (
                terminateds[env_ids].to(buffer_device)
            )
        if "truncated" in buffer_keys:
            self.rollout_buffer["truncated"][buffer_env_ids, buffer_step_ids] = (
                truncateds[env_ids].to(buffer_device)
            )
        if "segment_end" in buffer_keys:
            self.rollout_buffer["segment_end"][buffer_env_ids, buffer_step_ids] = (
                terminal[env_ids].to(buffer_device)
            )

        self.rollout_steps[env_ids] += 1
        next_env_ids = env_ids[self.rollout_steps[env_ids] < self._max_rollout_steps]
        self._seed_expert_observations(obs, next_env_ids)
        self.current_rollout_step = int(self.rollout_steps.max().item())

    def _write_trajectory_step(self) -> None:
        """Complete one pre-action ``state`` + ``action`` trajectory row."""
        if self._traj_buffer is None:
            return
        max_steps = self._traj_buffer.shape[1]
        step = self._traj_steps
        mask = step < max_steps
        demo_active = getattr(self, "_demo_active_mask", None)
        if demo_active is not None:
            mask &= demo_active.to(mask.device)
        idx = mask.nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            return
        st = step[idx]
        if self._traj_raw_action is not None:
            self._traj_buffer["actions"][idx, st] = self._traj_raw_action[idx]
        self._traj_steps[idx] += 1
        next_env_ids = idx[self._traj_steps[idx] < self._traj_buffer.shape[1]]
        self._seed_trajectory_states(next_env_ids)
        self._traj_raw_action = None

    def _write_rl_rollout_step(
        self,
        obs: EnvObs,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        terminateds: torch.Tensor | None,
        truncateds: torch.Tensor | None,
    ) -> None:
        """Write environment-side fields into an externally managed RL rollout buffer."""
        buffer_device = self.rollout_buffer.device
        self.rollout_buffer["reward"][: self.num_envs, self.current_rollout_step].copy_(
            rewards.to(buffer_device), non_blocking=True
        )
        self.rollout_buffer["done"][: self.num_envs, self.current_rollout_step].copy_(
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
        self.rollout_buffer["terminated"][
            : self.num_envs, self.current_rollout_step
        ].copy_(terminateds.to(buffer_device), non_blocking=True)
        self.rollout_buffer["truncated"][
            : self.num_envs, self.current_rollout_step
        ].copy_(truncateds.to(buffer_device), non_blocking=True)

    def _normalize_demo_action(
        self, action: EnvAction | ProcessedEnvAction
    ) -> EnvAction | ProcessedEnvAction:
        """Normalize one raw action or preserve a controller-ready envelope."""
        if isinstance(action, ProcessedEnvAction):
            value = action.value
            if value.ndim == 0:
                raise ValueError(
                    "Processed demo actions must have a leading environment "
                    "dimension."
                )
            if value.shape[0] != self.num_envs:
                raise ValueError(
                    "Processed demo action batch size must match num_envs."
                )
            return action.snapshot()
        expected_dim = int(np.prod(self.single_action_space.shape))
        return self._normalize_demo_action_tensor(action, expected_dim)

    def _mask_demo_action(
        self,
        action: EnvAction | ProcessedEnvAction,
        active_mask: Sequence[bool],
    ) -> EnvAction | ProcessedEnvAction:
        """Accept an asynchronously completed vector-demo action.

        Raw actions may still require :class:`ActionManager` preprocessing, so
        the actual hold/no-op substitution is applied to the processed command
        in :meth:`_preprocess_action`. Subclasses with a specialized safe action
        may override this hook and return a replacement raw action.

        Args:
            action: Raw normalized action for the shared demo step.
            active_mask: Rows that still participate in the episode.

        Returns:
            The raw action to preprocess.
        """
        self._set_demo_active_mask(active_mask)
        return action

    def _mask_processed_demo_action(self, action: EnvAction) -> EnvAction:
        """Replace inactive processed commands with a safe hold or no-op."""
        active_mask = self._demo_active_mask
        if bool(active_mask.all()):
            return action

        def replace_rows(
            value: torch.Tensor, replacement: torch.Tensor, key: str
        ) -> torch.Tensor:
            if value.ndim < 2 or value.shape[0] != self.num_envs:
                raise ValueError(
                    f"Cannot mask demo {key} action with shape {tuple(value.shape)}; "
                    f"expected a leading vector-env dimension of {self.num_envs}."
                )
            masked = value.clone()
            inactive = ~active_mask.to(value.device)
            masked[inactive] = replacement.to(device=value.device, dtype=value.dtype)[
                inactive
            ]
            return masked

        measured_qpos = self.robot.get_qpos()
        active_qpos = measured_qpos[:, self.active_joint_ids]

        def qpos_replacement(value: torch.Tensor) -> torch.Tensor:
            """Select the measured qpos layout matching the processed command."""
            if value.shape[1:] == active_qpos.shape[1:]:
                return active_qpos
            if value.shape[1:] == measured_qpos.shape[1:]:
                return measured_qpos
            raise ValueError(
                "Cannot construct a qpos hold command for processed demo action "
                f"shape {tuple(value.shape)}; measured active/full layouts are "
                f"{tuple(active_qpos.shape)} and {tuple(measured_qpos.shape)}."
            )

        if isinstance(action, torch.Tensor):
            return replace_rows(action, qpos_replacement(action), "qpos")
        if not isinstance(action, TensorDict):
            raise TypeError(
                "Processed demo actions must be torch.Tensor or TensorDict, "
                f"got {type(action).__name__}."
            )

        masked_action = action.clone()
        supported_key = False
        for key in ("qpos", "qvel", "qf"):
            if key not in masked_action:
                continue
            supported_key = True
            value = masked_action[key]
            replacement = (
                qpos_replacement(value) if key == "qpos" else torch.zeros_like(value)
            )
            masked_action[key] = replace_rows(value, replacement, key)
        if not supported_key:
            raise ValueError(
                "Cannot mask a processed demo TensorDict without qpos, qvel, or qf."
            )
        return masked_action

    def _normalize_demo_action_list(
        self, action_list: Sequence[EnvAction] | torch.Tensor | None
    ) -> Sequence[EnvAction] | torch.Tensor | None:
        """Validate/convert demo action outputs to match single action-space dim."""
        if action_list is None:
            return None

        # Use the per-env action space, not the (batched) ``action_space`` whose
        # shape is ``(num_envs, dim)``. Otherwise demo actions shaped
        # ``(num_envs, dim)`` are rejected with "action dim < expected" for
        # ``num_envs > 1`` (expected would be ``num_envs * dim``).
        expected_dim = int(np.prod(self.single_action_space.shape))

        if isinstance(action_list, torch.Tensor):
            return self._normalize_demo_action_tensor(action_list, expected_dim)

        if not isinstance(action_list, Sequence):
            raise TypeError(
                "create_demo_action_list must return None, a torch.Tensor, or a sequence of actions. "
                f"Got {type(action_list)}."
            )

        normalized_action_list = [
            self._normalize_demo_action(action) for action in action_list
        ]
        return type(action_list)(normalized_action_list)

    def _normalize_demo_action_tensor(
        self, action: EnvAction | torch.Tensor, expected_dim: int
    ) -> EnvAction | torch.Tensor:
        """Normalize one action tensor to the expected action dimension.

        Conversion rule:
        - If last-dim equals action-space dim, keep as-is.
        - If last-dim is larger, slice with ``active_joint_ids``.
        - If last-dim is smaller, raise ``ValueError``.
        """
        if isinstance(action, TensorDict):
            return self._normalize_demo_action_tensordict(action, expected_dim)

        if not isinstance(action, torch.Tensor):
            raise TypeError(
                "Each demo action must be a torch.Tensor or TensorDict. "
                f"Got {type(action)}."
            )

        if action.ndim == 0:
            raise ValueError(
                "Demo action tensor must have at least one dimension with action features on the last axis."
            )

        action_dim = int(action.shape[-1])
        if action_dim == expected_dim:
            return action
        if action_dim < expected_dim:
            raise ValueError(
                "Demo action dim is smaller than action space dim and cannot be auto-converted. "
                f"Got action dim={action_dim}, expected={expected_dim}."
            )
        return self._slice_action_with_active_joint_ids(
            action, action_dim, expected_dim
        )

    def _normalize_demo_action_tensordict(
        self, action: TensorDict, expected_dim: int
    ) -> TensorDict:
        """Normalize tensor entries in a TensorDict action payload."""
        converted_action = action.clone()
        for key in ("qpos", "qvel", "qf"):
            if key not in converted_action:
                continue
            value = converted_action[key]
            if value.ndim == 0:
                raise ValueError(
                    f"Demo action TensorDict['{key}'] must have at least one dimension."
                )
            action_dim = int(value.shape[-1])
            if action_dim == expected_dim:
                continue
            if action_dim < expected_dim:
                raise ValueError(
                    f"Demo action TensorDict['{key}'] dim={action_dim} is smaller than expected action dim={expected_dim}."
                )
            converted_action[key] = self._slice_action_with_active_joint_ids(
                value, action_dim, expected_dim
            )
        return converted_action

    def _slice_action_with_active_joint_ids(
        self, action: torch.Tensor, action_dim: int, expected_dim: int
    ) -> torch.Tensor:
        """Slice a high-dimensional action to active joints.

        This is used when demo actions are generated in full-DoF form while the
        environment action-space only controls active joints.
        """
        if len(self.active_joint_ids) != expected_dim:
            raise ValueError(
                "Cannot convert demo action by active_joint_ids because their length does not match the action space dim. "
                f"len(active_joint_ids)={len(self.active_joint_ids)}, expected={expected_dim}."
            )

        if len(self.active_joint_ids) == 0:
            raise ValueError(
                "Cannot convert demo action by active_joint_ids because active_joint_ids is empty."
            )

        return action[..., self.active_joint_ids]

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

        def active_joint_command(command: torch.Tensor, key: str) -> torch.Tensor:
            """Normalize full-robot commands to the active-joint layout."""
            active_dim = len(self.active_joint_ids)
            if command.shape[-1] == active_dim:
                return command.to(self.device)
            full_dim = int(self.robot.get_qpos().shape[-1])
            if command.shape[-1] == full_dim:
                return command[..., self.active_joint_ids].to(self.device)
            raise ValueError(
                f"Processed {key} action has dim {command.shape[-1]}; expected "
                f"active-joint dim {active_dim} or full robot dim {full_dim}."
            )

        if isinstance(action, TensorDict):
            # Support multiple control modes simultaneously
            action = action.clone()
            if "qpos" in action:
                action["qpos"] = active_joint_command(action["qpos"], "qpos")
                self.robot.set_qpos(
                    qpos=action["qpos"], joint_ids=self.active_joint_ids
                )
            if "qvel" in action:
                action["qvel"] = active_joint_command(action["qvel"], "qvel")
                self.robot.set_qvel(
                    qvel=action["qvel"], joint_ids=self.active_joint_ids
                )
            if "qf" in action:
                action["qf"] = active_joint_command(action["qf"], "qf")
                self.robot.set_qf(qf=action["qf"], joint_ids=self.active_joint_ids)
        elif isinstance(action, torch.Tensor):
            action = active_joint_command(action, "qpos")
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

    def _preprocess_action(self, action: EnvAction | ProcessedEnvAction) -> EnvAction:
        """Apply raw preprocessing once and stash the executed controller action."""
        is_processed = isinstance(action, ProcessedEnvAction)
        if is_processed:
            action = action.value
        if self._traj_buffer is not None:
            self._traj_raw_action = (
                action.clone() if hasattr(action, "clone") else action
            )
        if self.action_manager is not None and not is_processed:
            action = self.action_manager.process_action(action, mode="pre")
        elif not is_processed:
            action = super()._preprocess_action(action)
        if getattr(self, "_demo_no_auto_reset", False):
            action = self._mask_processed_demo_action(action)
        return action

    def _postprocess_action(self, action):
        if self.action_manager is not None:
            return self.action_manager.process_action(action, mode="post")
        return super()._postprocess_action(action)

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
            if "env_map" in self.cfg.light.indirect:
                path = get_data_path(self.cfg.light.indirect["env_map"])
                self.sim.set_indirect_lighting(path)

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

        Note:
            Subclass outputs are automatically post-processed by the base class:
            action last-dimension must match ``single_action_space``. If larger,
            actions are sliced by ``active_joint_ids``; if smaller, ``ValueError``
            is raised.
        """
        raise NotImplementedError(
            "The method 'create_demo_action_list' must be implemented in subclasses."
        )

    def compile_expert_program(
        self,
        program: ExpertProgramCfg,
    ) -> CompiledProgram:
        """Compile a configured Expert Program through explicit scene providers.

        Declarative environments override this hook to supply their authoritative
        scene registry/resolver to :class:`ExpertProgramCompiler`. Keeping the
        provider boundary explicit prevents the base environment from inferring
        identities or scanning mutable simulator internals.

        Args:
            program: Strict Expert Program configuration attached to ``cfg``.

        Returns:
            Provider-free compiled program ready for runtime assembly.

        Raises:
            NotImplementedError: If an environment enables ``expert_program``
                without supplying the compiler/provider integration.
        """
        raise NotImplementedError(
            "An environment with cfg.expert_program must implement "
            "compile_expert_program() using an explicit scene resolver."
        )

    def create_expert_program_bridge(
        self,
        program: CompiledProgram,
    ) -> AtomicDemoBridge:
        """Create the Gym demo bridge through explicit runtime-port factories.

        Declarative environments override this hook to assemble ``SkillRuntime``
        and Gym-aware command/clock/post-policy/validator ports. The returned
        bridge must emit commands through normal ``env.step()`` processing.

        Args:
            program: Compiled provider-free Expert Program.

        Returns:
            Atomic demo bridge whose segments are consumed lazily.

        Raises:
            NotImplementedError: If no explicit runtime factory is available.
        """
        raise NotImplementedError(
            "An environment with cfg.expert_program must implement "
            "create_expert_program_bridge() using explicit runtime ports."
        )

    def create_demo_segments(self, *args, **kwargs) -> Iterable[DemoSegment] | None:
        """Create the semantic segments that make up one task episode.

        When ``cfg.expert_program`` is configured, the environment compiles it
        through an explicit scene-provider hook and creates an atomic demo bridge
        through an explicit runtime-port factory hook. Otherwise, the default
        adapter wraps ``create_demo_action_list`` in one segment. Multi-object
        tasks may return a lazy generator so each segment can be planned from
        the scene state left by the previous one.

        Args:
            *args: Positional arguments forwarded to the legacy planner.
            **kwargs: Keyword arguments forwarded to the legacy planner.

        Returns:
            Segment sequence, or ``None`` when planning fails.
        """
        expert_program = getattr(getattr(self, "cfg", None), "expert_program", None)
        if expert_program is not None:
            compiled_program = self.compile_expert_program(expert_program)
            bridge = self.create_expert_program_bridge(compiled_program)
            return bridge.iter_segments()

        actions = self.create_demo_action_list(*args, **kwargs)
        if actions is None:
            return None
        return (DemoSegment(actions=actions, name="legacy"),)

    def save_trajectory(self, path: str, env_ids: Sequence[int] | None = None) -> str:
        """Save a causally aligned trajectory to a ``.pt`` file.

        ``states[t]`` is the state immediately before ``actions[t]`` is applied,
        matching the frame alignment used by expert/LeRobot trajectories.

        Args:
            path: Destination ``.pt`` file path.
            env_ids: Env indices to save (default: all). Each saved env's actual
                recorded length is stored in ``meta["lengths"]``.

        Raises:
            RuntimeError: If trajectory recording was never enabled.
        """
        if self._traj_buffer is None:
            raise RuntimeError(
                "Trajectory recording is not enabled (set cfg.record_trajectory=True)."
            )
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        env_ids = list(env_ids)
        if not env_ids:
            raise ValueError("env_ids must contain at least one environment row.")
        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        lengths = self._traj_steps[env_ids_t]
        max_len = int(lengths.max().item()) if len(env_ids) > 0 else 0
        sub = self._traj_buffer[env_ids_t]
        states = sub["states"][:, :max_len].clone()
        actions = sub["actions"][:, :max_len].clone()
        meta = {
            "lengths": lengths.tolist(),
            "num_steps": max_len,
            "num_envs": int(len(env_ids)),
            # ``dt`` historically represented trajectory timing. Keep the key
            # for compatibility, but make it describe the actual interval
            # between recorded environment steps rather than a physics substep.
            "dt": self.step_dt,
            "physics_dt": self.physics_dt,
            "sim_steps_per_control": int(self.cfg.sim_steps_per_control),
            "step_dt": self.step_dt,
            "control_frequency": self.control_frequency,
            "active_joint_ids": list(self.active_joint_ids),
            "robot_uid": self.robot.uid,
            "robot_dof": int(self.robot.dof),
            "articulation_uids": list(self.sim._articulations.keys()),
            "articulation_dofs": {
                uid: int(art.dof) for uid, art in self.sim._articulations.items()
            },
            "rigid_object_uids": list(self.sim._rigid_objects.keys()),
            "env_ids": [int(e) for e in env_ids],
            "demo_episodes": [
                self.get_demo_episode_metadata(int(env_id)) for env_id in env_ids
            ],
        }
        torch.save({"states": states, "actions": actions, "meta": meta}, path)
        return path

    def _save_trajectory_for_env(self, env_id: int) -> str | None:
        """Persist one explicitly committed environment trajectory.

        I/O errors are intentionally propagated: reset is the commit boundary,
        and clearing the in-memory trajectory after a failed write would report
        success while losing committed data.
        """
        if self._traj_buffer is None or not self.cfg.trajectory_auto_save:
            return None
        if int(self._traj_steps[env_id].item()) == 0:
            return None
        base = self.cfg.trajectory_save_dir
        if base is None:
            base = os.path.join(
                EMBODICHAIN_DEFAULT_DATA_ROOT, "trajectories", self._traj_run_id
            )
        os.makedirs(base, exist_ok=True)
        path = os.path.join(base, f"traj_env{env_id}_{self._traj_save_count:06d}.pt")
        saved_path = self.save_trajectory(path, env_ids=[env_id])
        self._traj_save_count += 1
        return saved_path

    def _discard_pending_recordings(self) -> None:
        """Abort recorder state that has not crossed an explicit reset commit."""
        errors: list[str] = []
        if self.cfg.events and self.event_manager is not None:
            from embodichain.lab.gym.envs.managers.record import record_camera_data

            for mode_cfgs in self.event_manager._mode_functor_cfgs.values():
                for functor_cfg in mode_cfgs:
                    if isinstance(functor_cfg.func, record_camera_data):
                        recorder = functor_cfg.func
                        recorder_name = type(recorder).__name__
                        try:
                            recorder.discard_and_clear()
                        except Exception as error:
                            errors.append(f"{recorder_name} discard: {error}")
                        try:
                            recorder.finalize()
                        except Exception as error:
                            errors.append(f"{recorder_name} finalize: {error}")

        if self._traj_steps is not None:
            try:
                self._traj_steps.zero_()
            except Exception as error:
                errors.append(f"trajectory discard: {error}")
        if self.rollout_buffer is not None and self._rollout_buffer_mode != "rl":
            try:
                env_ids = torch.arange(self.num_envs, device=self.device)
                self._clear_expert_rollout_rows(env_ids)
                self.rollout_steps.zero_()
                self.current_rollout_step = 0
            except Exception as error:
                errors.append(f"expert rollout discard: {error}")

        if errors:
            raise RuntimeError(
                f"Failed to abort {len(errors)} pending recorder operation(s): "
                + "; ".join(errors)
            )

    def close(self, *, exit_process: bool | None = None) -> None:
        """Abort pending data, finalize committed writes, and release resources.

        Closing is idempotent and is never an implicit episode commit. A demo
        episode enters the dataset only through ``reset(save_data=True)``;
        partial data left by failure, cancellation, or interpreter shutdown is
        discarded before recorder finalization.

        Args:
            exit_process: Forwarded to :meth:`SimulationManager.destroy` after
                successful cleanup. Error paths always disable process exit so
                the durability exception can propagate.

        Raises:
            RuntimeError: If one or more recorders fail their durability barrier.
        """
        close_lock = getattr(self, "_close_lock", None)
        if close_lock is None:
            close_lock = threading.RLock()
            self._close_lock = close_lock

        with close_lock:
            if getattr(self, "_closed", False):
                close_error = getattr(self, "_close_error", None)
                if close_error is not None:
                    raise close_error
                return

            errors: list[Exception] = []
            try:
                self._discard_pending_recordings()
            except Exception as error:
                errors.append(error)
            if self.dataset_manager:
                try:
                    self.dataset_manager.finalize()
                except Exception as error:
                    errors.append(error)

            # Report before sim.destroy(): the default destroy path exits the
            # process, so diagnostics and durability checks must finish first.
            try:
                self._profiler.report()
            except Exception as error:
                errors.append(error)

            if errors:
                # Queue simulator cleanup without hiding persistence failures
                # behind SimulationManager's default os._exit(0).
                try:
                    self.sim.destroy(exit_process=False)
                except Exception as error:
                    errors.append(error)
                messages = "; ".join(str(error) for error in errors)
                close_error = RuntimeError(
                    f"Failed to close EmbodiedEnv cleanly: {messages}"
                )
                self._close_error = close_error
                self._closed = True
                raise close_error from errors[0]

            try:
                if exit_process is None:
                    self.sim.destroy()
                else:
                    self.sim.destroy(exit_process=exit_process)
            except Exception as error:
                self._close_error = error
                self._closed = True
                raise
            self._closed = True
