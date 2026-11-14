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

from collections.abc import Callable, Mapping, Sequence, Iterator
from contextlib import contextmanager
from math import isfinite, log
from functools import wraps
from datetime import datetime
import logging
import os
import threading
import copy
from pathlib import Path
import tempfile
from uuid import uuid4
import embodichain
import torch
import numpy as np
import gymnasium as gym

from embodichain.lab.sim.cfg.fluid import FluidObjectCfg

from dataclasses import MISSING, fields
from typing import (
    TYPE_CHECKING,
    Dict,
    Union,
    Tuple,
    Any,
    Iterable,
    List,
    Literal,
    Optional,
)
from tensordict import TensorDict

from embodichain.lab.sim.cfg import (
    RobotCfg,
    RobotPresetCfg,
    RigidObjectCfg,
    VisualObjectCfg,
    RigidObjectGroupCfg,
    ArticulationCfg,
    UsdImportCfg,
    CableObjectCfg,
    LightCfg,
)
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.sensors import BaseSensor, Camera, SensorCfg
from embodichain.lab.sim.types import EnvObs, EnvAction
from embodichain.lab.gym.envs import BaseEnv, EnvCfg
from embodichain.lab.gym.envs._physical_replay import _PhysicalReplay, _ReplaySurface
from embodichain.lab.gym.utils._replay import _AuthoredJointStream
from embodichain.lab.gym.envs._startup_summary import format_functor_summary
from embodichain.lab.gym.envs.objectives import (
    MeasuredRigidObjectState,
    OrderedPlacementObjective,
    OrderedPlacementObjectiveCfg,
)
from embodichain.lab.gym.envs.demo import (
    DEMO_SCHEMA_VERSION,
    DemoExecutionCfg,
    DemoEpisodeResult,
    DemoSegment,
    DemoSegmentResult,
    _as_bool_tuple,
)
from embodichain.lab.gym.utils.trajectory_state import FluidTrajectoryRecorder
from embodichain.lab.gym.envs.types import ControllerAction
from embodichain.lab.gym.envs.expert_trajectory import (
    ExpertTrajectoryCfg,
    build_expert_action_spec,
    encode_expert_action,
)
from embodichain.lab.gym.envs.managers import (
    EventManager,
    ObservationManager,
    RewardManager,
    ActionManager,
    ActionTrace,
    DatasetManager,
    FunctorCfg,
    TerminationManager,
    RLTaskCfg,
)
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.gym.utils.gym_utils import (
    _collect_sensor_cfgs,
    build_trajectory_buffer,
    init_rollout_buffer_from_gym_space,
    _init_raw_rollout,
    _write_raw_rollout,
    _close_raw_rollout,
)
from embodichain.lab.gym.envs.managers.observation_types import ObservationFrame
from embodichain.lab.gym.envs.task_program._final_acceptance import (
    _capture_evaluation_frame,
    _FINAL_OBSERVATIONS_KEY,
    _FinalMeasurement,
    _deployment_final_rules,
    _deployment_process_rules,
    _evaluate_process_frame,
    _export_final_observations,
    _final_acceptance_rows,
    _measure_final_conditions,
)
from embodichain.lab.gym.utils.trajectory_state import (
    trajectory_observation_cfgs,
    _evaluation_observation_cfgs,
    _validate_evaluation_observation_bindings,
    export_observation_history,
)
from embodichain.data_pipeline.recording import build_recording_provenance
from embodichain.utils import configclass, logger
from embodichain.utils.string import string_to_callable
from embodichain.data import get_data_path
from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATA_ROOT

if TYPE_CHECKING:
    from embodichain.lab.task_program.runtime.expansion import (
        TaskProgramNominalSourceFactory,
        TaskProgramPlanTransformFactory,
    )
    from embodichain.lab.sim.motion.expansion import (
        CombinedExpansionProfile,
    )
    from embodichain.lab.task_program import CompiledTaskProgram, TaskProgramCfg
    from embodichain.lab.task_program.integrations import (
        TaskProgramAdapterFactory,
        TaskProgramEnvironmentAdapter,
        TaskProgramExpansionRecord,
    )
    from embodichain.lab.gym.envs.task_program.bridge import TaskProgramDemoBridge

__all__ = ["EmbodiedEnvCfg", "EmbodiedEnv"]


def _outcome_row(value: Any, env_id: int) -> Any:
    if isinstance(value, torch.Tensor):
        return _outcome_row(value[env_id].detach().cpu().tolist(), env_id)
    if isinstance(value, Mapping):
        return {key: _outcome_row(child, env_id) for key, child in value.items()}
    if isinstance(value, list):
        return [_outcome_row(child, env_id) for child in value]
    if isinstance(value, float) and not isfinite(value):
        return None
    return value


def _demo_recorded_length(env: EmbodiedEnv, env_id: int) -> int:
    rollout_steps = getattr(env, "rollout_steps", None)
    steps = (
        rollout_steps
        if rollout_steps is not None and env.rollout_buffer is not None
        else env._demo_steps
    )
    return int(steps[env_id].item())


def _demo_last_rollout_flags(
    env: EmbodiedEnv, env_id: int, length: int
) -> tuple[bool, bool]:
    buffer = env.rollout_buffer
    if buffer is None or length == 0:
        return False, False
    return (
        bool(
            "terminated" in buffer.keys() and buffer["terminated"][env_id, length - 1]
        ),
        bool("truncated" in buffer.keys() and buffer["truncated"][env_id, length - 1]),
    )


def _segment_outcome(segment: Mapping[str, Any], env_id: int) -> dict[str, Any]:
    """Project cached bridge and legacy validator evidence onto one row."""
    validation = segment.get("metadata", {}).get("validation")
    if not isinstance(validation, Mapping):
        validation = {}
    env_ids = validation.get("env_ids", ())
    index = env_ids.index(env_id) if env_id in env_ids else None
    outcome = {
        "segment_id": segment["segment_id"],
        "attempt_id": segment.get("attempt_id"),
        "start_step": segment.get("start_step"),
        "end_step": segment.get("end_step"),
        "kind": segment.get("outcome_kind"),
        "accepted": segment["success"],
        "reason": segment.get("failure_reason"),
        "validator_accepted": segment.get("validator_accepted"),
    }
    for name in ("runtime_success", "post_policy_success", "accepted"):
        mask = validation.get(f"{name}_mask")
        outcome[f"bridge_{name}"] = (
            mask[index] if index is not None and mask is not None else None
        )
    return outcome


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
    - **robot**: `RobotCfg | RobotPresetCfg` (required) — one portable robot
        definition or replace-only complete alternatives selected by the active
        physics backend.
    - **control_parts**: Optional[List[str]] — named robot parts to control. If
        `None`, all controllable joints are used.
    - **active_joint_ids**: List[int] — explicit joint indices to use for
        control (alternative to `control_parts`).
    - **sensor**: List[`SensorCfg`] — independent sensors. Mounted cameras
        declared by `robot.sensors` are included in the effective sensor suite.
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

    robot: RobotCfg | RobotPresetCfg = MISSING

    control_parts: list[str] | None = None
    """List of robot parts to control. If None, all controllable joints will be used. 
    This is useful when we want to control only a subset of the robot joints for certain tasks or demonstrations.
    """

    active_joint_ids: List[int] = []
    """List of active joint IDs for control. User also can directly specify the active joint IDs instead of control \
    parts. This is useful when the control parts are not well defined or we want to have more fine-grained control.
    """

    sensor: List[SensorCfg] = []
    """Independent sensors, combined with the robot's mounted sensor suite."""
    enable_sensor: bool = True
    """Whether configured sensors are instantiated and sampled.

    Independent declarations and robot-mounted defaults remain available when
    a deployment only needs proprioception. Set this to ``False`` to skip all
    sensor creation, including camera-group allocation and image fetching.
    """

    light: EnvLightCfg = EnvLightCfg()

    ground_plane_visible: bool | None = None
    """Native rendering visibility of the default ground plane.

    ``None`` hides the ground when an active HDR indirect-lighting randomizer
    is configured, and shows it otherwise. Explicit booleans override that
    default. Visibility affects native windows and cameras without changing
    ground collisions or the Viser grid.
    """

    visual_object: list[VisualObjectCfg] = []
    """Render-only assets with no collision or rigid-body state."""

    background: List[RigidObjectCfg] = []

    rigid_object: List[RigidObjectCfg] = []

    rigid_object_group: List[RigidObjectGroupCfg] = []

    articulation: List[ArticulationCfg] = []

    usd_scene: UsdImportCfg | None = None
    """Optional whole-scene USD/USDZ or standard AssetRevision ZIP import.

    Robot construction remains owned by the embodiment. Imported resources
    coexist with the ordinary object lists and must have distinct UIDs.
    """

    default_ground: bool | None = None
    """Explicit default-plane selection; USD scenes default to no added plane.

    Without a USD scene, ``None`` preserves ``sim_cfg.default_ground``.
    """
    cable_object: list[CableObjectCfg] = []
    """Fixed-topology cables declared before simulation preparation."""
    fluid_object: list[FluidObjectCfg] = []
    """Fixed-count fluids owned by the physical environment component."""

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

    rollout_action_semantics: Literal["expert_controller_command", "policy_request"] = (
        "expert_controller_command"
    )
    """Meaning of episode rollout actions, independently of the expert schema.

    ``policy_request`` stores the flat ActionManager input and requires an
    ActionManager. Managed position-only experts record the converted manager
    request while retaining their exact FK execution target. It also enables
    raw-policy recording without a DatasetManager.
    The default preserves active-joint expert-controller recording.
    """

    objective: OrderedPlacementObjectiveCfg | None = None
    """Optional measured physical objective, independent of task success and persistence."""

    task_instruction: str | None = None
    """Overall task language for handwritten/direct-control demonstrations.

    Task Programs own their instruction in the program configuration. Explicit
    episode instructions take precedence; recorder instructions are legacy
    fallbacks and are not required to enable task language.
    """

    expert_trajectory: ExpertTrajectoryCfg = ExpertTrajectoryCfg()
    """Source-neutral expert trajectory control and recording settings."""

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
    ``trajectory_save_dir`` before an explicitly committed episode reset.
    Closing discards trajectories that were never committed."""

    rl_task: embodichain.lab.gym.envs.managers.RLTaskCfg | None = None
    """Optional measured RL success, failure and truncation conditions.

    Configured conditions use cached per-step outcomes and recorded reset-target
    context. Task Program declarations may coexist only in explicit policy mode.
    """

    task_program: TaskProgramCfg | None = None
    """Optional declarative Task Program used to generate demo segments.

    The program remains inert until :meth:`EmbodiedEnv.create_demo_segments`
    requests an explicit environment compiler and bridge through the dedicated
    hooks. No live provider, planner, or callable is stored in this config.
    """

    task_program_execution: Literal["expert", "policy"] = "expert"
    """Action producer; policy mode loads declarations without an expert adapter."""
    task_program_action_owners: dict[str, str] = {}
    """Explicit policy ownership of the declared robot resource IDs."""

    replay: _AuthoredJointStream | None = None
    """Explicit physical command-stream replay, independent of semantic lowerers."""

    replay_surface: _ReplaySurface | None = None
    """Surface asset used by a configured native replay."""


class _EpisodeMetadataRows(Sequence):
    """Retain reset-time metadata, materializing row copies when recording uses them."""

    def __init__(self, count: int, template: dict[str, Any]) -> None:
        pending = (copy.deepcopy(template),)
        self._rows = [pending] * count

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int | slice) -> dict[str, Any] | list[dict[str, Any]]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        record = self._rows[index]
        if isinstance(record, tuple):
            record = copy.deepcopy(record[0])
            episode_uuid = str(uuid4())
            record.update(
                env_id=index % len(self),
                episode_uuid=episode_uuid,
                source_episode_uuid=episode_uuid,
            )
            self._rows[index] = record
        return record

    def __setitem__(self, index: int, record: dict[str, Any]) -> None:
        self._rows[index] = record

    def reset_rows(self, ids: Sequence[int], template: dict[str, Any]) -> None:
        pending = (copy.deepcopy(template),)
        for index in ids:
            self._rows[index] = pending


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
    - affordance_datas: The affordance data that can be used to store the intermediate results or information
    """

    _defer_initialization_summary: bool = True
    _manager_summary_fields: tuple[tuple[str, str], ...] = (
        ("EventManager", "event_manager"),
        ("ObservationManager", "observation_manager"),
        ("ActionManager", "action_manager"),
        ("RewardManager", "reward_manager"),
        ("DatasetManager", "dataset_manager"),
    )

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

    def __init__(
        self,
        cfg: EmbodiedEnvCfg,
        *,
        task_program_adapter_factory: TaskProgramAdapterFactory | None = None,
        **kwargs,
    ):
        replay = getattr(cfg, "replay", None)
        if replay is not None:
            cfg = copy.deepcopy(cfg)
        rl_task = getattr(cfg, "rl_task", None)
        if rl_task is not None:
            if not isinstance(rl_task, RLTaskCfg):
                raise TypeError("rl_task must be an RLTaskCfg or None.")
            if getattr(cfg, "task_program_execution", "expert") != "policy" and (
                getattr(cfg, "task_program", None) is not None
                or task_program_adapter_factory is not None
            ):
                raise ValueError(
                    "rl_task and task_program cannot share one deployment."
                )
        if getattr(cfg, "task_program_execution", "expert") not in ("expert", "policy"):
            raise ValueError("task_program_execution must be expert or policy.")
        if getattr(cfg, "task_program_execution", "expert") == "policy" and (
            rl_task is None
            or cfg.task_program is None
            or task_program_adapter_factory is None
        ):
            raise ValueError(
                "Policy Task Program declarations require rl_task and a registered deployment."
            )
        self.task_program_progress = None
        self._program_transition = 0
        self.termination_manager: (
            embodichain.lab.gym.envs.managers.TerminationManager | None
        ) = None
        self._task_initial_states: list[dict[str, Any]] = []
        if task_program_adapter_factory is not None:
            from embodichain.lab.task_program.integrations import (
                TaskProgramAdapterFactory,
            )

            if not isinstance(
                task_program_adapter_factory,
                TaskProgramAdapterFactory,
            ):
                raise TypeError(
                    "task_program_adapter_factory must implement "
                    "TaskProgramAdapterFactory or be None."
                )
        self.physical_objective: OrderedPlacementObjective | None = None
        self._physical_objective_object = None
        self.affordance_datas = {}
        self._task_program_adapter: TaskProgramEnvironmentAdapter | None = None
        self._active_task_program_bridge: TaskProgramDemoBridge | None = None

        extensions = getattr(cfg, "extensions", {}) or {}
        if replay is not None and any(
            name.startswith("_") or hasattr(cfg, name) or hasattr(self, name)
            for name in extensions
        ):
            raise ValueError(
                "Extensions cannot override configured acceptance or environment fields."
            )

        if cfg.task_program_execution == "policy":
            from embodichain.lab.task_program.integrations.progress import (
                _validate_policy_extensions,
            )

            _validate_policy_extensions(cfg)
            if any(hasattr(self, name) for name in extensions):
                raise ValueError(
                    "Policy extensions cannot override environment-owned state or methods."
                )
        for name, value in extensions.items():
            setattr(cfg, name, value)
            setattr(self, name, value)

        self._physical_replay = None if replay is None else _PhysicalReplay(cfg, replay)
        self.event_manager: EventManager | None = None
        self.observation_manager: ObservationManager | None = None
        self.reward_manager: RewardManager | None = None
        self.action_manager: ActionManager | None = None
        self.dataset_manager: DatasetManager | None = None
        if cfg.rollout_action_semantics not in {
            "expert_controller_command",
            "policy_request",
        }:
            raise ValueError("Unknown rollout_action_semantics.")
        self._explicit_policy_rollout = cfg.rollout_action_semantics == "policy_request"
        self._record_raw_actions = self._explicit_policy_rollout
        self._last_action_manager_qpos: torch.Tensor | None = None
        self._last_action_trace: ActionTrace | None = None
        self._expert_controller_qpos_history: list[list[torch.Tensor]] | None = None

        registration = getattr(task_program_adapter_factory, "registration", None)
        if cfg.task_program_execution == "policy":
            from embodichain.lab.task_program.integrations.progress import (
                _validate_policy_deployment,
            )

            _validate_policy_deployment(cfg, registration)
        from embodichain.lab.gym.envs.objectives.ordered_placement import (
            _objective_rule,
        )

        self._physical_objective_rule = _objective_rule(
            cfg.objective,
            getattr(registration, "evaluation_conditions", {}),
        )
        self._final_acceptance_rules = _deployment_final_rules(registration)
        self._final_program_completion: tuple[tuple[int, ...], torch.Tensor] | None = (
            None
        )
        self._final_bridge_episodes: tuple[int, ...] | None = None
        self._process_evaluation_rules = _deployment_process_rules(registration)
        self._sealed_final_observations: dict[int, ObservationFrame] = {}
        self._pending_final_acceptance: (
            tuple[
                tuple[int, ...],
                tuple[bool, ...],
                torch.Tensor,
                _FinalMeasurement | None,
            ]
            | None
        ) = None
        self._evaluation_recording = (
            _evaluation_observation_cfgs(registration)
            if getattr(registration, "evaluation_conditions", None)
            else {}
        )
        super().__init__(cfg, **kwargs)

        try:
            if self._evaluation_recording:
                _validate_evaluation_observation_bindings(
                    self, self._evaluation_recording
                )
                from embodichain.lab.task_program.integrations.simulation.bindings import (
                    _bind_evaluation_joint_names,
                )

                bound = _bind_evaluation_joint_names(
                    registration.evaluation_conditions, self.robot
                )
                self._final_acceptance_rules = tuple(
                    (ref, bound[ref]) for ref, _ in self._final_acceptance_rules
                )
                self._process_evaluation_rules = tuple(
                    (ref, bound[ref]) for ref, _ in self._process_evaluation_rules
                )
                self._physical_objective_rule = _objective_rule(cfg.objective, bound)
                if cfg.task_program_execution == "policy":
                    from embodichain.lab.task_program.integrations.progress import (
                        TaskProgramProgress,
                        _validate_policy_action_coverage,
                    )

                    _validate_policy_action_coverage(
                        registration, self.robot, self.action_manager
                    )

                    self.task_program_progress = TaskProgramProgress(
                        registration.catalog.preflight(
                            cfg.task_program, strict_contracts=True
                        ),
                        bound,
                        env_ids=tuple(range(self.num_envs)),
                        # Unstarted rows expect their first reset (episode 0).
                        # Frames remain unqualified while the owner is at -1.
                        episode_ids=tuple(
                            max(0, episode)
                            for episode in self.episode_context.episode_ids
                        ),
                        device=self.device,
                        final_rules=dict(self._final_acceptance_rules),
                        process_rules=dict(self._process_evaluation_rules),
                    )
            if self._explicit_policy_rollout and self.action_manager is None:
                raise ValueError(
                    "policy_request rollout recording requires an ActionManager."
                )
            self._initialize_physical_objective()
            self.expert_action_spec = build_expert_action_spec(
                joint_names=[
                    self.robot.joint_names[joint_id]
                    for joint_id in self.active_joint_ids
                ],
                joint_command_mode=self.cfg.expert_trajectory.joint_command_mode,
            )
            self._expert_action_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_envs, self.expert_action_spec.width),
                dtype=np.float32,
            )

            if (
                task_program_adapter_factory is not None
                and cfg.task_program_execution == "expert"
            ):
                from embodichain.lab.task_program.integrations import (
                    TaskProgramEnvironmentAdapter,
                )

                adapter = task_program_adapter_factory.create_adapter(self)
                if type(adapter) is not TaskProgramEnvironmentAdapter:
                    raise TypeError(
                        "TaskProgramAdapterFactory.create_adapter() must return "
                        "exactly TaskProgramEnvironmentAdapter."
                    )
                self._task_program_adapter = adapter

            dataset_terms = getattr(self.cfg.dataset, "__dict__", self.cfg.dataset)
            if dataset_terms and not self.cfg.filter_dataset_saving:
                self.dataset_manager = DatasetManager(self.cfg.dataset, self)
                self.cfg.init_rollout_buffer = True
                self._record_raw_actions |= self.dataset_manager.requires_raw_actions

            # One storage owner; policy, raw-state and training windows are views.
            self._rollout_storage = TensorDict({}, batch_size=[self.num_envs])
            self._expert_obs_buffer: TensorDict | None = None
            self.rollout_buffer: TensorDict | None = None
            self._max_rollout_steps = 0
            self._rollout_buffer_mode: str | None = None
            self._rollout_action_semantics: str | None = None
            if self.cfg.init_rollout_buffer:
                rollout_action_space = (
                    self.action_space
                    if self._record_raw_actions
                    else self._expert_action_space
                )
                policy_storage = init_rollout_buffer_from_gym_space(
                    obs_space=self.observation_space,
                    action_space=rollout_action_space,
                    max_episode_steps=self.max_episode_steps + 1,
                    num_envs=self.num_envs,
                    device=self.device,
                )
                self._rollout_storage["policy"] = policy_storage
                self._expert_obs_buffer = policy_storage["obs"]
                self.rollout_buffer = policy_storage[:, : self.max_episode_steps]
                self._max_rollout_steps = self.max_episode_steps
                self._rollout_buffer_mode = "expert"
                self._rollout_action_semantics = (
                    "policy_request"
                    if self._record_raw_actions
                    else "expert_controller_command"
                )

            self._traj_buffer: TensorDict | None = None
            self._fluid_trajectory_recorder: FluidTrajectoryRecorder | None = None
            self._traj_steps: torch.Tensor | None = None
            self._traj_raw_action: EnvAction | None = None
            self._trajectory_action_kind: str | None = None
            # Recording methods allocate row histories on their first sample.
            # Training resets then avoid downloading IDs to clear empty lists.
            self._raw_action_history: list[list[EnvAction]] | None = None
            self._expert_controller_qpos_history = None
            self._traj_save_count = 0
            self._traj_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.cfg.record_trajectory:
                self._trajectory_action_kind = (
                    "policy" if self.action_manager is not None else "expert"
                )
                self._traj_buffer = build_trajectory_buffer(
                    env=self,
                    max_steps=self.max_episode_steps + 1,
                    num_envs=self.num_envs,
                    device=self.device,
                    uids=self.cfg.trajectory_uids,
                    action_space=self._trajectory_action_space(),
                )
                self._rollout_storage["trajectory"] = self._traj_buffer
                fields = {
                    "/".join(path): values
                    for path, values in self._traj_buffer["states"].items(
                        include_nested=True, leaves_only=True
                    )
                }
                _init_raw_rollout(
                    self._rollout_storage,
                    self.episode_context,
                    fields,
                    self._traj_buffer["actions"][:, : self.max_episode_steps],
                )
                self.observation_manager.bind_rollout_storage(self._rollout_storage)
                # Private aliases preserve existing replay/persistence adapters;
                # acquisition, cursors and lifetime belong to the host rollout.
                self._traj_steps = self._rollout_storage["raw_steps"]

            self.rollout_steps = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self._rollout_storage["policy_steps"] = self.rollout_steps
            self._demo_steps = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self.current_rollout_step = 0

            # Segment recording is intentionally separate from task planning. The
            # common demo executor updates this context while the regular rollout
            # writer turns it into per-frame annotations.
            self._demo_episode_index = 0
            self._demo_execution_cfg = DemoExecutionCfg()
            self._demo_attempt_id = 0
            self._demo_continuity_id = 0
            self._demo_program_run_id = "0:0"
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
            self._demo_metadata_initialized = False
            self._demo_episode_metadata: list[dict[str, Any]] = [
                {} for _ in range(self.num_envs)
            ]

            self.episode_success_status: torch.Tensor = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self._closed = False
            self._close_error: BaseException | None = None
            self._close_lock = threading.RLock()

            self._log_initialization_summary()
        except Exception:
            if self.dataset_manager is not None:
                try:
                    self.dataset_manager.finalize()
                except Exception as cleanup_error:
                    logger.log_warning(
                        f"Failed to finalize dataset after initialization error: {cleanup_error!r}"
                    )
            self._cleanup_failed_initialization()
            raise

    def _initialization_summary_lines(self) -> list[str]:
        """Append a separate functor table after the environment summary."""
        lines = super()._initialization_summary_lines()
        if not lines:
            return lines
        managers = []
        for name, attribute in self._manager_summary_fields:
            manager = getattr(self, attribute, None)
            if manager is not None:
                managers.append(
                    (name, manager, self._manager_functor_groups(name, manager))
                )
        details = format_functor_summary(
            managers, full=self.sim_cfg.startup_summary == "full"
        )
        if details:
            lines.extend(["", *details.splitlines()])
        return lines

    def _extra_initialization_summary_rows(self) -> list[tuple[str, str, str]]:
        """Keep manager status and counts in the main initialization table."""
        manager_summaries: list[tuple[str, list[tuple[str, list[str]]] | None, int]] = (
            []
        )
        active_manager_count = 0
        total_functor_count = 0

        for manager_name, attribute_name in self._manager_summary_fields:
            manager = getattr(self, attribute_name, None)
            if manager is None:
                manager_summaries.append((manager_name, None, 0))
                continue

            groups = self._manager_functor_groups(manager_name, manager)
            functor_count = sum(len(names) for _, names in groups)
            manager_summaries.append((manager_name, groups, functor_count))
            active_manager_count += 1
            total_functor_count += functor_count

        functor_noun = "functor" if total_functor_count == 1 else "functors"
        rows = [
            (
                "Managers",
                "Total",
                f"{active_manager_count}/{len(manager_summaries)} active, "
                f"{total_functor_count} {functor_noun}",
            )
        ]
        for manager_name, groups, functor_count in manager_summaries:
            if groups is None:
                rows.append(("Managers", manager_name, "disabled"))
                continue

            manager_functor_noun = "functor" if functor_count == 1 else "functors"
            rows.append(
                ("Managers", manager_name, f"{functor_count} {manager_functor_noun}")
            )
        return rows

    @staticmethod
    def _manager_functor_groups(
        manager_name: str, manager: object
    ) -> list[tuple[str, list[str]]]:
        """Normalize a manager's active functors into display groups."""
        active_functors = manager.active_functors
        if isinstance(active_functors, Mapping):
            return [
                (str(mode), [str(name) for name in names])
                for mode, names in active_functors.items()
            ]

        names = [str(name) for name in active_functors]
        return [("terms", names)] if names else []

    def _after_episode_reset(self, obs: EnvObs, env_ids: torch.Tensor) -> None:
        if getattr(self, "_final_acceptance_rules", ()):
            bridge = self._active_task_program_bridge
            if env_ids.numel() == self.num_envs:
                self._final_program_completion = None
            elif (bridge is not None and bridge.program_completed) or getattr(
                self, "_final_program_completion", None
            ) is not None:
                completion = self._final_completion_mask().clone()
                completion.index_fill_(0, env_ids.to(completion.device), False)
                self._final_program_completion = (
                    self.episode_context.episode_ids,
                    completion,
                )
        self._active_task_program_bridge = None
        self._final_bridge_episodes = None
        self._seed_recording_state(obs, env_ids)
        if not self.cfg.events or self.event_manager is None or not len(env_ids):
            return
        from embodichain.lab.gym.envs.managers.record import record_camera_data

        for mode_cfgs in self.event_manager._mode_functor_cfgs.values():
            for functor_cfg in mode_cfgs:
                if isinstance(functor_cfg.func, record_camera_data):
                    recorder = functor_cfg.func
                    # A reset event may capture before episode_prepare restores
                    # the final state. Replace only the newly reset live rows.
                    recorder.discard_and_clear(env_ids=env_ids)
                    params = dict(functor_cfg.params)
                    params.setdefault("name", recorder._name)
                    recorder(self, env_ids=env_ids, **params)

    def _seed_recording_state(self, obs: EnvObs, env_ids: torch.Tensor) -> None:
        """Seed pending expert policy observations from the reset observation."""
        self._seed_expert_observations(obs, env_ids)

    def _seed_expert_observations(self, obs: EnvObs, env_ids: torch.Tensor) -> None:
        """Copy current observations into pending expert-transition slots."""
        if (
            self.rollout_buffer is None
            or getattr(self, "_rollout_buffer_mode", "expert") == "rl"
            or env_ids.numel() == 0
        ):
            return
        target = getattr(self, "_expert_obs_buffer", None)
        if target is None:
            target = self.rollout_buffer["obs"]
        eligible = self.rollout_steps[env_ids] < target.shape[1]
        env_ids = env_ids[eligible]
        if env_ids.numel() == 0:
            return
        step_ids = self.rollout_steps[env_ids]
        buffer_device = target.device
        buffer_env_ids = env_ids.to(buffer_device)
        buffer_step_ids = step_ids.to(buffer_device)
        obs_device = getattr(obs, "device", None) or self.device
        obs_env_ids = env_ids.to(obs_device)
        target[buffer_env_ids, buffer_step_ids] = obs[obs_env_ids].to(buffer_device)

    def _record_initial_observation(
        self, frame: ObservationFrame, env_ids: torch.Tensor
    ) -> None:
        manager = getattr(self, "observation_manager", None)
        if manager is not None and manager.has_history:
            if frame.episode_owner is not self.episode_context:
                raise ValueError("Observation frame belongs to another episode owner.")
            manager._bind_schema(frame)
            _write_raw_rollout(
                self._rollout_storage,
                self.episode_context,
                frame,
                field_ids=tuple(dict.fromkeys(manager.field_ids.values())),
                env_ids=env_ids,
            )
            storage = getattr(self, "_rollout_storage", None)
            if (
                storage is not None
                and "policy" in storage.keys()
                and self._rollout_buffer_mode == "expert"
            ):
                policy = storage["policy"]
                if policy.shape[1] == self._rollout_storage["raw_actions"].shape[1] + 1:
                    for key in (
                        "raw_obs",
                        "raw_valid",
                        "raw_revision",
                        "raw_timestamp",
                    ):
                        policy[key] = storage[key]
                    # Existing consumers retain their transition-length view.
                    self.rollout_buffer = policy[:, : self._max_rollout_steps]
        recorder = getattr(self, "_fluid_trajectory_recorder", None)
        if recorder is not None:
            recorder.capture(self, env_ids.tolist(), reset=True)

    def _set_fluid_trajectory_recorder(
        self, recorder: FluidTrajectoryRecorder | None
    ) -> None:
        """Bind one episode's fluid spool, or detach it after publication."""
        if recorder is None:
            self._fluid_trajectory_recorder = None
            return
        if not isinstance(recorder, FluidTrajectoryRecorder):
            raise TypeError("recorder must be a FluidTrajectoryRecorder.")
        if not self.cfg.record_trajectory or not self.cfg.fluid_object:
            raise ValueError(
                "Fluid recording requires trajectory and fluid configuration."
            )
        if self._fluid_trajectory_recorder is not None:
            raise RuntimeError("A fluid trajectory recorder is already attached.")
        self._fluid_trajectory_recorder = recorder

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
        self._rollout_action_semantics = (
            "policy_request"
            if self._rollout_buffer_mode == "rl"
            or getattr(self, "_record_raw_actions", False)
            else "expert_controller_command"
        )
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
        storage = getattr(self, "_rollout_storage", None)
        if storage is not None:
            if self._rollout_buffer_mode == "rl":
                storage["training"] = rollout_buffer
                if "policy" in storage.keys():
                    del storage["policy"]
                self._expert_obs_buffer = None
            else:
                storage["policy"] = rollout_buffer
                self._expert_obs_buffer = rollout_buffer["obs"]
                if "training" in storage.keys():
                    del storage["training"]
        self.rollout_steps.zero_()
        self.current_rollout_step = 0
        if self._rollout_buffer_mode != "rl":
            env_ids = torch.arange(self.num_envs, device=self.device)
            self._clear_expert_rollout_rows(env_ids)
            self._seed_expert_observations(self.get_obs(), env_ids)

    def _init_sim_state(self, **kwargs):
        """Initialize the simulation state at the beginning of scene creation."""

        if not self.sim.has_native_renderer:
            self._filter_native_visual_functors()

        if self._physical_replay is not None:
            self._physical_replay.bind(self)

        # create event manager
        self.cfg: EmbodiedEnvCfg
        if self.cfg.events:
            self.event_manager = EventManager(self.cfg.events, self)

            # perform events at the start of the simulation
            if "startup" in self.event_manager.available_modes:
                self.event_manager.apply(mode="startup")

        policy_program = (
            getattr(getattr(self, "cfg", None), "task_program_execution", "expert")
            == "policy"
        )
        if self.cfg.observations or self.cfg.record_trajectory or policy_program:
            recording = (
                trajectory_observation_cfgs(self, self.cfg.trajectory_uids)
                if self.cfg.record_trajectory
                else {}
            )
            if self.cfg.record_trajectory or policy_program:
                recording.update(getattr(self, "_evaluation_recording", {}))
            self.observation_manager = ObservationManager(
                self.cfg.observations, self, recording=recording
            )

        if self.cfg.rewards:
            self.reward_manager = RewardManager(self.cfg.rewards, self)

        if self.cfg.actions:
            self.action_manager = ActionManager(self.cfg.actions, self)
            # Override action space to match ActionManager output dim.
            self.single_action_space = self.action_manager.single_action_space

        if getattr(self.cfg, "rl_task", None) is not None:
            self.termination_manager = TerminationManager(self.cfg.rl_task, self)
            self._task_initial_states = [{} for _ in range(self.num_envs)]

    def _apply_functor_filter(self) -> None:
        """Apply functor filters to the environment components based on configuration.

        This method is used to filter out certain components of the environment, such as visual randomization,
        based on the configuration settings. For example, if `filter_visual_rand` is set to True in the configuration,
        all visual randomization functors will be removed from the event manager.
        """
        from embodichain.utils.module_utils import get_all_exported_items_from_module
        from embodichain.lab.gym.envs.managers.cfg import (
            EventCfg,
            ObservationCfg,
            SceneEntityCfg,
        )

        sensor_enabled = getattr(self.cfg, "enable_sensor", True)
        sensor_uids = {cfg.uid for cfg in self._sensor_configs() if cfg.uid is not None}

        def uses_disabled_sensor(value: object, field_name: str | None = None) -> bool:
            """Return whether nested functor params reference a disabled sensor."""
            if isinstance(value, SceneEntityCfg):
                return value.uid in sensor_uids
            if isinstance(value, str):
                return value == "all_sensors" or (
                    field_name is not None
                    and ("uid" in field_name.lower() or "sensor" in field_name.lower())
                    and value in sensor_uids
                )
            if isinstance(value, Mapping):
                return any(
                    (isinstance(key, str) and key in sensor_uids)
                    or uses_disabled_sensor(item, str(key))
                    for key, item in value.items()
                )
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return any(uses_disabled_sensor(item, field_name) for item in value)
            return False

        if not sensor_enabled and sensor_uids:
            for collection_name, cfg_type in (
                ("events", EventCfg),
                ("observations", ObservationCfg),
            ):
                collection = getattr(self.cfg, collection_name, None)
                if collection is None:
                    continue
                items = (
                    collection.items()
                    if isinstance(collection, Mapping)
                    else ((name, getattr(collection, name)) for name in dir(collection))
                )
                for name, functor_cfg in items:
                    if not isinstance(functor_cfg, cfg_type):
                        continue
                    name_targets_sensor = isinstance(functor_cfg, ObservationCfg) and (
                        functor_cfg.name == "sensor"
                        or functor_cfg.name.startswith("sensor/")
                    )
                    if name_targets_sensor or uses_disabled_sensor(functor_cfg.params):
                        logger.log_info(
                            f"Filtering out {collection_name[:-1]} functor '{name}' "
                            "because sensor acquisition is disabled."
                        )
                        if isinstance(collection, Mapping):
                            collection[name] = None
                        else:
                            setattr(collection, name, None)

        functors_to_remove = {
            name
            for name in get_all_exported_items_from_module(
                "embodichain.lab.gym.envs.managers.randomization.visual"
            )
            if name.startswith("randomize_")
        }
        if self.cfg.filter_visual_rand and self.cfg.events:
            for collection, name, term in self._iter_configured_functors():
                if collection is not self.cfg.events:
                    continue
                func = self._configured_functor_callable(term)
                if getattr(func, "__name__", None) in functors_to_remove:
                    logger.log_info(
                        f"Filtering out visual randomization functor: {func.__name__}"
                    )
                    if isinstance(collection, Mapping):
                        collection[name] = None
                    else:
                        setattr(collection, name, None)

    def _iter_configured_functors(
        self,
    ) -> Iterable[tuple[object, str, FunctorCfg | dict[str, Any]]]:
        """Iterate terms of managers that will actually be initialized."""
        for manager_name in ("events", "observations", "rewards", "actions", "dataset"):
            if manager_name == "dataset" and getattr(
                self.cfg, "filter_dataset_saving", False
            ):
                continue
            collection = getattr(self.cfg, manager_name, None)
            if collection is None:
                continue
            items = (
                collection.items()
                if isinstance(collection, Mapping)
                else vars(collection).items()
            )
            for name, term in items:
                if isinstance(term, FunctorCfg) or (
                    manager_name == "dataset"
                    and isinstance(term, dict)
                    and "func" in term
                ):
                    yield collection, name, term

    @staticmethod
    def _configured_functor_callable(
        term: FunctorCfg | dict[str, Any],
    ) -> Callable[..., Any]:
        """Resolve a configured function/class without constructing it."""
        func = term["func"] if isinstance(term, dict) else term.func
        return string_to_callable(func) if isinstance(func, str) else func

    def _setup_scene(self, **kwargs) -> None:
        """Filter configuration independently of the overridable demand hook."""
        self._apply_functor_filter()
        recipe = getattr(self.cfg, "usd_scene", None)
        ground = getattr(self.cfg, "default_ground", None)
        if ground is not None and type(ground) is not bool:
            raise TypeError("default_ground must be a bool or None.")
        if ground is not None or recipe is not None:
            self.sim_cfg.default_ground = False if ground is None else ground
        if recipe is not None:
            from embodichain.lab.sim.cfg.physics import physics_backend_from_cfg
            from embodichain.lab.sim.spawn._usd_scene import (
                _assert_usd_scene_current,
                _inspect_usd_scene,
            )

            previous = getattr(self.cfg, "_usd_scene_inventory", None)
            if previous is not None:
                _assert_usd_scene_current(previous)
            reserved = {
                item.uid
                for field in (
                    "background",
                    "rigid_object",
                    "articulation",
                    "visual_object",
                    "rigid_object_group",
                    "cable_object",
                    "fluid_object",
                )
                for item in getattr(self.cfg, field)
                if item.uid is not None
            }
            for group in self.cfg.rigid_object_group:
                reserved.update(
                    item.uid
                    for item in group.rigid_objects.values()
                    if item.uid is not None
                )
            selected_robot = (
                self.cfg.robot.resolve(self.sim_cfg.physics_cfg)
                if isinstance(self.cfg.robot, RobotPresetCfg)
                else self.cfg.robot
            )
            reserved.add(selected_robot.uid or "robot")
            if self.cfg.enable_sensor:
                reserved.update(
                    item.uid for item in selected_robot.sensors if item.uid is not None
                )
            if self.sim_cfg.default_ground:
                reserved.add("default_plane")
            if self.cfg.enable_sensor:
                reserved.update(
                    item.uid for item in self.cfg.sensor if item.uid is not None
                )
            reserved.update(
                item.uid for item in self.cfg.light.direct if item.uid is not None
            )
            self._usd_scene_inventory = _inspect_usd_scene(
                recipe,
                reserved_uids=tuple(reserved),
                physics_backend=physics_backend_from_cfg(self.sim_cfg.physics_cfg),
                device=str(self.sim_cfg.device),
                gravity=self.sim_cfg.physics_cfg.gravity,
            )
        self.cfg.ground_plane_visible = self._resolve_ground_plane_visibility()
        super()._setup_scene(**kwargs)

    def _resolve_ground_plane_visibility(self) -> bool:
        """Resolve the native ground default from filtered event configuration."""
        visible = self.cfg.ground_plane_visible
        if visible is not None:
            if type(visible) is not bool:
                raise TypeError("ground_plane_visible must be a bool or None.")
            return visible

        from embodichain.lab.gym.envs.managers.randomization.visual import (
            randomize_indirect_lighting,
        )

        for collection, _, term in self._iter_configured_functors():
            if collection is not self.cfg.events:
                continue
            func = self._configured_functor_callable(term)
            if (
                isinstance(func, type)
                and issubclass(func, randomize_indirect_lighting)
                and term.params.get("path") is not None
            ):
                return False
        return True

    def _requires_native_renderer(self) -> bool:
        """Predict cameras from the already-filtered manager configuration."""
        from embodichain.lab.gym.envs.managers.record import (
            record_camera_data,
            validation_cameras,
        )
        from embodichain.lab.sim import SimulationManager

        if super()._requires_native_renderer():
            return True
        inventory = getattr(self, "_usd_scene_inventory", None)
        if inventory is not None and {"camera", "light"}.intersection(
            inventory["resource_types"].values()
        ):
            return True
        if self.cfg.enable_sensor:
            for sensor_cfg in self._sensor_configs():
                sensor_factory = SimulationManager.SUPPORTED_SENSOR_TYPES.get(
                    sensor_cfg.sensor_type
                )
                if isinstance(sensor_factory, type) and issubclass(
                    sensor_factory, Camera
                ):
                    return True
        for _, _, term in self._iter_configured_functors():
            func = self._configured_functor_callable(term)
            if not isinstance(func, type):
                continue
            if issubclass(func, record_camera_data):
                return True
            if issubclass(func, validation_cameras):
                params = (
                    term.get("params", {}) if isinstance(term, dict) else term.params
                )
                if params.get("cameras", []):
                    return True
        return False

    def _filter_native_visual_functors(self) -> None:
        """Skip built-in native visual effects when the World uses NoRender."""
        from embodichain.lab.gym.envs.managers.randomization import visual

        visual_functions = {getattr(visual, name) for name in visual.__all__}
        for collection, name, term in self._iter_configured_functors():
            func = self._configured_functor_callable(term)
            if any(func is builtin for builtin in visual_functions) or (
                isinstance(func, type)
                and issubclass(
                    func,
                    (
                        visual.randomize_visual_material,
                        visual.randomize_indirect_lighting,
                    ),
                )
            ):
                logger.log_info(
                    f"Skipping native visual functor '{name}' in no-render mode."
                )
                if isinstance(collection, Mapping):
                    collection[name] = None
                else:
                    setattr(collection, name, None)

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
            self._write_trajectory_step(
                kwargs.get("observation_frame"),
                active=kwargs.get("recording_active_mask"),
            )

        frame = kwargs.get("observation_frame")
        bridge = getattr(self, "_active_task_program_bridge", None)
        if (
            getattr(self, "_evaluation_recording", None)
            and bridge is not None
            and frame is not None
        ):
            process_rules = getattr(self, "_process_evaluation_rules", ())
            if not getattr(self.cfg, "record_trajectory", False):
                segment_fields = bridge._pending_evaluation_fields()
                if segment_fields is not None:
                    fields = dict.fromkeys(segment_fields)
                    fields.update(
                        dict.fromkeys(
                            field
                            for _, rule in process_rules
                            for field in rule.required_fields
                        )
                    )
                    if fields:
                        frame = _capture_evaluation_frame(self, fields)
            if process_rules:
                bridge._observe_segment_frame(
                    frame,
                    episode_owner=self.episode_context,
                    process_check=lambda: _evaluate_process_frame(
                        process_rules, frame, self.device
                    ),
                    hold_qpos=lambda: self.robot.get_qpos(target=True),
                )
            else:
                bridge._observe_segment_frame(frame, episode_owner=self.episode_context)
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
        self.episode_success_status.copy_(
            torch.where(update_mask, success, self.episode_success_status)
        )

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
        self.sim.set_ground_plane_visibility(self.cfg.ground_plane_visible)
        self._setup_background()
        self._setup_interactive_objects()
        self._setup_lights()
        if self.cfg.usd_scene is not None:
            from embodichain.lab.sim.spawn._usd_scene import _assert_usd_scene_current

            _assert_usd_scene_current(self._usd_scene_inventory)
            self.usd_scene_import = self.sim.add_usd(self.cfg.usd_scene)
            for field in (
                "resource_uids",
                "resource_types",
                "resource_source_paths",
                "constraint_uids",
            ):
                if (
                    dict(getattr(self.usd_scene_import, field))
                    != self._usd_scene_inventory[field]
                ):
                    raise ValueError(f"USD scene {field} changed since preflight.")
        if self._physical_replay is not None:
            self._physical_replay.prepare_scene(self)

    def _update_sim_state(self, **kwargs) -> None:
        """Perform the simulation step and apply events if configured.

        The events manager applies its functors after physics simulation and rendering,
        and before the observation and reward computation (if applicable).
        """
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                with self._profiler.section("event_interval"):
                    self.event_manager.apply(mode="interval")

        termination = getattr(self, "termination_manager", None)
        if (
            termination is not None
            and getattr(getattr(self, "cfg", None), "task_program_execution", "expert")
            != "policy"
        ):
            # BaseEnv advances elapsed steps after rewards. Supply this
            # transition's deadline once; Gym flags later consume this cache.
            termination.compute(
                self._elapsed_steps + 1 >= self.max_episode_steps,
                ignore_terminations=self.cfg.ignore_terminations,
            )

    def _after_observation(self, frame: ObservationFrame) -> None:
        """Advance policy progress once before reward and termination read caches."""
        progress = getattr(self, "task_program_progress", None)
        if progress is None:
            return
        progress.advance(
            frame, step=self._program_transition, episode_owner=self.episode_context
        )
        self._program_transition += 1
        self.termination_manager.compute(
            self._elapsed_steps + 1 >= self.max_episode_steps,
            ignore_terminations=self.cfg.ignore_terminations,
        )

    def _initialize_physical_objective(self) -> None:
        """Bind an optional objective to an existing measured rigid object."""
        cfg = getattr(self.cfg, "objective", None)
        self.physical_objective = None
        self._physical_objective_object = None
        if cfg is None:
            return
        objective = OrderedPlacementObjective(
            cfg,
            self.num_envs,
            self.device,
            condition=getattr(self, "_physical_objective_rule", None),
        )
        obj = self.sim.get_rigid_object(objective.cfg.object_uid)
        if obj is None:
            raise ValueError(
                f"Physical objective object_uid {objective.cfg.object_uid!r} "
                "does not name an existing rigid object."
            )
        if obj.body_data is None:
            raise ValueError(
                f"Physical objective object_uid {objective.cfg.object_uid!r} "
                "requires measured rigid body data; static objects are unsupported."
            )
        self._physical_objective_object = obj
        self.physical_objective = objective

    def _update_physical_objective(self) -> None:
        """Observe measured state after interval events, once per control step."""
        objective = getattr(self, "physical_objective", None)
        if objective is None:
            return
        obj = self._physical_objective_object
        objective.update(
            MeasuredRigidObjectState(
                position=obj.get_local_pose()[:, :3],
                linear_velocity=obj.body_data.lin_vel,
                angular_velocity=obj.body_data.ang_vel,
            ),
            self.step_dt,
        )

    def _reset_physical_objective(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        """Clear objective rows after recorders and episode reset events."""
        objective = getattr(self, "physical_objective", None)
        if objective is not None:
            objective.reset(env_ids)
        if getattr(self, "termination_manager", None) is not None:
            self.initialize_task_outcome(env_ids)
        progress = getattr(self, "task_program_progress", None)
        if progress is not None:
            ids = (
                env_ids.tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
            )
            progress.reset(ids, [self.episode_context.episode_ids[row] for row in ids])

    def _rl_target_keys(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    cfg.params["target_pose_key"]
                    for group in ("success", "failure", "truncation")
                    for cfg in getattr(self.cfg.rl_task, group).values()
                    if "target_pose_key" in cfg.params
                }
            )
        )

    def initialize_task_outcome(
        self, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> None:
        """Initialize configured RL rows after reset events or replay restore.

        Args:
            env_ids: Rows with their final initial physical state restored.
        """
        if getattr(self, "termination_manager", None) is None:
            return
        ids = self.episode_context._ids(env_ids)
        targets = {}
        for key in self._rl_target_keys():
            value = getattr(self, key, None)
            if not isinstance(value, torch.Tensor) or value.shape not in (
                (self.num_envs, 3),
                (self.num_envs, 4, 4),
            ):
                raise ValueError(f"Missing or malformed reset target {key!r}.")
            if not value.is_floating_point() or not bool(
                torch.isfinite(value[ids]).all()
            ):
                raise ValueError(
                    f"Reset target {key!r} must contain finite floating poses."
                )
            targets[key] = value
        self.termination_manager.reset(ids)
        for env_id in ids.cpu().tolist():
            self._task_initial_states[env_id] = {
                "schema_version": 1,
                "targets": {
                    key: value[env_id].detach().cpu().tolist()
                    for key, value in targets.items()
                },
            }

    def get_task_initial_state(self, env_id: int = 0) -> dict[str, Any]:
        """Return an owned initial virtual-target context for configured RL.

        Args:
            env_id: Environment row to snapshot.

        Returns:
            Versioned JSON-compatible reset target values.

        Raises:
            ValueError: The environment has no configured RL task.
        """
        if getattr(self, "termination_manager", None) is None:
            raise ValueError("Initial task context requires an rl_task declaration.")
        return copy.deepcopy(self._task_initial_states[env_id])

    def restore_task_initial_state(self, state: Mapping[str, Any]) -> None:
        """Restore a recorded single-environment configured RL target context.

        Args:
            state: Versioned initial state from episode trajectory metadata.

        Raises:
            ValueError: RL context is absent, malformed, or replay is vectorized.
        """
        if getattr(self, "termination_manager", None) is None:
            raise ValueError("Task context restore requires an rl_task declaration.")
        if self.num_envs != 1:
            raise ValueError("Task-outcome replay requires a single environment.")
        if not isinstance(state, Mapping) or state.get("schema_version") != 1:
            raise ValueError(
                "Dynamic task replay requires task_initial_state metadata."
            )
        targets = state.get("targets")
        if not isinstance(targets, Mapping) or set(targets) != set(
            self._rl_target_keys()
        ):
            raise ValueError("Replay virtual targets do not match the configured task.")
        prepared = {}
        for key, value in targets.items():
            current = getattr(self, key, None)
            restored = torch.as_tensor(
                value, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            if (
                current is None
                or restored.shape != current.shape
                or not bool(torch.isfinite(restored).all())
            ):
                raise ValueError(f"Invalid replay target {key!r}.")
            prepared[key] = restored.to(dtype=current.dtype)
        for key, value in prepared.items():
            getattr(self, key).copy_(value)
        self.initialize_task_outcome()

    def _task_success_before_reset(self, env_ids: torch.Tensor) -> torch.Tensor:
        rules = (
            ()
            if getattr(self, "task_program_progress", None) is not None
            else getattr(self, "_final_acceptance_rules", ())
        )
        rows = getattr(self, "_demo_episode_metadata", ())
        if not rows and not rules:
            return super()._task_success_before_reset(env_ids)
        success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        pending = []
        sealed = []
        accepted = []
        for env_id in env_ids.tolist():
            snapshot = rows[env_id].get("episode_outcome")
            if snapshot is None or snapshot["execution"]["source"] == "environment":
                pending.append(env_id)
            else:
                sealed.append(env_id)
                accepted.append(snapshot["acceptance"]["accepted"])
        if pending:
            if rules:
                self._seal_final_acceptance_before_reset(
                    pending, self.episode_context.episode_ids, success
                )
            else:
                success = super()._task_success_before_reset(env_ids).clone()
        if sealed:
            success[sealed] = torch.tensor(
                accepted, dtype=torch.bool, device=self.device
            )
        return success

    def _seal_final_acceptance_before_reset(
        self,
        env_ids: list[int],
        episode_ids: tuple[int, ...],
        success: torch.Tensor,
    ) -> None:
        """Finalize a completed bridge driven outside the demo executor."""
        accepted, completion, measurement = self._evaluate_final_task_acceptance()
        if measurement is None:
            return
        final_rows = _final_acceptance_rows(
            self._final_acceptance_rules,
            episode_ids,
            tuple(accepted.tolist()),
            (True,) * self.num_envs,
            completion,
            measurement,
        )
        for env_id in env_ids:
            length = _demo_recorded_length(self, env_id)
            terminated, truncated = _demo_last_rollout_flags(self, env_id, length)
            snapshot = final_rows[env_id]
            passed = snapshot["accepted"] and not truncated
            snapshot["accepted"] = passed
            snapshot["execution_eligible"] = not truncated
            success[env_id] = passed
            self._sealed_final_observations[env_id] = measurement.frame
            reason = "success" if passed else "task_incomplete"
            if truncated:
                reason = "truncated"
            elif terminated and not passed:
                reason = "failure"
            self._demo_episode_metadata[env_id].update(
                final_acceptance=snapshot,
                length=length,
                completed=passed,
                success=passed,
                terminated=terminated,
                truncated=truncated,
                terminal_reason=reason,
            )
            self._seal_episode_outcome(
                env_id,
                episode_id=episode_ids[env_id],
                source="task_program",
                eligible=not truncated,
            )

    def _seal_episode_outcome(
        self,
        env_id: int,
        *,
        episode_id: int,
        source: str,
        eligible: bool,
        program_accepted: bool | None = None,
        info: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Own the terminal row before any recorder or reset can consume it."""
        metadata = self._demo_episode_metadata[env_id]
        if "episode_outcome" in metadata:
            return metadata["episode_outcome"]
        for key in ("task_outcome", "physical_objective"):
            if info is not None:
                if key in info:
                    metadata[key] = _outcome_row(info[key], env_id)
            elif key not in metadata:
                if key == "task_outcome":
                    manager = getattr(self, "termination_manager", None)
                    if manager is not None:
                        metadata[key] = _outcome_row(manager.snapshot(), env_id)
                else:
                    objective = getattr(self, "physical_objective", None)
                    if objective is not None:
                        metadata[key] = objective.snapshot_row(env_id)
        final = metadata.get("final_acceptance")
        if final is not None:
            program_accepted = final["program_accepted"]
        if (
            "task_outcome" in metadata
            and getattr(self, "termination_manager", None) is not None
        ):
            metadata["task_initial_state"] = self.get_task_initial_state(env_id)
        outcome = {
            "identity": {
                **{
                    key: metadata.get(key)
                    for key in (
                        "run_uuid",
                        "episode_uuid",
                        "env_id",
                        "seed",
                        "program_run_id",
                        "program_hash",
                        "config_hash",
                    )
                },
                "episode": episode_id,
            },
            "execution": {
                "source": source,
                "length": metadata["length"],
                "completed": metadata["completed"],
                "program_accepted": program_accepted,
                "segments": [
                    _segment_outcome(segment, env_id)
                    for segment in metadata.get("segments", ())
                ],
            },
            "physical": {
                "objective": metadata.get("physical_objective"),
                "conditions": metadata.get("task_outcome"),
                "final_acceptance": final,
                **(
                    {
                        "stage_progress": _outcome_row(
                            self.task_program_progress.snapshot(), env_id
                        )
                    }
                    if getattr(self, "task_program_progress", None) is not None
                    else {}
                ),
            },
            "acceptance": {
                "eligible": eligible,
                "accepted": metadata["success"],
                "reason": metadata["terminal_reason"],
            },
            "termination": {
                "terminated": metadata["terminated"],
                "truncated": metadata["truncated"],
                "reason": metadata["terminal_reason"],
            },
            "persistence": None,
        }
        metadata["episode_outcome"] = copy.deepcopy(outcome)
        return metadata["episode_outcome"]

    def _seal_step_outcomes(
        self,
        env_ids: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        info: dict[str, Any],
    ) -> None:
        if (
            "success" not in info
            or not getattr(self, "_demo_episode_metadata", None)
            or (
                getattr(self, "_final_acceptance_rules", ())
                and getattr(self, "task_program_progress", None) is None
            )
        ):
            return
        rows = env_ids.tolist()
        success = _as_bool_tuple(info["success"], self.num_envs)
        terminated = terminated[env_ids].tolist()
        truncated = truncated[env_ids].tolist()
        lengths = self._elapsed_steps[env_ids].tolist()
        reports = info.setdefault("terminal_outcomes", {})
        episode_ids = self.episode_context.episode_ids
        for row, term, trunc, length in zip(
            rows, terminated, truncated, lengths, strict=True
        ):
            metadata = self._demo_episode_metadata[row]
            if "episode_outcome" in metadata:
                reports[row] = copy.deepcopy(metadata["episode_outcome"])
                continue
            accepted = success[row] and not trunc
            metadata.update(
                length=length,
                completed=accepted,
                success=accepted,
                terminated=term,
                truncated=trunc,
                terminal_reason=(
                    "truncated" if trunc else "success" if accepted else "failure"
                ),
            )
            reports[row] = copy.deepcopy(
                self._seal_episode_outcome(
                    row,
                    episode_id=episode_ids[row],
                    source="rl" if "task_outcome" in info else "environment",
                    eligible=accepted,
                    info=info,
                )
            )

    def _before_episode_reset(
        self, env_ids: Sequence[int] | None = None, **kwargs
    ) -> dict[int, dict[str, Any]]:
        if logger.logger.isEnabledFor(logging.DEBUG):
            logger.log_debug(
                f"Initializing episode for env_ids: {env_ids}", color="blue"
            )
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

        commit_env_ids = kwargs.get("commit_env_ids")
        if commit_env_ids is None:
            env_ids_to_commit = (
                env_ids_to_process
                if save_data
                else torch.empty(0, device=status_device, dtype=torch.long)
            )
        elif isinstance(commit_env_ids, torch.Tensor):
            env_ids_to_commit = commit_env_ids.to(
                device=status_device, dtype=torch.long
            ).reshape(-1)
        else:
            env_ids_to_commit = torch.tensor(
                list(commit_env_ids), device=status_device, dtype=torch.long
            )

        if commit_env_ids is not None and env_ids_to_commit.numel() > 0:
            if torch.unique(env_ids_to_commit).numel() != env_ids_to_commit.numel():
                raise ValueError("commit_env_ids must not contain duplicate rows.")
            if not bool(torch.isin(env_ids_to_commit, env_ids_to_process).all().item()):
                raise ValueError(
                    "commit_env_ids must be a subset of the rows being reset."
                )

        # Persist replay state before the dataset recorder snapshots metadata
        # (including asynchronous snapshots). The existing state buffer retains
        # the source episode's initial state even when training exports fragments.
        _traj_buffer = getattr(self, "_traj_buffer", None)
        valid_recording_ids = self.episode_context.filter_recordable(env_ids_to_process)
        terminal_outcomes = {}
        if getattr(self, "_demo_episode_metadata", None):
            episode_ids = self.episode_context.episode_ids
            for env_id in env_ids_to_process.tolist():
                metadata = self._demo_episode_metadata[env_id]
                if (
                    "episode_outcome" not in metadata
                    and getattr(self, "_demo_steps", None) is not None
                    and _demo_recorded_length(self, env_id) > 0
                ):
                    metadata.update(self.get_demo_episode_metadata(env_id))
                    self._seal_episode_outcome(
                        env_id,
                        episode_id=episode_ids[env_id],
                        source="rl" if "task_outcome" in metadata else "legacy_expert",
                        eligible=not metadata["truncated"],
                    )
                if "episode_outcome" in metadata:
                    terminal_outcomes[env_id] = copy.deepcopy(
                        metadata["episode_outcome"]
                    )
        trajectories = {}
        if (
            save_data
            and _traj_buffer is not None
            and getattr(self.cfg, "trajectory_auto_save", False)
        ):
            with self._profiler.section("trajectory_save"):
                for env_id in valid_recording_ids.tolist():
                    trajectories[env_id] = self._save_trajectory_for_env(env_id)

        # Save dataset before clearing buffers for environments that are being reset
        selected = set()
        receipts = {}
        if env_ids_to_commit.numel() > 0 and self.dataset_manager:
            if "save" in self.dataset_manager.available_modes:
                if commit_env_ids is not None:
                    env_ids_to_save = env_ids_to_commit
                elif self.dataset_manager.save_failed_episodes:
                    env_ids_to_save = env_ids_to_process
                else:
                    successful_envs = self._task_success
                    if not getattr(self, "_final_acceptance_rules", ()):
                        successful_envs = successful_envs | self.episode_success_status
                    fragment_envs = torch.tensor(
                        [
                            EmbodiedEnv._demo_fragment_row_is_persistable(
                                self, int(env_id)
                            )
                            for env_id in env_ids_to_process.cpu().tolist()
                        ],
                        dtype=torch.bool,
                        device=status_device,
                    )
                    env_ids_to_save = env_ids_to_process[
                        successful_envs[env_ids_to_process] | fragment_envs
                    ]

                env_ids_to_save = self.episode_context.filter_recordable(
                    env_ids_to_save
                )
                if env_ids_to_save.numel() > 0:
                    with self._profiler.section("dataset_save"):
                        selected = (
                            set(env_ids_to_save.tolist())
                            if terminal_outcomes
                            else set()
                        )
                        receipts = (
                            self.dataset_manager.apply(
                                mode="save",
                                env_ids=env_ids_to_save,
                            )
                            or {}
                        )

        # Save recorded camera data before resetting
        if self.cfg.events and self.event_manager is not None:
            from embodichain.lab.gym.envs.managers.record import record_camera_data

            with self._profiler.section("record_camera_save"):
                for mode_cfgs in self.event_manager._mode_functor_cfgs.values():
                    for functor_cfg in mode_cfgs:
                        if isinstance(functor_cfg.func, record_camera_data):
                            if save_data and valid_recording_ids.numel():
                                functor_cfg.func.save_and_clear(
                                    env_ids=valid_recording_ids
                                )
                            discard_ids = (
                                env_ids_to_process[
                                    ~torch.isin(env_ids_to_process, valid_recording_ids)
                                ]
                                if save_data
                                else env_ids_to_process
                            )
                            if discard_ids.numel():
                                functor_cfg.func.discard_and_clear(env_ids=discard_ids)

        _traj_steps = getattr(self, "_traj_steps", None)
        if _traj_steps is not None:
            _traj_steps.index_fill_(0, env_ids_to_process, 0)

        # Clear episode buffers only after every recorder has consumed them.
        if self.rollout_buffer is not None and self._rollout_buffer_mode != "rl":
            self._clear_expert_rollout_rows(env_ids_to_process)
            rollout_steps = getattr(self, "rollout_steps", None)
            if rollout_steps is not None:
                rollout_ids = env_ids_to_process.to(rollout_steps.device)
                rollout_steps[rollout_ids] = 0
                self.current_rollout_step = int(rollout_steps.max().item())

        active_segment_ids = getattr(self, "_demo_active_segment_ids", None)
        if active_segment_ids is not None:
            demo_ids = env_ids_to_process.to(active_segment_ids.device)
            active_segment_ids.index_fill_(0, demo_ids, 0)
            self._demo_active_mask.index_fill_(0, demo_ids, True)
            self._demo_segment_participants.index_fill_(0, demo_ids, False)
            self._demo_active_segment_start_steps.index_fill_(0, demo_ids, 0)
            self._demo_active_rollout_start_steps.index_fill_(0, demo_ids, 0)
            self._demo_steps.index_fill_(0, demo_ids, 0)

        self.episode_success_status.index_fill_(0, env_ids_to_process, False)
        for env_id, outcome in terminal_outcomes.items():
            outcome["persistence"] = {
                "dataset_selected": env_id in selected,
                "dataset": receipts.get(env_id, []),
                "trajectory": trajectories.get(env_id),
            }
        return terminal_outcomes

    def _initialize_episode(
        self, env_ids: Sequence[int] | torch.Tensor | None = None, **kwargs: Any
    ) -> None:
        env_ids_to_process = self.episode_context._ids(env_ids)
        replay = getattr(self, "_physical_replay", None)
        if replay is not None:
            replay.reset(self, env_ids_to_process)
        # Keep one shared provenance snapshot per seed epoch, even without a
        # recorder. Individual episode dictionaries and UUIDs stay lazy.
        seed = getattr(self.cfg, "seed", None)
        snapshot = getattr(self, "_demo_reset_provenance", None)
        if snapshot is None or snapshot["seed"] != seed:
            self._recording_provenance = build_recording_provenance(self)
            snapshot = {"seed": seed, **self._recording_provenance}
            self._demo_reset_provenance = snapshot
        rows = getattr(self, "_demo_episode_provenance", None)
        if rows is None:
            rows = self._demo_episode_provenance = [snapshot] * self.num_envs
        metadata_ids = env_ids_to_process.cpu().tolist()
        for env_id in metadata_ids:
            rows[env_id] = snapshot
        episode_metadata = getattr(self, "_demo_episode_metadata", None)
        if not getattr(self, "_demo_metadata_initialized", True) and (
            self.dataset_manager is not None
            or getattr(self, "rollout_buffer", None) is not None
            or getattr(self, "_traj_buffer", None) is not None
            or getattr(self, "task_program_progress", None) is not None
        ):
            episode_metadata = self._demo_episode_metadata = _EpisodeMetadataRows(
                self.num_envs, self._new_demo_episode_metadata(0)
            )
            self._demo_metadata_initialized = True
        if episode_metadata is not None and getattr(
            self, "_demo_metadata_initialized", True
        ):
            if isinstance(episode_metadata, _EpisodeMetadataRows):
                if metadata_ids:
                    episode_metadata.reset_rows(
                        metadata_ids, self._new_demo_episode_metadata(metadata_ids[0])
                    )
            else:
                for env_id in metadata_ids:
                    episode_metadata[env_id] = self._new_demo_episode_metadata(env_id)
        final_observations = getattr(self, "_sealed_final_observations", {})
        for env_id in metadata_ids:
            final_observations.pop(env_id, None)
        # Stateful managers reset selected rows before reset-mode events run.
        action_manager = getattr(self, "action_manager", None)
        if action_manager is not None:
            action_manager.reset(env_ids=env_ids)
        if self.cfg.events:
            self.event_manager.reset(env_ids=env_ids)
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

        raw_action_history = getattr(self, "_raw_action_history", None)
        if raw_action_history is not None:
            for env_id in env_ids_to_process.cpu().tolist():
                if 0 <= env_id < len(raw_action_history):
                    raw_action_history[env_id].clear()
        executed_qpos_history = getattr(self, "_expert_controller_qpos_history", None)
        if executed_qpos_history is not None:
            for env_id in env_ids_to_process.cpu().tolist():
                if 0 <= env_id < len(executed_qpos_history):
                    executed_qpos_history[env_id].clear()

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
            "segment_accepted",
            "terminated",
            "truncated",
        ):
            if key in self.rollout_buffer.keys():
                self.rollout_buffer[key][buffer_ids] = False
        for key in (
            "episode_step",
            "segment_id",
            "segment_step",
            "segment_attempt_id",
            "continuity_id",
        ):
            if key in self.rollout_buffer.keys():
                self.rollout_buffer[key][buffer_ids] = -1

    def _demo_fragment_row_is_persistable(self, env_id: int) -> bool:
        """Return whether one fragment-mode row owns an eligible frame span."""
        episode_metadata = getattr(self, "_demo_episode_metadata", None)
        if episode_metadata is None or env_id >= len(episode_metadata):
            return False
        metadata = episode_metadata[env_id]
        if metadata.get("output_mode") != "segment_fragments":
            return False
        include_failed = bool(metadata.get("save_failed_fragments", False))
        for segment in metadata.get("segments", []):
            if int(segment.get("end_step", 0)) <= int(segment.get("start_step", 0)):
                continue
            if bool(segment.get("success", False)) or include_failed:
                return True
        return False

    def _new_demo_episode_metadata(self, env_id: int) -> dict[str, Any]:
        """Create an empty metadata record for one environment row."""
        if getattr(self, "_recording_run_uuid", None) is None:
            self._recording_run_uuid = str(uuid4())
        if getattr(self, "_recording_provenance", None) is None:
            self._recording_provenance = build_recording_provenance(self)
        episode_uuid = str(uuid4())
        execution_cfg = getattr(self, "_demo_execution_cfg", None)
        if execution_cfg is None:
            execution_cfg = DemoExecutionCfg()
        attempt_id = int(getattr(self, "_demo_attempt_id", 0))
        from .demo import _episode_instruction

        instruction, instruction_source = _episode_instruction(self)
        metadata = {
            "instruction": instruction,
            "instruction_source": instruction_source,
            "schema_version": DEMO_SCHEMA_VERSION,
            "run_uuid": self._recording_run_uuid,
            "episode_uuid": episode_uuid,
            "source_episode_uuid": episode_uuid,
            "episode_index": int(getattr(self, "_demo_episode_index", 0)),
            "seed": getattr(getattr(self, "cfg", None), "seed", None),
            "output_mode": execution_cfg.mode,
            "save_failed_fragments": execution_cfg.save_failed_fragments,
            "attempt_id": attempt_id,
            "continuity_id": int(getattr(self, "_demo_continuity_id", 0)),
            "program_run_id": str(
                getattr(
                    self,
                    "_demo_program_run_id",
                    f"{int(getattr(self, '_demo_episode_index', 0))}:{attempt_id}",
                )
            ),
            "env_id": env_id,
            "length": 0,
            "completed": False,
            "success": False,
            "terminated": False,
            "truncated": False,
            "terminal_reason": "unknown",
            "segments": [],
        }
        rows = getattr(self, "_demo_episode_provenance", None)
        provenance = rows[env_id] if rows is not None else self._recording_provenance
        metadata.update(copy.deepcopy(provenance))
        expert_action_spec = getattr(self, "expert_action_spec", None)
        if expert_action_spec is not None:
            metadata.update(expert_action_spec.metadata(step_dt=self.step_dt))
        return metadata

    def _begin_demo_episode_recording(
        self,
        episode_index: int = 0,
        execution_cfg: DemoExecutionCfg | None = None,
        attempt_id: int = 0,
        active_env_ids: Sequence[int] | None = None,
        *,
        replan: bool = False,
    ) -> None:
        """Start annotation metadata for a new demonstration episode."""
        if (
            getattr(getattr(self, "cfg", None), "task_program_execution", "expert")
            == "policy"
        ):
            raise RuntimeError(
                "Policy-owned resources cannot start expert demonstration execution."
            )
        selected = range(self.num_envs) if active_env_ids is None else active_env_ids
        bridge = getattr(self, "_active_task_program_bridge", None)
        if (not replan and getattr(bridge, "program_completed", False)) or (
            getattr(self, "_final_acceptance_rules", ())
            and any(
                "episode_outcome" in self._demo_episode_metadata[row]
                for row in selected
            )
        ):
            raise RuntimeError(
                "Task Program execution requires a reset before retrying a completed attempt."
            )
        completion = (
            self._final_completion_mask().clone()
            if getattr(self, "_final_acceptance_rules", ())
            else None
        )
        if completion is not None:
            completion[list(selected)] = False
        if replan:
            # A fresh planner may bind a bridge after recording starts.
            self._active_task_program_bridge = bridge = None
            self._final_bridge_episodes = None
        if execution_cfg is None:
            execution_cfg = DemoExecutionCfg()
        self._demo_episode_index = episode_index
        self._demo_execution_cfg = execution_cfg
        self._demo_attempt_id = attempt_id
        self._demo_continuity_id = 0
        self._demo_program_run_id = f"{episode_index}:{attempt_id}"
        self._pending_final_acceptance = None
        self._final_program_completion = (
            None
            if completion is None
            else (self.episode_context.episode_ids, completion)
        )
        final_observations = getattr(self, "_sealed_final_observations", {})
        self._sealed_final_observations = final_observations
        self._demo_active_segment_id = 0
        self._demo_active_segment_ids.zero_()
        self._demo_active_mask.fill_(True)
        self._demo_segment_participants.fill_(False)
        self._demo_active_segment_start_steps = self._demo_steps.clone()
        self._demo_active_rollout_start_steps = self.rollout_steps.clone()
        self._demo_metadata_initialized = True
        metadata = self._demo_episode_metadata
        if isinstance(metadata, _EpisodeMetadataRows):
            if selected:
                metadata.reset_rows(
                    selected, self._new_demo_episode_metadata(selected[0])
                )
        else:
            for env_id in selected:
                metadata[env_id] = self._new_demo_episode_metadata(env_id)
        for env_id in selected:
            final_observations.pop(env_id, None)
        program = getattr(bridge, "_program", None)
        if program is not None:
            from embodichain.data_pipeline.recording import stable_config_hash

            program_hash = stable_config_hash(program)
            for env_id in selected:
                self._demo_episode_metadata[env_id]["program_hash"] = program_hash

    def _capture_demo_terminal_physical(
        self, env_ids: list[int], info: Mapping[str, Any]
    ) -> None:
        for key in ("task_outcome", "physical_objective"):
            if key in info:
                for env_id in env_ids:
                    self._demo_episode_metadata[env_id][key] = _outcome_row(
                        info[key], env_id
                    )

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
            if (
                self.rollout_buffer is not None
                and "segment_accepted" in self.rollout_buffer.keys()
                and rollout_end > rollout_start
            ):
                accepted = (
                    result.successes[env_id] if result.successes else result.success
                )
                self.rollout_buffer["segment_accepted"][
                    env_id, rollout_start:rollout_end
                ] = accepted

            metadata = result.to_metadata(env_id if result.start_steps else None)
            metadata["start_step"] = start
            metadata["end_step"] = end
            self._demo_episode_metadata[env_id]["segments"].append(metadata)

    def _end_demo_episode_recording(self, result: DemoEpisodeResult) -> None:
        """Finalize per-environment metadata after demonstration execution."""
        accepted = result.success
        final_rows = None
        rules = getattr(self, "_final_acceptance_rules", ())
        episode_ids = self.episode_context.episode_ids
        if rules:
            pending = getattr(self, "_pending_final_acceptance", None)
            self._pending_final_acceptance = None
            completion, measurement = None, None
            eligible = (False,) * self.num_envs
            if pending is not None and result.terminal_reason != "execution_error":
                if pending[0] != episode_ids:
                    raise RuntimeError("Final acceptance belongs to another episode.")
                _, eligible, completion, measurement = pending
            final_rows = _final_acceptance_rows(
                rules, episode_ids, result.success, eligible, completion, measurement
            )
            accepted = tuple(row["accepted"] for row in final_rows)
        bridge = getattr(self, "_active_task_program_bridge", None)
        program = getattr(bridge, "_program", None)
        program_hash = None
        if program is not None:
            from embodichain.data_pipeline.recording import stable_config_hash

            program_hash = stable_config_hash(program)
        program_accepted = (
            self._task_program_completion_mask().tolist()
            if program is not None and final_rows is None
            else None
        )
        for env_id in range(self.num_envs):
            metadata = self._demo_episode_metadata[env_id]
            if "episode_outcome" in metadata or (
                result.terminal_reasons and result.terminal_reasons[env_id] == "pending"
            ):
                continue
            if final_rows is not None:
                sealed = metadata.get("final_acceptance")
                if (
                    sealed is not None
                    and sealed["episode_id"] == final_rows[env_id]["episode_id"]
                ):
                    continue
                metadata["final_acceptance"] = final_rows[env_id]
                if measurement is not None:
                    self._sealed_final_observations[env_id] = measurement.frame
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
            if final_rows is not None and not accepted[env_id]:
                completed = False
                if terminal_reason == "success":
                    terminal_reason = (
                        "execution_error"
                        if result.terminal_reason == "execution_error"
                        else "task_incomplete"
                    )
            metadata.update(
                {
                    "episode_index": result.episode_index,
                    "length": length,
                    "completed": completed,
                    "success": accepted[env_id],
                    "terminated": result.terminated[env_id],
                    "truncated": result.truncated[env_id],
                    "terminal_reason": terminal_reason,
                }
            )
            if result.instruction is not None:
                metadata["instruction"] = result.instruction
                metadata["instruction_source"] = result.instruction_source
            collection_stats = getattr(self, "_collection_stats", None)
            if isinstance(collection_stats, Mapping):
                metadata["collection"] = dict(collection_stats)
            if bridge is not None:
                metadata["program_hash"] = program_hash
                records = bridge.expansion_records
                metadata["expansion"] = []
                for record in records:
                    if getattr(record, "env_id", None) not in (None, env_id):
                        continue
                    expansion_metadata = dict(record.to_metadata())
                    selected_identity = next(
                        (
                            candidate.identity
                            for candidate in getattr(record, "candidates", ())
                            if candidate.identity.candidate_id
                            == record.selected_candidate_id
                        ),
                        None,
                    )
                    if selected_identity is not None:
                        # Keep the actual source/case identity. Some adapters
                        # put a recipe candidate in template_id, so downstream
                        # grouping must not assume it names a shared reference.
                        expansion_metadata["selected_candidate_lineage"] = {
                            name: getattr(selected_identity, name)
                            for name in (
                                "scene_case_id",
                                "initial_state_id",
                                "candidate_id",
                                "geometry_family_id",
                                "source_id",
                                "source_revision",
                                "template_id",
                                "attempt_id",
                                "parent_id",
                            )
                        }
                    metadata["expansion"].append(expansion_metadata)
                source_uuids = {
                    record["source_episode_uuid"]
                    for record in metadata["expansion"]
                    if isinstance(record.get("source_episode_uuid"), str)
                }
                if len(source_uuids) == 1:
                    metadata["source_episode_uuid"] = next(iter(source_uuids))
                elif source_uuids:
                    metadata["source_episode_uuids"] = sorted(source_uuids)
            self._seal_episode_outcome(
                env_id,
                episode_id=episode_ids[env_id],
                source="task_program" if bridge is not None else "legacy_expert",
                eligible=(
                    final_rows[env_id]["execution_eligible"]
                    if final_rows is not None
                    else (
                        result.execution_eligible[env_id]
                        if result.execution_eligible
                        else completed
                    )
                ),
                program_accepted=(
                    program_accepted[env_id] if program_accepted is not None else None
                ),
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
        if "episode_uuid" not in self._demo_episode_metadata[env_id]:
            metadata = EmbodiedEnv._new_demo_episode_metadata(self, env_id)
            metadata.update(self._demo_episode_metadata[env_id])
            self._demo_episode_metadata[env_id] = metadata
        self._demo_metadata_initialized = True
        metadata = copy.deepcopy(self._demo_episode_metadata[env_id])
        replay = getattr(self, "_physical_replay", None)
        if replay is not None:
            metadata["replay_source"] = replay.snapshot.metadata()
        metadata["episode_context"] = self.episode_context.snapshot([env_id])[0]
        metadata.setdefault("segments", [])
        termination_manager = getattr(self, "termination_manager", None)
        terminal = "episode_outcome" in metadata
        outcome = metadata.get("task_outcome")
        if termination_manager is not None and not terminal:
            outcome = _outcome_row(termination_manager.snapshot(), env_id)
            metadata["task_initial_state"] = self.get_task_initial_state(env_id)
            metadata["task_outcome"] = outcome
            if not metadata["segments"]:
                accepted = bool(outcome["accepted"])
                reason = (
                    "truncated"
                    if outcome["truncation"]
                    else (
                        "failure"
                        if outcome["failure"]
                        else "success" if accepted else "task_incomplete"
                    )
                )
                metadata.update(
                    success=accepted,
                    completed=accepted,
                    terminal_reason=reason,
                    terminated=bool(outcome["terminated"]),
                    truncated=bool(outcome["truncation"]),
                )
        sealed = metadata.get("final_acceptance")
        has_terminal_acceptance = terminal or (
            sealed is not None
            and sealed["episode_id"] == metadata["episode_context"]["episode"]
        )
        length = (
            metadata["length"]
            if has_terminal_acceptance
            else _demo_recorded_length(self, env_id)
        )
        metadata["length"] = length
        if metadata.get("instruction") is None:
            from .demo import resolve_demo_instruction

            metadata["instruction"], metadata["instruction_source"] = (
                resolve_demo_instruction(self)
            )
        if length > 0 and not metadata["segments"]:
            instruction = metadata["instruction"]
            if outcome is not None or has_terminal_acceptance:
                success = metadata["success"]
                terminated = metadata["terminated"]
                truncated = metadata["truncated"]
                terminal_reason = metadata["terminal_reason"]
            else:
                success_status = getattr(self, "episode_success_status", None)
                task_success = getattr(self, "_task_success", None)
                success = bool(
                    not getattr(self, "_final_acceptance_rules", ())
                    and (
                        (success_status is not None and success_status[env_id])
                        or (task_success is not None and task_success[env_id])
                    )
                )
                terminated, truncated = _demo_last_rollout_flags(self, env_id, length)
                if truncated:
                    success = False
                    terminal_reason = "truncated"
                elif success:
                    terminal_reason = "success"
                elif terminated:
                    terminal_reason = "failure"
                else:
                    terminal_reason = "task_incomplete"
            if not has_terminal_acceptance:
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
                    "attempt_id": int(metadata.get("attempt_id", 0)),
                    "continuity_id": int(metadata.get("continuity_id", 0)),
                    "outcome_kind": (
                        "succeeded"
                        if success
                        else ("truncated" if truncated else "validation_failed")
                    ),
                    "target_uid": None,
                    "instruction": instruction,
                    "instruction_source": "task_fallback",
                    "failure_reason": None if success else terminal_reason,
                    "metadata": {},
                }
            ]
        objective = getattr(self, "physical_objective", None)
        if objective is not None and not terminal:
            metadata["physical_objective"] = objective.snapshot_row(env_id)
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

    @property
    def rollout_action_semantics(self) -> str | None:
        """Return the explicit meaning of ``rollout_buffer['actions']``.

        Values are ``policy_request`` for RL/raw-policy buffers and
        ``expert_controller_command`` for expert/demo buffers.
        """
        return self._rollout_action_semantics

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

        expert_action_spec = getattr(self, "expert_action_spec", None)
        policy_action = bool(getattr(self, "_record_raw_actions", False))
        executed_qpos = getattr(self, "_last_action_manager_qpos", None)
        if policy_action:
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    "Policy rollout recording requires one flat torch.Tensor action."
                )
            action_to_store = action
        elif getattr(self, "action_manager", None) is not None and isinstance(
            action, torch.Tensor
        ):
            action_to_store = self._last_action_manager_qpos
            if action_to_store is None:
                raise RuntimeError(
                    "ActionManager dataset recording has no processed qpos snapshot."
                )
        elif (
            expert_action_spec is not None
            and expert_action_spec.joint_command_mode == "position_velocity"
        ):
            action_to_store = encode_expert_action(
                action,
                spec=expert_action_spec,
                active_joint_ids=self.active_joint_ids,
            )
        elif isinstance(action, TensorDict):
            if "qpos" not in action:
                raise ValueError(
                    "Expert joint-position recording requires qpos; received "
                    f"controller keys {list(action.keys())}."
                )
            action_to_store = action["qpos"]
        elif isinstance(action, torch.Tensor):
            action_to_store = action
        else:
            logger.log_warning(
                f"Unexpected action type {type(action)} in _hook_after_sim_step; "
                "skipping action storage in rollout buffer."
            )
            action_to_store = None

        if (
            executed_qpos is None
            and not policy_action
            and not (
                expert_action_spec is not None
                and expert_action_spec.joint_command_mode == "position_velocity"
            )
        ):
            if isinstance(action_to_store, torch.Tensor):
                executed_qpos = action_to_store
        record_executed_qpos = getattr(self, "_record_expert_controller_qpos", None)
        if callable(record_executed_qpos):
            record_executed_qpos(executed_qpos, env_ids)
        if (
            action_to_store is not None
            and not policy_action
            and not (
                expert_action_spec is not None
                and expert_action_spec.joint_command_mode == "position_velocity"
            )
        ):
            robot = getattr(self, "robot", None)
            get_qpos = getattr(robot, "get_qpos", None)
            active_joint_ids = getattr(self, "active_joint_ids", None)
            if callable(get_qpos) and active_joint_ids is not None:
                active_dim = len(active_joint_ids)
                full_dim = int(get_qpos().shape[-1])
                if action_to_store.shape[-1] == full_dim and full_dim != active_dim:
                    action_to_store = action_to_store[..., active_joint_ids]
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
        if "segment_accepted" in buffer_keys:
            # Acceptance is unknown until the segment validator commits. The
            # end-segment hook fills this complete span retroactively.
            self.rollout_buffer["segment_accepted"][
                buffer_env_ids, buffer_step_ids
            ] = False
        if "segment_attempt_id" in buffer_keys:
            self.rollout_buffer["segment_attempt_id"][
                buffer_env_ids, buffer_step_ids
            ] = int(getattr(self, "_demo_attempt_id", 0))
        if "continuity_id" in buffer_keys:
            self.rollout_buffer["continuity_id"][buffer_env_ids, buffer_step_ids] = int(
                getattr(self, "_demo_continuity_id", 0)
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
        self._seed_expert_observations(obs, env_ids)
        self.current_rollout_step = int(self.rollout_steps.max().item())

    def _write_trajectory_step(
        self,
        frame: ObservationFrame | None = None,
        *,
        active: torch.Tensor | None = None,
    ) -> None:
        """Commit the actual action and its captured post-transition state."""
        manager = getattr(self, "observation_manager", None)
        if manager is None or not manager.has_history:
            return
        if frame is None or self._traj_raw_action is None:
            raise RuntimeError(
                "Trajectory recording requires an explicit observation frame and action."
            )
        manager._bind_schema(frame)
        selected = (
            active if active is not None else getattr(self, "_demo_active_mask", None)
        )
        _write_raw_rollout(
            self._rollout_storage,
            self.episode_context,
            frame,
            action=self._traj_raw_action,
            active=selected,
        )
        recorder = getattr(self, "_fluid_trajectory_recorder", None)
        if recorder is not None:
            ids = (
                list(range(self.num_envs))
                if selected is None
                else torch.as_tensor(selected, dtype=torch.bool)
                .nonzero()
                .flatten()
                .tolist()
            )
            recorder.capture(self, ids)
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
        self, action: EnvAction | ControllerAction
    ) -> EnvAction | ControllerAction:
        """Normalize one raw action or preserve a controller-ready envelope."""
        if isinstance(action, ControllerAction):
            return action.snapshot()
        expected_dim = int(np.prod(self.single_action_space.shape))
        return self._normalize_demo_action_tensor(action, expected_dim)

    def _trajectory_action_space(self) -> gym.spaces.Space:
        """Return the fixed action layout owned by trajectory recording."""
        if self.action_manager is not None:
            return self.action_space
        return self._expert_action_space

    def _trajectory_action_metadata(self) -> dict[str, Any]:
        """Describe the action layout stored in trajectory files."""
        if getattr(self, "_trajectory_action_kind", None) == "policy":
            if self.action_manager is None:
                raise RuntimeError(
                    "Policy trajectory metadata requires an ActionManager."
                )
            return {
                "action_kind": "policy",
                "action_dim": self.action_manager.total_action_dim,
                "action_terms": [
                    descriptor.to_dict()
                    for descriptor in self.action_manager.descriptors
                ],
            }
        return {
            "action_kind": "expert",
            **self.expert_action_spec.metadata(step_dt=self.step_dt),
        }

    def _mask_demo_action(
        self,
        action: EnvAction | ControllerAction,
        active_mask: Sequence[bool],
    ) -> EnvAction | ControllerAction:
        """Accept an asynchronously completed vector-demo action.

        Raw actions may still require :class:`ActionManager` preprocessing, so
        the actual hold/no-op substitution is applied to the controller command
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

    def _prepare_controller_action(self, action: EnvAction) -> EnvAction:
        """Validate and normalize one command at the robot-control boundary.

        Both raw actions emitted by ``ActionManager`` and explicit
        :class:`ControllerAction` values pass through this method exactly once.
        Controller tensors must use a supported command key, match the vector
        environment batch, and target either active joints or the full robot.

        Args:
            action: Candidate controller command.

        Returns:
            A controller command whose tensors are on the environment device.

        Raises:
            TypeError: If the action or one of its commands has an invalid type.
            ValueError: If its keys, batch, dimensions, or values are invalid.
        """
        supported_keys = ("qpos", "qvel", "qf")
        active_dim = len(self.active_joint_ids)
        full_dim = int(self.robot.get_qpos().shape[-1])

        def prepare_command(command: torch.Tensor, key: str) -> torch.Tensor:
            if not isinstance(command, torch.Tensor):
                raise TypeError(f"Controller {key} command must be a torch.Tensor.")
            if not command.is_floating_point():
                raise TypeError(f"Controller {key} command must be floating point.")
            if command.dim() != 2 or command.shape[0] != self.num_envs:
                raise ValueError(
                    f"Controller {key} command must have shape (num_envs, D); "
                    f"received {tuple(command.shape)} for {self.num_envs} environments."
                )
            if command.shape[1] not in {active_dim, full_dim}:
                raise ValueError(
                    f"Controller {key} command has dim {command.shape[1]}; expected "
                    f"active-joint dim {active_dim} or full robot dim {full_dim}."
                )
            return command.to(self.device)

        if isinstance(action, torch.Tensor):
            return prepare_command(action, "qpos")
        if not isinstance(action, TensorDict):
            raise TypeError(
                "Controller action must be a torch.Tensor or TensorDict, "
                f"got {type(action).__name__}."
            )
        if tuple(action.batch_size) != (self.num_envs,):
            raise ValueError(
                "Controller TensorDict batch size must match num_envs; "
                f"received {tuple(action.batch_size)} for {self.num_envs} environments."
            )
        keys = tuple(action.keys())
        control_keys = tuple(key for key in supported_keys if key in action)
        if not control_keys:
            raise ValueError(
                "Controller TensorDict must contain at least one of "
                f"{list(supported_keys)}; received keys {list(keys)}."
            )

        prepared = action.clone().to(self.device)
        for key in control_keys:
            prepared[key] = prepare_command(prepared[key], key)
        return prepared

    def _mask_controller_demo_action(self, action: EnvAction) -> EnvAction:
        """Replace inactive controller commands with a safe hold or no-op."""
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
            """Select the measured qpos layout matching the controller command."""
            if value.shape[1:] == active_qpos.shape[1:]:
                return active_qpos
            if value.shape[1:] == measured_qpos.shape[1:]:
                return measured_qpos
            raise ValueError(
                "Cannot construct a qpos hold command for controller demo action "
                f"shape {tuple(value.shape)}; measured active/full layouts are "
                f"{tuple(active_qpos.shape)} and {tuple(measured_qpos.shape)}."
            )

        if isinstance(action, torch.Tensor):
            return replace_rows(action, qpos_replacement(action), "qpos")
        if not isinstance(action, TensorDict):
            raise TypeError(
                "Controller demo actions must be torch.Tensor or TensorDict, "
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
                "Cannot mask a controller demo TensorDict without qpos, qvel, or qf."
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

    def _step_action(self, action: EnvAction | ControllerAction) -> EnvAction:
        """Apply a policy action or a direct controller-ready command.

        Flat policy tensors are applied by :class:`ActionManager`. Explicit
        :class:`ControllerAction` values bypass the manager and are sent to the
        robot through the qpos/qvel/qf controller boundary.

        Args:
            action: Prepared policy or controller action.

        Returns:
            The raw policy tensor or applied controller value.
        """

        replay = getattr(self, "_physical_replay", None)
        if replay is not None:
            return replay.consume(self, action)
        if isinstance(action, ControllerAction):
            if getattr(self, "_step_uses_action_manager", False):
                self.action_manager.apply_action()
                self._last_action_trace = self.action_manager.action_trace()
                if getattr(self, "_record_raw_actions", False):
                    return self.action_manager.action
                return action.value
            return EmbodiedEnv._apply_controller_action(self, action.value)
        action_manager = getattr(self, "action_manager", None)
        if action_manager is not None:
            action_manager.apply_action()
            self._last_action_trace = action_manager.action_trace()
            return action
        return EmbodiedEnv._apply_controller_action(self, action)

    def _replay_snapshot(self, source_id: str):
        """Return the host-owned command source after native robot binding."""
        replay = self._physical_replay
        if replay is None or replay.snapshot.source.stream_id != source_id:
            raise ValueError(f"No physical replay source {source_id!r} is bound.")
        return replay.snapshot

    def _advance_physics(self) -> None:
        """Connect manager callbacks to physics and sensor sampling."""
        replay = getattr(self, "_physical_replay", None)
        if replay is not None and replay.hold:
            return
        manager = getattr(self, "action_manager", None)
        if not getattr(self, "_step_uses_action_manager", False) or manager is None:
            return super()._advance_physics()
        refresh_actions = manager.get_substep_callback(self.cfg.sim_steps_per_control)
        if refresh_actions is None:
            return super()._advance_physics()

        sensors = [
            sensor for sensor in self.sensors.values() if sensor.requires_substep_update
        ]
        for sensor in sensors:
            sensor.begin_control_step()

        def observe_substep(dt: float) -> None:
            for sensor in sensors:
                sensor.update_physics_step(dt)
            refresh_actions(dt)

        self.sim.update(
            self.physics_dt,
            self.cfg.sim_steps_per_control,
            after_substep=observe_substep,
            render_final_step=False,
        )

    @property
    def last_action_trace(self) -> ActionTrace | None:
        """Return the most recent ActionManager request/command trace."""
        trace = getattr(self, "_last_action_trace", None)
        return None if trace is None else trace.clone()

    def _apply_controller_action(self, action: EnvAction) -> EnvAction:
        """Apply one validated controller command directly to active joints."""

        def active_joint_command(command: torch.Tensor, key: str) -> torch.Tensor:
            """Normalize full-robot commands to the active-joint layout."""
            active_dim = len(self.active_joint_ids)
            if command.shape[-1] == active_dim:
                return command.to(self.device)
            full_dim = int(self.robot.get_qpos().shape[-1])
            if command.shape[-1] == full_dim:
                return command[..., self.active_joint_ids].to(self.device)
            raise ValueError(
                f"Controller {key} action has dim {command.shape[-1]}; expected "
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

        Configured ``rl_task`` conditions return cached acceptance. Subclasses
        can override this method for handwritten task-specific logic.

        Returns:
            tuple[torch.Tensor, torch.Tensor, dict[str, Any]]: Success and failure
            masks of shape ``(num_envs,)``, followed by the metric tensor mapping.
        """
        outcome = getattr(self, "termination_manager", None)
        if outcome is not None:
            return outcome.task_state()
        success = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        fail = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        metrics: Dict[str, Any] = {}
        return success, fail, metrics

    def get_info(self, **kwargs) -> Dict[str, Any]:
        """Get environment info dictionary.

        Calls compute_task_state() for success/failure/metrics and includes an
        owned outcome snapshot when ``rl_task`` is configured.

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
        objective = getattr(self, "physical_objective", None)
        if objective is not None:
            info["physical_objective"] = objective.snapshot()
        outcome = getattr(self, "termination_manager", None)
        if outcome is not None:
            info["task_outcome"] = outcome.snapshot()
        progress = getattr(self, "task_program_progress", None)
        if progress is not None:
            info["task_program_progress"] = progress.snapshot()
        if getattr(self.cfg, "episode_context_in_info", True):
            info["episode_context"] = self.episode_context.snapshot()
        return info

    def check_truncated(self, obs: EnvObs, info: dict[str, Any]) -> torch.Tensor:
        """Return cached RL truncation, or the ordinary host's custom mask.

        Args:
            obs: Current observations.
            info: Cached task information.

        Returns:
            Per-environment truncation mask.
        """
        outcome = getattr(self, "termination_manager", None)
        if outcome is not None:
            return outcome.truncated
        return super().check_truncated(obs, info)

    def _get_termination_flags(
        self, obs: EnvObs, info: Dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Use the same cached flags consumed by observations and rewards."""
        termination = getattr(self, "termination_manager", None)
        if termination is not None:
            return termination.terminated, termination.truncated
        return super()._get_termination_flags(obs, info)

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

    def _preprocess_action(
        self, action: EnvAction | ControllerAction
    ) -> EnvAction | ControllerAction:
        """Resolve one raw or controller-ready action for robot control."""
        self._last_action_manager_qpos = None
        self._last_action_trace = None
        is_controller_action = isinstance(action, ControllerAction)
        if (
            is_controller_action
            and getattr(getattr(self, "cfg", None), "task_program_execution", "expert")
            == "policy"
        ):
            raise ValueError(
                "Policy-owned resources require ActionManager policy actions."
            )
        expert_cfg = getattr(getattr(self, "cfg", None), "expert_trajectory", None)
        managed_expert = is_controller_action and bool(
            getattr(expert_cfg, "use_action_manager", False)
        )
        if managed_expert and self.action_manager is None:
            raise ValueError(
                "ActionManager expert execution requires configured action terms."
            )
        self._step_uses_action_manager = self.action_manager is not None and (
            not is_controller_action or managed_expert
        )
        controller_metadata = action.metadata if is_controller_action else None
        policy_trajectory = self._traj_buffer is not None and (
            getattr(self, "_trajectory_action_kind", None) == "policy"
        )
        policy_recording = bool(getattr(self, "_record_raw_actions", False))
        if (
            is_controller_action
            and not managed_expert
            and (policy_recording or policy_trajectory)
        ):
            incompatible = []
            if policy_recording:
                incompatible.append("policy action recording")
            if policy_trajectory:
                incompatible.append("policy trajectory")
            raise ValueError(
                "ControllerAction cannot be used with "
                + " or ".join(incompatible)
                + "."
            )
        if is_controller_action:
            action = action.value
        record_controller_targets = (
            self._traj_buffer is not None
            and not policy_trajectory
            and not (managed_expert and policy_recording)
            and getattr(self, "expert_action_spec", None) is not None
            and (
                is_controller_action
                or self.expert_action_spec.joint_command_mode == "position_velocity"
            )
        )
        retain_raw_action = getattr(self, "_record_raw_actions", False) or (
            self._traj_buffer is not None and not record_controller_targets
        )
        raw_action = (
            (action.clone() if hasattr(action, "clone") else copy.deepcopy(action))
            if retain_raw_action
            else action
        )
        if self.action_manager is not None and (
            not is_controller_action or managed_expert
        ):
            if managed_expert:
                action = self._prepare_controller_action(action)
                if getattr(self, "_demo_no_auto_reset", False):
                    action = self._mask_controller_demo_action(action)
                qpos = action if isinstance(action, torch.Tensor) else action["qpos"]
                full_targets = self.robot.get_qpos().clone()
                full_targets[:, self.active_joint_ids] = self._active_qpos_command(qpos)
                self.action_manager.process_joint_targets(full_targets)
            if policy_recording:
                policy_action = self.action_manager.action if managed_expert else action
                explicit_policy = getattr(self, "_explicit_policy_rollout", False)
                if explicit_policy:
                    self._validate_policy_rollout_action(policy_action)
                if self.dataset_manager is not None:
                    self.dataset_manager.validate_policy_action(policy_action)
                elif not explicit_policy:
                    raise RuntimeError(
                        "Policy action recording requires an initialized DatasetManager."
                    )
            if not managed_expert:
                self.action_manager.process_action(action)
            if getattr(self, "_demo_no_auto_reset", False):
                self.action_manager.mask_inactive(self._demo_active_mask)
            if self.dataset_manager is not None and not policy_recording:
                processed_qpos = getattr(self.action_manager, "_processed_qpos", None)
                self._last_action_manager_qpos = (
                    processed_qpos().detach().clone()
                    if callable(processed_qpos)
                    else None
                )
            elif self.dataset_manager is not None or (
                getattr(self, "rollout_buffer", None) is not None
                and getattr(self, "_rollout_buffer_mode", None) != "rl"
            ):
                processed_qpos = getattr(self.action_manager, "_processed_qpos", None)
                try:
                    self._last_action_manager_qpos = (
                        processed_qpos().detach().clone()
                        if callable(processed_qpos)
                        else None
                    )
                except ValueError:
                    # A policy action may contain velocity/effort terms, for
                    # which there is no executed joint-position snapshot.
                    self._last_action_manager_qpos = None
            raw_action = self.action_manager.action
        elif not is_controller_action:
            action = super()._preprocess_action(action)
            action = self._prepare_controller_action(action)
            if getattr(self, "_demo_no_auto_reset", False):
                action = self._mask_controller_demo_action(action)
        elif is_controller_action:
            action = self._prepare_controller_action(action)
            if getattr(self, "_demo_no_auto_reset", False):
                action = self._mask_controller_demo_action(action)
            if isinstance(action, torch.Tensor):
                self._last_action_manager_qpos = self._active_qpos_command(action)
            elif "qpos" in action:
                self._last_action_manager_qpos = self._active_qpos_command(
                    action["qpos"]
                )
        if policy_trajectory:
            self._traj_raw_action = raw_action.clone()
        elif record_controller_targets:
            self._traj_raw_action = encode_expert_action(
                action,
                spec=self.expert_action_spec,
                active_joint_ids=self.active_joint_ids,
            )
        elif self._traj_buffer is not None:
            self._traj_raw_action = (
                raw_action.clone()
                if hasattr(raw_action, "clone")
                else copy.deepcopy(raw_action)
            )
        # Do not retain malformed actions: validation, masking and trajectory
        # encoding above are the ownership boundary for raw-action history.
        if not is_controller_action or managed_expert:
            self._record_raw_action(raw_action)
        if is_controller_action:
            return ControllerAction(value=action, metadata=controller_metadata)
        return action

    def _validate_policy_rollout_action(self, action: EnvAction) -> None:
        """Validate explicit raw-policy recording before changing term state."""
        if not isinstance(action, torch.Tensor) or not action.is_floating_point():
            raise TypeError("Policy rollout actions must be floating tensors.")
        expected = (self.num_envs, self.action_manager.total_action_dim)
        if tuple(action.shape) != expected:
            raise ValueError(f"Policy rollout action must have shape {expected}.")
        if not bool(torch.isfinite(action).all()):
            raise ValueError("Policy rollout actions must contain only finite values.")
        space = self.action_manager.single_action_space
        low = torch.as_tensor(space.low, dtype=action.dtype, device=action.device)
        high = torch.as_tensor(space.high, dtype=action.dtype, device=action.device)
        if bool(((action < low) | (action > high)).any()):
            raise ValueError(
                "Policy rollout actions must remain within action-space bounds."
            )

    def _record_raw_action(self, action: EnvAction) -> None:
        """Keep a CPU copy of raw policy actions for explicit dataset contracts."""
        if not getattr(self, "_record_raw_actions", False):
            return
        num_envs = int(self.num_envs)
        raw_action_history = getattr(self, "_raw_action_history", None)
        if raw_action_history is None:
            raw_action_history = [[] for _ in range(num_envs)]
            self._raw_action_history = raw_action_history

        if isinstance(action, TensorDict):
            if tuple(action.batch_size) != (num_envs,):
                return
            batch_action = action.detach().cpu().clone()
            rows = [batch_action[index].clone() for index in range(num_envs)]
        elif isinstance(action, Mapping):
            tensors = {
                key: torch.as_tensor(value).detach().cpu().clone()
                for key, value in action.items()
            }
            if any(
                value.ndim == 0 or value.shape[0] != num_envs
                for value in tensors.values()
            ):
                return
            tensor_action = TensorDict(tensors, batch_size=[num_envs])
            rows = [tensor_action[index].clone() for index in range(num_envs)]
        elif isinstance(action, torch.Tensor):
            if action.ndim == 0 or action.shape[0] != num_envs:
                return
            batch_action = action.detach().cpu().clone()
            rows = list(batch_action.unbind(0))
        else:
            rows = [copy.deepcopy(action) for _ in range(self.num_envs)]
        active_mask = getattr(self, "_demo_active_mask", None)
        if active_mask is not None and getattr(self, "_demo_no_auto_reset", False):
            active_mask = torch.as_tensor(active_mask, device="cpu", dtype=torch.bool)
        else:
            active_mask = None
        for index, row in enumerate(rows):
            if active_mask is None or bool(active_mask[index]):
                raw_action_history[index].append(row)

    def _active_qpos_command(self, command: torch.Tensor) -> torch.Tensor:
        """Return a detached active-joint qpos snapshot for recorder history."""
        active_dim = len(self.active_joint_ids)
        full_dim = int(self.robot.get_qpos().shape[-1])
        if command.shape[-1] == active_dim:
            return command.detach().clone()
        if command.shape[-1] == full_dim:
            return command[..., self.active_joint_ids].detach().clone()
        raise ValueError(
            "Cannot snapshot qpos command with dimension "
            f"{command.shape[-1]}; expected {active_dim} or {full_dim}."
        )

    def _record_expert_controller_qpos(
        self, action: torch.Tensor | None, env_ids: torch.Tensor
    ) -> None:
        """Keep expert controller qpos commands for compatible recorders."""
        if action is None:
            return
        snapshots = self._active_qpos_command(action)
        history = getattr(self, "_expert_controller_qpos_history", None)
        if history is None:
            history = [[] for _ in range(self.num_envs)]
            self._expert_controller_qpos_history = history
        for env_id in env_ids.detach().cpu().tolist():
            history[env_id].append(snapshots[env_id].cpu().clone())

    def get_expert_controller_qpos_history(
        self, env_id: int, length: int | None = None
    ) -> torch.Tensor:
        """Return expert controller qpos commands for one episode."""
        history = getattr(self, "_expert_controller_qpos_history", None)
        if history is None:
            return torch.empty((0, len(self.active_joint_ids)), dtype=torch.float32)
        rows = history[int(env_id)]
        if length is not None:
            rows = rows[:length]
        if not rows:
            return torch.empty((0, len(self.active_joint_ids)), dtype=torch.float32)
        return torch.stack(rows, dim=0)

    def _record_executed_qpos(
        self, action: torch.Tensor | None, env_ids: torch.Tensor
    ) -> None:
        """Compatibility alias for :meth:`_record_expert_controller_qpos`."""
        self._record_expert_controller_qpos(action, env_ids)

    def get_executed_qpos_history(
        self, env_id: int, length: int | None = None
    ) -> torch.Tensor:
        """Compatibility alias for expert controller qpos history."""
        return self.get_expert_controller_qpos_history(env_id, length)

    def get_raw_action_history(self, env_id: int, length: int | None = None) -> Any:
        """Return raw actions recorded for one environment episode."""
        raw_action_history = getattr(self, "_raw_action_history", None)
        if raw_action_history is None:
            return torch.empty((0, 0), dtype=torch.float32)
        rows = raw_action_history[int(env_id)]
        if length is not None:
            rows = rows[:length]
        if not rows:
            return torch.empty((0, 0), dtype=torch.float32)
        if all(isinstance(row, torch.Tensor) for row in rows):
            return torch.stack(rows, dim=0)
        if all(isinstance(row, TensorDict) for row in rows):
            return TensorDict.stack(rows, dim=0)
        return rows

    def _declare_robot(self, **kwargs) -> Robot:
        """Declare the configured robot without reading articulation metadata."""
        del kwargs
        if self.cfg.robot is None:
            logger.log_error("Robot configuration is not provided.")
        robot_cfg = copy.deepcopy(self.cfg.robot)
        if isinstance(robot_cfg, RobotPresetCfg):
            for item in fields(robot_cfg):
                candidate = getattr(robot_cfg, item.name)
                if isinstance(candidate, RobotCfg):
                    candidate.enable_sensor = self.cfg.enable_sensor
            selected_cfg = robot_cfg.resolve(
                self.sim.sim_config.physics_cfg,
                newton_solver_type=self.sim._active_newton_solver_type,
            )
        else:
            robot_cfg.enable_sensor = self.cfg.enable_sensor
            selected_cfg = robot_cfg
        _collect_sensor_cfgs(selected_cfg, self.cfg.sensor)
        self._robot_sensor_cfgs = selected_cfg.sensors
        return self.sim.add_robot(robot_cfg)

    def _setup_robot(self, **kwargs) -> Robot:
        """Configure the finalized robot interface for the environment.

        Currently, only joint position control is supported. Would be extended to support joint velocity and torque
            control in the future.

        Returns:
            Robot: The robot instance added to the scene.
        """
        del kwargs
        robot = self.robot
        if robot is None:
            logger.log_error("Robot was not declared before simulation prepare.")

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

            self.active_joint_ids = [
                joint_id
                for part_name in self.cfg.control_parts
                for joint_id in robot.get_joint_ids(name=part_name, remove_mimic=True)
            ]
        elif self.cfg.active_joint_ids:
            # Check env active joint ids are valid
            for joint_id in self.cfg.active_joint_ids:
                if joint_id not in robot.active_joint_ids:
                    logger.log_error(
                        f"Invalid active joint id: {joint_id}. The supported active joint ids are: {robot.active_joint_ids}"
                    )
            self.active_joint_ids = list(self.cfg.active_joint_ids)
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

    def _sensor_configs(self) -> list[SensorCfg]:
        """Return independent and selected robot sensor declarations."""
        robot_cfg = getattr(self.cfg, "robot", None)
        if hasattr(self, "_robot_sensor_cfgs"):
            return _collect_sensor_cfgs(
                None, [*getattr(self.cfg, "sensor", []), *self._robot_sensor_cfgs]
            )
        if isinstance(robot_cfg, RobotPresetCfg):
            # Filtering and render-demand prediction run before World creation.
            robot_cfg = robot_cfg.resolve(self.sim_cfg.physics_cfg)
        return _collect_sensor_cfgs(robot_cfg, getattr(self.cfg, "sensor", []))

    def _setup_sensors(self, **kwargs) -> Dict[str, BaseSensor]:
        """Setup the sensors in the environment.

        Returns:
            Dict[str, BaseSensor]: A dictionary mapping sensor UIDs to sensor instances.
        """

        if not self.cfg.enable_sensor:
            logger.log_info(
                "Sensor acquisition is disabled; configured sensors will not be instantiated."
            )
            return {}

        sensors = {}
        for cfg in self.cfg.sensor:
            sensor = self.sim.add_sensor(cfg)
            sensors[cfg.uid] = sensor
        for cfg in getattr(self, "_robot_sensor_cfgs", []):
            sensor = self.sim.get_sensor(cfg.uid)
            if sensor is None:
                raise RuntimeError(f"Mounted sensor {cfg.uid!r} was not initialized.")
            sensors[cfg.uid] = sensor
        return sensors

    def _setup_lights(self) -> None:
        """Setup the lights in the environment."""
        if not self.sim.has_native_renderer:
            return
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

        def validate_resolved(cfg: RigidObjectCfg) -> None:
            if cfg.body_type == "dynamic":
                logger.log_error(
                    "Background object must be kinematic or static rigid object."
                )

        for cfg in self.cfg.visual_object:
            self.sim.add_visual_object(cfg)
        for cfg in self.cfg.background:
            self.sim.add_rigid_object(cfg=cfg, validate_resolved=validate_resolved)

    def _setup_interactive_objects(self) -> None:
        """Setup the interactive objects in the environment."""

        def validate_resolved(cfg: RigidObjectCfg) -> None:
            if cfg.body_type != "dynamic":
                logger.log_error(
                    "Interactive rigid object must be dynamic rigid object."
                )

        for cfg in self.cfg.articulation:
            self.sim.add_articulation(cfg=cfg)

        for cfg in self.cfg.rigid_object:
            self.sim.add_rigid_object(cfg=cfg, validate_resolved=validate_resolved)

        for cfg in self.cfg.rigid_object_group:
            self.sim.add_rigid_object_group(cfg=cfg)

        for cfg in self.cfg.cable_object:
            self.sim.add_cable_object(cfg=cfg)
        for cfg in self.cfg.fluid_object:
            self.sim.add_fluid_object(cfg)

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

    @property
    def task_program_adapter(self) -> TaskProgramEnvironmentAdapter:
        """Return the adapter injected after the environment built its scene.

        A registered environment normally receives an
        :class:`TaskProgramAdapterFactory` through its :class:`EnvSpec`.
        Advanced integrations may still override this property.
        """
        adapter = self._task_program_adapter
        if adapter is None:
            raise NotImplementedError(
                "An environment with an Task Program must receive an "
                "task_program_adapter_factory or override "
                "task_program_adapter."
            )
        return adapter

    def _checked_task_program_adapter(self) -> TaskProgramEnvironmentAdapter:
        """Return the exact configured adapter before touching live providers."""
        from embodichain.lab.task_program.integrations import (
            TaskProgramEnvironmentAdapter,
        )

        adapter = self.task_program_adapter
        if type(adapter) is not TaskProgramEnvironmentAdapter:
            raise TypeError(
                "task_program_adapter must be exactly " "TaskProgramEnvironmentAdapter."
            )
        return adapter

    def compile_task_program(
        self,
        program: TaskProgramCfg,
    ) -> CompiledTaskProgram:
        """Compile a configured Task Program through the explicit adapter.

        Args:
            program: Strict Task Program configuration attached to ``cfg``.

        Returns:
            Provider-free compiled program ready for runtime assembly.

        """
        return self._checked_task_program_adapter().compile(program)

    @contextmanager
    def task_program_bridge_scope(
        self, bridge: TaskProgramDemoBridge
    ) -> Iterator[None]:
        """Publish a borrowed bridge for one execution and restore the prior binding.

        Args:
            bridge: Bridge whose commands are consumed by this environment owner.

        Yields:
            No value. The caller retains reset, stepping and cleanup ownership.
        """
        from embodichain.lab.gym.envs.task_program.bridge import TaskProgramDemoBridge

        if not isinstance(bridge, TaskProgramDemoBridge):
            raise TypeError("bridge must be a TaskProgramDemoBridge")
        previous = self._active_task_program_bridge
        previous_episodes = getattr(self, "_final_bridge_episodes", None)
        self._active_task_program_bridge = bridge
        if getattr(self, "_final_acceptance_rules", ()):
            self._final_bridge_episodes = self.episode_context.episode_ids
        try:
            yield
        finally:
            self._active_task_program_bridge = previous
            self._final_bridge_episodes = previous_episodes

    def create_task_program_bridge(
        self,
        program: CompiledTaskProgram,
        *,
        expansion_profile: CombinedExpansionProfile | None = None,
        expansion_candidate_index: int = 0,
        expansion_recipe_indices: tuple[int, ...] | None = None,
        nominal_source_factory: TaskProgramNominalSourceFactory | None = None,
        plan_transform_factory: TaskProgramPlanTransformFactory | None = None,
    ) -> TaskProgramDemoBridge:
        """Create the Gym demo bridge through the explicit adapter.

        Args:
            program: Compiled provider-free Task Program.
            expansion_profile: Optional episode-scoped Expansion Profile.
            expansion_candidate_index: FIFO candidate ordinal selected from
                each matching Atomic plan.

        Returns:
            Atomic demo bridge whose segments are consumed lazily.

        """
        if (
            getattr(getattr(self, "cfg", None), "task_program_execution", "expert")
            == "policy"
        ):
            raise RuntimeError(
                "Policy-only Task Program declarations cannot activate an expert bridge."
            )
        kwargs = {}
        if nominal_source_factory is not None:
            kwargs["nominal_source_factory"] = nominal_source_factory
        if plan_transform_factory is not None:
            kwargs["plan_transform_factory"] = plan_transform_factory
        bridge = self._checked_task_program_adapter().create_bridge(
            program,
            expansion_profile=expansion_profile,
            candidate_index=expansion_candidate_index,
            recipe_indices=expansion_recipe_indices,
            **kwargs,
        )
        if any(
            segment.contract is not None
            and any(entry.spec is not None for entry in segment.contract.prerequisites)
            for segment in program.iter_segments()
        ):
            from embodichain.lab.task_program.integrations.simulation.bindings import (
                _bind_evaluation_joint_names,
            )

            bridge._contract_conditions = _bind_evaluation_joint_names(
                {
                    entry.ref: entry.spec
                    for segment in program.iter_segments()
                    if segment.contract is not None
                    for entry in segment.contract.prerequisites
                    if entry.spec is not None
                },
                self.robot,
            )
            bridge._contract_capture = lambda fields: _capture_evaluation_frame(
                self, fields
            )
            bridge._contract_episode_owner = self.episode_context
        return bridge

    @property
    def task_program_expansion_records(
        self,
    ) -> tuple[TaskProgramExpansionRecord, ...]:
        """Return records owned by the active generated Task Program episode."""
        bridge = getattr(self, "_active_task_program_bridge", None)
        return () if bridge is None else bridge.expansion_records

    def is_task_success(self, **kwargs: Any) -> torch.Tensor:
        """Return configured RL, completed Task Program, or legacy acceptance.

        Task Program success requires a completed bridge and, when selected,
        a fresh measurement of the deployment's final predicates.

        Args:
            **kwargs: Compatibility keywords forwarded for tasks without a Task Program.

        Returns:
            Per-environment task-success mask.
        """
        outcome = getattr(self, "termination_manager", None)
        if outcome is not None:
            return outcome.task_state()[0]
        if getattr(self, "_final_acceptance_rules", ()):
            return self._evaluate_final_task_acceptance()[0]
        if (
            self._active_task_program_bridge is None
            and getattr(getattr(self, "cfg", None), "task_program", None) is None
        ):
            return super().is_task_success(**kwargs)
        return self._task_program_completion_mask()

    def _task_program_completion_mask(self) -> torch.Tensor:
        bridge = self._active_task_program_bridge
        if bridge is None or not bridge.program_completed:
            return torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )
        completion = bridge.completion_mask
        if completion.dtype != torch.bool or completion.shape != (self.num_envs,):
            raise ValueError(
                "Task Program completion mask must be bool with one value per "
                "environment."
            )
        completion = completion.to(device=self.device)
        replay = getattr(self, "_physical_replay", None)
        if replay is not None and replay.frame(self) != replay.snapshot.steps:
            return torch.zeros_like(completion)
        return completion

    def _evaluate_final_task_acceptance(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, _FinalMeasurement | None]:
        completion = self._final_completion_mask()
        bridge = self._active_task_program_bridge
        if (bridge is None or not bridge.program_completed) and not bool(
            completion.any()
        ):
            return completion, completion, None
        measured = _measure_final_conditions(self, self._final_acceptance_rules)
        return completion & measured.satisfied, completion, measured

    def _final_completion_mask(self) -> torch.Tensor:
        saved = getattr(self, "_final_program_completion", None)
        current = self.episode_context.episode_ids

        def qualified(episodes, mask):
            if episodes is None or episodes == current:
                return mask
            return mask & torch.tensor(
                [
                    before == after
                    for before, after in zip(episodes, current, strict=True)
                ],
                device=mask.device,
                dtype=torch.bool,
            )

        mask = qualified(
            getattr(self, "_final_bridge_episodes", None),
            self._task_program_completion_mask(),
        )
        if saved is not None:
            episodes, previous = saved
            mask = mask | qualified(episodes, previous)
        replay = getattr(self, "_physical_replay", None)
        if replay is not None and replay.frame(self) != replay.snapshot.steps:
            return torch.zeros_like(mask)
        return mask

    def _finalize_demo_task_success(
        self, execution_eligible: tuple[bool, ...]
    ) -> torch.Tensor | None:
        """Take the final demo measurement for its terminal result and metadata."""
        if not getattr(self, "_final_acceptance_rules", ()):
            return None
        accepted, completion, measurement = self._evaluate_final_task_acceptance()
        self._pending_final_acceptance = (
            self.episode_context.episode_ids,
            execution_eligible,
            completion.clone(),
            measurement,
        )
        return accepted

    def create_demo_segments(
        self,
        *args,
        task_program: TaskProgramCfg | CompiledTaskProgram | None = None,
        expansion_profile: CombinedExpansionProfile | None = None,
        expansion_candidate_index: int = 0,
        expansion_recipe_indices: tuple[int, ...] | None = None,
        nominal_source_factory: TaskProgramNominalSourceFactory | None = None,
        plan_transform_factory: TaskProgramPlanTransformFactory | None = None,
        **kwargs,
    ) -> Iterable[DemoSegment] | None:
        """Create the semantic segments that make up one task episode.

        An episode-level ``task_program`` takes precedence over the static
        configuration. This lets trusted callers supply a model-produced,
        already compiled program without mutating :attr:`cfg`. Otherwise, a
        configured program is compiled through the injected adapter. With no
        selected program, the legacy action-list path remains unchanged.

        Args:
            *args: Positional arguments forwarded to the legacy planner.
            task_program: Optional episode-level program config or provider-free
                compiled program.
            expansion_profile: Optional source-neutral profile applied to
                matching grounded Atomic calls in this episode.
            expansion_candidate_index: FIFO candidate ordinal selected from
                each generated reference.
            **kwargs: Keyword arguments forwarded to the legacy planner.

        Returns:
            Segment sequence, or ``None`` when planning fails.
        """
        selected_program = task_program
        if selected_program is None:
            selected_program = getattr(
                getattr(self, "cfg", None),
                "task_program",
                None,
            )
        if selected_program is not None:
            from embodichain.lab.task_program import CompiledTaskProgram

            compiled_program = (
                selected_program
                if type(selected_program) is CompiledTaskProgram
                else self.compile_task_program(selected_program)
            )
            if (
                expansion_profile is None
                and expansion_candidate_index == 0
                and expansion_recipe_indices is None
                and nominal_source_factory is None
                and plan_transform_factory is None
            ):
                bridge = self.create_task_program_bridge(compiled_program)
            else:
                bridge_kwargs = {}
                if nominal_source_factory is not None:
                    bridge_kwargs["nominal_source_factory"] = nominal_source_factory
                if plan_transform_factory is not None:
                    bridge_kwargs["plan_transform_factory"] = plan_transform_factory
                bridge = self.create_task_program_bridge(
                    compiled_program,
                    expansion_profile=expansion_profile,
                    expansion_candidate_index=expansion_candidate_index,
                    expansion_recipe_indices=expansion_recipe_indices,
                    **bridge_kwargs,
                )
            self._active_task_program_bridge = bridge
            if getattr(self, "_final_acceptance_rules", ()):
                self._final_bridge_episodes = self.episode_context.episode_ids
            if getattr(self, "_demo_instruction_snapshot", None) is None:
                from .demo import resolve_demo_instruction

                text, source = resolve_demo_instruction(
                    self, task_program=compiled_program
                )
                for metadata in getattr(self, "_demo_episode_metadata", []):
                    metadata["instruction"] = text
                    metadata["instruction_source"] = source
            return bridge.iter_segments()

        if expansion_profile is not None:
            raise ValueError("expansion_profile requires a selected Task Program")
        if nominal_source_factory is not None:
            raise ValueError("nominal_source_factory requires a selected Task Program")
        if plan_transform_factory is not None:
            raise ValueError("plan_transform_factory requires a selected Task Program")
        if expansion_candidate_index != 0:
            raise ValueError(
                "expansion_candidate_index requires a selected Task Program"
            )
        if expansion_recipe_indices is not None:
            raise ValueError(
                "expansion_recipe_indices requires a selected Task Program"
            )

        actions = self.create_demo_action_list(*args, **kwargs)
        if actions is None:
            return None
        return (DemoSegment(actions=actions, name="legacy"),)

    def save_trajectory(
        self,
        path: str,
        env_ids: Sequence[int] | None = None,
        *,
        purpose: str = "recording",
        extra_meta: Mapping[str, Any] | None = None,
    ) -> str:
        """Save a causally aligned trajectory to a ``.pt`` file.

        ``states[t]`` is the state immediately before ``actions[t]`` is applied,
        matching the frame alignment used by expert/LeRobot trajectories. New
        recordings also include ``terminal_states`` and described
        ``raw_observations`` with all T+1 state boundaries.
        A sealed deployment evaluation also includes its exact raw capture in
        ``final_acceptance_observations``.

        Args:
            path: Destination ``.pt`` file path.
            env_ids: Env indices to save (default: all). Each saved env's actual
                recorded length is stored in ``meta["lengths"]``.
            purpose: Normal recording or explicitly labeled diagnostic export.
            extra_meta: Additional artifact metadata; standard fields cannot be replaced.

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
        if purpose not in ("recording", "diagnostic"):
            raise ValueError("Trajectory purpose must be recording or diagnostic.")
        if purpose == "recording" and len(
            self.episode_context.filter_recordable(env_ids)
        ) != len(env_ids):
            raise RuntimeError(
                "Cannot save uninitialized, pending or rejected recording rows."
            )
        if not env_ids:
            raise ValueError("env_ids must contain at least one environment row.")
        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        lengths = self._traj_steps[env_ids_t]
        max_len = int(lengths.max().item()) if len(env_ids) > 0 else 0
        manager = getattr(self, "observation_manager", None)
        recording = (
            self._rollout_storage
            if manager is not None and manager.has_history
            else None
        )
        history = (
            export_observation_history(manager, env_ids, device=self.device)
            if recording is not None and purpose == "recording"
            else None
        )
        if history is not None:
            all_states = TensorDict(
                {}, batch_size=[len(env_ids), max_len + 1], device=self.device
            )
            for key in self._traj_buffer["states"].keys(
                include_nested=True, leaves_only=True
            ):
                all_states[key] = history["fields"]["/".join(key)]["values"]
            states = all_states[:, :max_len]
            terminal_states = all_states[
                torch.arange(len(env_ids), device=self.device), lengths
            ]
            actions = recording["raw_actions"][env_ids_t, :max_len].detach()
        else:
            sub = self._traj_buffer[env_ids_t]
            states = sub["states"][:, :max_len].clone()
            actions = sub["actions"][:, :max_len].clone()
            terminal_states = (
                sub["states"][
                    torch.arange(len(env_ids), device=self.device), lengths
                ].clone()
                if recording is not None
                else None
            )
        meta = {
            "artifact_purpose": purpose,
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
        if "cables" in states.keys():
            # Cable observations retain world coordinates. Keep each exported
            # row's origin so action replay can validate another arena placement.
            meta["arena_origins_w"] = (
                self.sim.arena_offsets[env_ids_t].detach().cpu().tolist()
            )
        meta.update(self._trajectory_action_metadata())
        final_observations = _export_final_observations(
            getattr(self, "_final_acceptance_rules", ()),
            getattr(self, "_sealed_final_observations", {}),
            meta["demo_episodes"],
        )
        meta["run_uuid"] = meta["demo_episodes"][0].get("run_uuid")
        meta["episode_uuids"] = [
            episode.get("episode_uuid") for episode in meta["demo_episodes"]
        ]
        meta["source_episode_uuids"] = [
            episode.get("source_episode_uuid") for episode in meta["demo_episodes"]
        ]
        meta["config_hash"] = meta["demo_episodes"][0].get("config_hash")
        meta["provenance"] = copy.deepcopy(
            meta["demo_episodes"][0].get("provenance", {})
        )
        meta["initial_state_step"] = 0
        if len(env_ids) == 1:
            meta["episode_uuid"] = meta["episode_uuids"][0]
            meta["source_episode_uuid"] = meta["source_episode_uuids"][0]
            meta["program_hash"] = meta["demo_episodes"][0].get("program_hash")
        if extra_meta is not None:
            if not isinstance(extra_meta, Mapping) or not all(
                type(key) is str for key in extra_meta
            ):
                raise TypeError("extra_meta must be a string-keyed mapping.")
            duplicate = set(extra_meta) & set(meta)
            if duplicate:
                raise ValueError(
                    f"extra_meta cannot replace trajectory fields: {sorted(duplicate)}"
                )
            meta.update(copy.deepcopy(dict(extra_meta)))
        replay_path = str(Path(path).resolve())
        for episode in meta["demo_episodes"]:
            episode["trajectory_path"] = replay_path
            episode["replay_artifact"] = {
                "path": replay_path,
                "source_episode_uuid": episode.get("episode_uuid"),
                "env_id": episode["env_id"],
                "initial_state_step": 0,
                "length": episode["length"],
                "state_alignment": "source_episode",
                "external": True,
            }
        destination = Path(path)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                payload = {"states": states, "actions": actions, "meta": meta}
                if terminal_states is not None:
                    payload["terminal_states"] = terminal_states
                    meta["state_frame_layout"] = "pre_action_plus_terminal/v1"
                if history is not None:
                    payload["raw_observations"] = history
                if final_observations is not None:
                    payload[_FINAL_OBSERVATIONS_KEY] = final_observations
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            directory = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            Path(temporary).unlink(missing_ok=True)
        if purpose == "recording":
            for env_id, episode in zip(env_ids, meta["demo_episodes"]):
                live_metadata = self._demo_episode_metadata[env_id]
                live_metadata["trajectory_path"] = replay_path
                live_metadata["replay_artifact"] = copy.deepcopy(
                    episode["replay_artifact"]
                )
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
        metadata = self._demo_episode_metadata[env_id]
        existing_path = metadata.get("trajectory_path")
        replay = metadata.get("replay_artifact", {})
        if (
            existing_path is not None
            and Path(existing_path).is_file()
            and replay.get("length") == int(self._traj_steps[env_id].item())
        ):
            return existing_path
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

        executed_qpos_history = getattr(self, "_expert_controller_qpos_history", None)
        if executed_qpos_history is not None:
            for rows in executed_qpos_history:
                rows.clear()

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
            _close_raw_rollout(getattr(self, "_rollout_storage", None))
            try:
                self._discard_pending_recordings()
            except Exception as error:
                errors.append(error)
            if self.dataset_manager:
                try:
                    self.dataset_manager.finalize()
                except Exception as error:
                    errors.append(error)

            # The synchronous demo executor owns its normal abort/safe-hold
            # handshake. Closing here is pure cleanup, including for an
            # abandoned manual generator: never guess an action acknowledgement
            # or advance physics while detaching runtime observation hooks.
            adapter = getattr(self, "_task_program_adapter", None)
            if adapter is not None:
                try:
                    adapter.close()
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
