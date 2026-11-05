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

"""Observation manager for orchestrating operations based on different simulation observations."""

from __future__ import annotations

import inspect
from contextvars import ContextVar, Token
from itertools import count
from types import MappingProxyType
from collections.abc import Callable
from dataclasses import replace
import torch
from collections.abc import Sequence, Mapping
from copy import deepcopy
from prettytable import PrettyTable
from tensordict import TensorDictBase
from typing import TYPE_CHECKING, Union, Any

from embodichain.utils import logger
from embodichain.lab.sim.types import EnvObs
from embodichain.lab.gym.utils.gym_utils import (
    fetch_data_from_dict,
    assign_data_to_dict,
    _read_raw_rollout,
    _raw_rollout_state,
)
from .observation_types import (
    ObservationDescriptor,
    ObservationSample,
    ObservationFrame,
    ObservationHistorySnapshot,
)
from .manager_base import Functor, ManagerBase
from .cfg import ObservationCfg

__all__ = ["ObservationManager"]

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv
    from embodichain.lab.gym.envs.episode_context import EpisodeContext


class ObservationManager(ManagerBase):
    """Manager for orchestrating operations based on different simulation observations.

    The default observation space will contain two observation groups:
        - `robot`: Contains the default observations related to the robot.
            - `qpos`: The joint positions of the robot.
            - `qvel`: The joint velocities of the robot.
            - `qf`: The joint forces of the robot.
        - `sensor`: Contains the observations related to the sensors which are enabled in the environment.

    The observation manager offers two modes of operation:
        - `modify`: This mode perform data fetching and modification on existing observation data.
        - `add`: This mode perform new observation computation and add new observation data to the observation space.
    """

    _env: EmbodiedEnv
    """The environment instance."""

    def __init__(
        self,
        cfg: object,
        env: EmbodiedEnv,
        *,
        recording: Mapping[str, ObservationCfg] | None = None,
    ) -> None:
        """Initialize the observation manager.

        Args:
            cfg: A configuration object or dictionary (``dict[str, ObservationCfg]``).
            env: An environment object.
            recording: Required raw recording terms. Their callables and
                parameters are resolved here before matching configured sources.
        """

        self._recording_cfgs = deepcopy(dict(recording or {}))
        self._has_raw_channels = False
        self._required_fields: tuple[str, ...] = ()
        self._schema: dict[str, ObservationDescriptor] | None = None
        self._validated_frame: ObservationFrame | None = None
        self._field_ids: dict[str, str] = {}
        self._rollout_storage: TensorDictBase | None = None
        self._mode_functor_names: dict[str, list[str]] = dict()
        self._mode_functor_cfgs: dict[str, list[ObservationCfg]] = dict()
        self._mode_class_functor_cfgs: dict[str, list[ObservationCfg]] = dict()

        # call the base class (this will parse the functors config)
        # ManagerBase dispatches preparation only for a truthy configuration.
        super().__init__(cfg or self._recording_cfgs, env)

    def __str__(self) -> str:
        """Returns: A string representation for observation manager."""
        functor_num = sum(len(v) for v in self._mode_functor_names.values())
        msg = f"<ObservationManager> contains {functor_num} active functors.\n"

        # add info on each mode
        for mode in self._mode_functor_names:
            # create table for functor information
            table = PrettyTable()
            table.title = f"Active Observation Functors in Mode: '{mode}'"

            table.field_names = ["Index", "Name"]
            table.align["Name"] = "l"
            for index, name in enumerate(self._mode_functor_names[mode]):
                table.add_row([index, name])

            # convert table to string
            msg += table.get_string()
            msg += "\n"

        return msg

    """
    Properties.
    """

    @property
    def active_functors(self) -> dict[str, list[str]]:
        """Name of active observation functors.

        The keys are the modes of observation and the values are the names of the observation functors.
        """
        return self._mode_functor_names

    """
    Operations.
    """

    def reset(self, env_ids: Union[Sequence[int], None] = None) -> dict[str, float]:
        # call all functors that are classes
        for mode_cfg in self._mode_class_functor_cfgs.values():
            for functor_cfg in mode_cfg:
                functor_cfg.func.reset(env_ids=env_ids)

        # nothing to log here
        return {}

    @property
    def has_raw_channels(self) -> bool:
        """Whether this manager declares raw recording channels."""
        return self._has_raw_channels

    @property
    def is_capturing(self) -> bool:
        """Whether the host is inside an explicit observation boundary."""
        return _current_capture.get() is not None

    @property
    def schema(self) -> Mapping[str, ObservationDescriptor]:
        """Return the single bound raw-field schema, shared by all consumers."""
        return MappingProxyType(self._schema or {})

    @property
    def field_ids(self) -> Mapping[str, str]:
        """Map configured aliases to the canonical history field IDs."""
        return MappingProxyType(self._field_ids)

    def capture(
        self,
        acquire: Callable[[], EnvObs],
        *,
        episode_context: EpisodeContext,
        timestamp: float | None = None,
    ) -> ObservationFrame:
        """Capture one observation boundary and validate its bound schema.

        Args:
            acquire: Existing observation acquisition, preserving functor order.
            episode_context: Exact host episode owner.
            timestamp: Source simulation time, or None.

        Returns:
            The raw boundary and its actual policy view.
        """
        if self._env.episode_context is not episode_context:
            raise ValueError("Observation capture belongs to another episode owner.")
        with _ObservationCapture(
            episode_context.episode_ids, timestamp, episode_owner=episode_context
        ) as capture:
            policy = acquire()
        frame = capture.finish(policy)
        self._bind_schema(frame)
        return frame

    @property
    def has_history(self) -> bool:
        """Whether the host rollout contains raw observation fields."""
        return (
            self._rollout_storage is not None
            and "raw_obs" in self._rollout_storage.keys()
        )

    def bind_rollout_storage(self, storage: TensorDictBase) -> None:
        """Bind read access to host-owned rollout data without another cache.

        Args:
            storage: Existing rollout storage, including its raw observation group.
        """
        _raw_rollout_state(storage, self._env.episode_context)
        self._rollout_storage = storage

    def validate_history_owner(self, episode_context: EpisodeContext) -> None:
        """Validate the actual rollout episode owner before binding a reader.

        Args:
            episode_context: Consumer's expected source episode context.
        """
        _raw_rollout_state(self._rollout_storage, episode_context)

    def snapshot_history(
        self,
        env_ids: Sequence[int] | None = None,
        *,
        device: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """Copy canonical numeric history for a host export adapter.

        Args:
            env_ids: Selected original rows, or all rows.
            device: Destination snapshot device.

        Returns:
            Owned numeric history and temporal identities. Semantics remain
            available through this manager's schema and field IDs.
        """
        return _read_raw_rollout(
            self._rollout_storage,
            self._env.episode_context,
            env_ids=env_ids,
            device=device,
        )

    def _bind_schema(self, frame: ObservationFrame) -> None:
        if frame is self._validated_frame:
            return
        if self._schema is None:
            self._schema = {
                name: sample.descriptor for name, sample in frame.fields.items()
            }
            canonical = {descriptor: name for name, descriptor in self._schema.items()}
            for name in self._required_fields:
                if name in self._schema:
                    canonical[self._schema[name]] = name
            self._field_ids = {
                name: canonical[descriptor] for name, descriptor in self._schema.items()
            }
        if frame.fields.keys() != self._schema.keys():
            raise ValueError("Raw observation field schema changed after binding.")
        for name, descriptor in self._schema.items():
            observed = frame.fields[name].descriptor
            if observed is not descriptor and observed != descriptor:
                raise ValueError(f"Raw observation identity drift for {name!r}.")
        self._validated_frame = frame

    def read_history(
        self,
        frame_indices: Sequence[int],
        *,
        fields: Sequence[str],
        episode_ids: Sequence[int],
        env_ids: Sequence[int] | None = None,
    ) -> ObservationHistorySnapshot:
        """Read exact history slots using IDs from this observation schema.

        Args:
            frame_indices: Row-local state slots in query order.
            fields: Bound observation field IDs or aliases.
            episode_ids: Expected source episodes.
            env_ids: Original rows, or all rows when omitted.

        Returns:
            Owned field tensors and original temporal identities.
        """
        if not self.has_history:
            raise RuntimeError("Observation history is unavailable.")
        names = tuple(fields)
        if not names or len(set(names)) != len(names):
            raise ValueError("History queries require unique nonempty field IDs.")
        ids = {name: self._field_ids[name] for name in names}
        result = _read_raw_rollout(
            self._rollout_storage,
            self._env.episode_context,
            frame_indices,
            fields=tuple(dict.fromkeys(ids.values())),
            episode_ids=episode_ids,
            env_ids=env_ids,
        )
        return replace(
            result,
            values={name: result.values[key] for name, key in ids.items()},
            valid={name: result.valid[key] for name, key in ids.items()},
            acquisition_errors={
                name: result.acquisition_errors[key] for name, key in ids.items()
            },
        )

    def compute(
        self,
        obs: EnvObs,
    ) -> EnvObs:
        """Calls each observation functor in the specified mode.

        This function iterates over all the observation functors in the specified mode and calls the function
        corresponding to the functor. The function is called with the environment instance and the environment
        indices to apply the observation to.

        Args:
            obs: The observation data to apply the observation to.

        Returns:
            The modified observation data.

        Raises:
            ValueError: If the mode is not supported.
        """

        # iterate over all the observation functors
        for mode, functor_cfgs in self._mode_functor_cfgs.items():
            for functor_name, functor_cfg in zip(
                self._mode_functor_names[mode], functor_cfgs
            ):
                functor_cfg: ObservationCfg

                if mode == "modify":
                    data = fetch_data_from_dict(obs, functor_cfg.name)
                    data = self._call_functor(
                        functor_name, functor_cfg, self._env, data
                    )
                elif mode == "add":
                    data = self._call_functor(functor_name, functor_cfg, self._env, obs)
                    if "recording" in functor_cfg.consumers:
                        if not isinstance(data, ObservationSample):
                            raise TypeError(
                                f"Raw observation {functor_name!r} must return "
                                "ObservationSample; a policy Tensor has no physical identity."
                            )
                        capture = _current_capture.get()
                        if capture is not None:
                            if functor_cfg.name in capture.fields:
                                raise ValueError(
                                    f"Duplicate raw observation field {functor_cfg.name!r}."
                                )
                            capture.fields[functor_cfg.name] = capture.acquire(
                                data.descriptor, lambda: data
                            )
                    if "policy" in functor_cfg.consumers:
                        # In-place policy transforms must not alter original samples.
                        value = (
                            data.values.clone()
                            if isinstance(data, ObservationSample)
                            else data
                        )
                        assign_data_to_dict(obs, functor_cfg.name, value)
                else:
                    logger.log_error(f"Unsupported observation mode '{mode}'.")

        return obs

    def get_functor_cfg(self, functor_name: str) -> ObservationCfg:
        """Gets the configuration for the specified functor.

        The method finds the functor by name by searching through all the modes.
        It then returns the configuration of the functor with the first matching name.

        Args:
            functor_name: The name of the observation functor.

        Returns:
            The configuration of the observation functor.

        Raises:
            ValueError: If the functor name is not found.
        """
        for mode, functors in self._mode_functor_names.items():
            if functor_name in functors:
                return self._mode_functor_cfgs[mode][functors.index(functor_name)]
        logger.log_error(f"observation functor '{functor_name}' not found.")

    """
    Helper functions.
    """

    def _prepare_functors(self) -> None:
        items = dict(self.cfg if isinstance(self.cfg, dict) else vars(self.cfg))
        required = self._recording_cfgs
        self._required_fields = tuple(cfg.name for cfg in required.values())
        # Normalize through the same owner before comparing aliases. In
        # particular a module:callable string and its resolved function agree.
        prepared: dict[str, ObservationCfg] = {}
        fields: dict[str, ObservationCfg] = {}
        for name, cfg in items.items():
            if cfg is not None:
                self._prepare_source(name, cfg)
                prepared[name] = cfg
                if "recording" in cfg.consumers:
                    if cfg.name in fields:
                        raise ValueError(
                            f"Duplicate raw observation field {cfg.name!r}."
                        )
                    fields[cfg.name] = cfg
        for name, cfg in required.items():
            self._prepare_source(name, cfg)
            existing = fields.get(cfg.name)
            if existing is not None:
                if not self._same_source(existing, cfg):
                    raise ValueError(
                        f"Raw field {cfg.name!r} conflicts with the replay schema."
                    )
                continue
            if name in prepared:
                raise ValueError(
                    f"Observation name {name!r} collides with recording declarations."
                )
            prepared[name] = cfg
            fields[cfg.name] = cfg
        for name, cfg in prepared.items():
            mode = cfg.mode
            self._mode_functor_names.setdefault(mode, []).append(name)
            self._mode_functor_cfgs.setdefault(mode, []).append(cfg)
            self._mode_class_functor_cfgs.setdefault(mode, [])
            if isinstance(cfg.func, Functor):
                self._mode_class_functor_cfgs[mode].append(cfg)
            self._has_raw_channels |= "recording" in cfg.consumers

    def _prepare_source(self, name: str, cfg: ObservationCfg) -> None:
        if not isinstance(cfg, ObservationCfg):
            raise TypeError(f"Observation {name!r} requires ObservationCfg.")
        consumers = cfg.consumers
        if (
            not isinstance(consumers, (tuple, list))
            or not consumers
            or any(value not in ("policy", "recording") for value in consumers)
            or len(set(consumers)) != len(consumers)
        ):
            raise ValueError(f"Invalid observation consumers for {name!r}.")
        cfg.consumers = tuple(consumers)
        if cfg.mode == "modify" and cfg.consumers != ("policy",):
            raise ValueError("Modify functors apply only to policy observations.")
        self._resolve_common_functor_cfg(name, cfg, min_argc=2)

    @staticmethod
    def _same_source(left: ObservationCfg, right: ObservationCfg) -> bool:
        if left.func is not right.func:
            return False
        defaults = {
            name: parameter.default
            for name, parameter in inspect.signature(left.func).parameters.items()
            if parameter.default is not inspect.Parameter.empty
        }
        return {**defaults, **left.params} == {**defaults, **right.params}


_current_capture: ContextVar[_ObservationCapture | None] = ContextVar(
    "embodichain_observation_capture", default=None
)
_revisions = count()


class _ObservationCapture:
    """Memoize physical acquisition only inside an explicit boundary scope."""

    def __init__(
        self,
        episode_ids: tuple[int, ...],
        timestamp: float | None = None,
        *,
        episode_owner: EpisodeContext | None = None,
    ):
        self.episode_ids = episode_ids
        self.episode_owner = episode_owner
        self.timestamp = timestamp
        self.revision = next(_revisions)
        self.samples: dict[
            ObservationDescriptor | tuple[str, ...], ObservationSample
        ] = {}
        self.fields: dict[str, ObservationSample] = {}
        self._token: Token | None = None

    def __enter__(self) -> _ObservationCapture:
        self._token = _current_capture.set(self)
        return self

    def __exit__(self, *args: Any) -> None:
        _current_capture.reset(self._token)

    def acquire(
        self,
        key: ObservationDescriptor | tuple[str, ...],
        producer: Callable[[], ObservationSample],
    ) -> ObservationSample:
        if key not in self.samples:
            sample = producer()
            if sample.descriptor not in self.samples:
                self.samples[sample.descriptor] = ObservationSample(
                    sample.descriptor,
                    sample.values.detach().clone(),
                    sample.valid.detach().clone(),
                    sample.acquisition_errors,
                )
            self.samples[key] = self.samples[sample.descriptor]
        return self.samples[key]

    def finish(self, policy: Any) -> ObservationFrame:
        return ObservationFrame(
            policy,
            self.fields,
            self.episode_ids,
            self.revision,
            self.timestamp,
            "simulation_time" if self.timestamp is not None else "unavailable",
            self.episode_owner,
        )
