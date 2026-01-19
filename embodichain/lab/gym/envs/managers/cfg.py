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

from __future__ import annotations

import torch

from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any, Literal

from embodichain.lab.sim.objects import Articulation, RigidObject
from embodichain.utils import configclass

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager
    from .manager_base import Functor

    # from .recorder_manager import RecorderTerm


@configclass
class FunctorCfg:
    """Configuration for a functor."""

    func: Callable | Functor = MISSING
    """The function or class to be called for the functor.

    The function must take the environment object as the first argument.
    The remaining arguments are specified in the :attr:`params` attribute.

    It also supports `callable classes`_, i.e. classes that implement the :meth:`__call__`
    method. In this case, the class should inherit from the :class:`Functor` class
    and implement the required methods.

    .. _`callable classes`: https://docs.python.org/3/reference/datamodel.html#object.__call__
    """

    params: dict[str, Any | SceneEntityCfg] = dict()
    """The parameters to be passed to the function as keyword arguments. Defaults to an empty dict.

    .. note::
        If the value is a :class:`SceneEntityCfg` object, the manager will query the scene entity
        from the :class:`SimulationManager` and process the entity's joints and bodies as specified
        in the :class:`SceneEntityCfg` object.
    """


@configclass
class EventCfg(FunctorCfg):
    """Configuration for a event functor.

    The event functor is used to trigger events in the environment at specific times or under specific conditions.
    The `mode` attribute determines when the functor is applied.
    - `startup`: The functor is applied when the environment is started.
    - `interval`: The functor is applied at each env step.
    - `reset`: The functor is applied when the environment is reset.
    """

    mode: Literal["startup", "interval", "reset"] = "reset"
    """The mode in which the event functor is applied.

    Note:
        The mode name ``"interval"`` is a special mode that is handled by the
        manager Hence, its name is reserved and cannot be used for other modes.
    """

    # TODO: maybe support simulation time-based events (time = step * (physics_dt * sim_steps_per_control))
    interval_step: int = 10
    """The number of environment step after which the functor is applied. Defaults to 4."""

    is_global: bool = False
    """Whether the event should be tracked on a per-environment basis. Defaults to False.

    If True, the same interval step is used for all the environment instances.
    If False, the interval step is sampled independently for each environment instance
    and the functor is applied when the current step hits the interval step for that instance.

    Note:
        This is only used if the mode is ``"interval"``.
    """


@configclass
class ObservationCfg(FunctorCfg):
    """Configuration for an observation functor.

    The observation functor is used to compute observations for the environment. The `mode` attribute
    determines whether the observation is already present in the observation space or not.
    """

    mode: Literal["modify", "add"] = "modify"
    """The mode for the observation computation.

    - `modify`: The observation is already present in the observation space, updated the value in-place.
    - `add`: The observation is not present in the observation space, add a new entry to the observation space.
    """

    name: str = MISSING
    """The name of the observation.

    The name can be a new key to observation space, eg:
        - `object_position`: shape of (num_envs, 3)
        - `robot/eef_pose`: shape of (num_envs, 7) or (num_envs, 4, 4)
        - `sensor/cam_high/mask`: shape of (num_envs, H, W)
    or a existing key to modify, eg:
        - `robot/qpos`: shape of (num_envs, num_dofs)
    `/` is used to separate different levels of hierarchy in the observation dictionary.
    """


@configclass
class SceneEntityCfg:
    """Configuration for a scene entity that is used by the manager's functor.

    This class is used to specify the name of the scene entity that is queried from the
    :class:`SimulationManager` and passed to the manager's functor.
    """

    uid: str = MISSING
    """The name of the scene entity.

    This is the name defined in the scene configuration file. See the :class:`SimulationManagerCfg`
    class for more details.
    """

    joint_names: str | list[str] | None = None
    """The names of the joints from the scene entity. Defaults to None.

    The names can be either joint names or a regular expression matching the joint names.

    These are converted to joint indices on initialization of the manager and passed to the functor
    as a list of joint indices under :attr:`joint_ids`.
    """

    joint_ids: list[int] | slice = slice(None)
    """The indices of the joints from the asset required by the functor. Defaults to slice(None), which means
    all the joints in the asset (if present).

    If :attr:`joint_names` is specified, this is filled in automatically on initialization of the
    manager.
    """

    link_names: str | list[str] | None = None
    """The names of the links from the asset required by the functor. Defaults to None.

    The names can be either link names or a regular expression matching the link names.
    """

    control_parts: str | list[str] | None = None
    """The names of the control parts from the asset(only support for robot) required by the functor. Defaults to None.
    """

    # TODO: Maybe support tendon names and ids in the future.

    body_names: str | list[str] | None = None
    """The names of the bodies from the asset required by the functor. Defaults to None.

    The names can be either body names or a regular expression matching the body names.

    These are converted to body indices on initialization of the manager and passed to the functor
    function as a list of body indices under :attr:`body_ids`.
    """

    body_ids: list[int] | slice = slice(None)
    """The indices of the bodies from the asset required by the functor. Defaults to slice(None), which means
    all the bodies in the asset.

    If :attr:`body_names` is specified, this is filled in automatically on initialization of the
    manager.
    """

    # TODO: Maybe support object collection (same as IsaacLab definitions).

    preserve_order: bool = False
    """Whether to preserve indices ordering to match with that in the specified joint, body, or object collection names.
    Defaults to False.

    If False, the ordering of the indices are sorted in ascending order (i.e. the ordering in the entity's joints,
    bodies, or object in the object collection). Otherwise, the indices are preserved in the order of the specified
    joint, body, or object collection names.

    For more details, see the :meth:`isaaclab.utils.string.resolve_matching_names` function.

    .. note::
        This attribute is only used when :attr:`joint_names`, :attr:`body_names` are specified.

    """

    def resolve(self, scene: SimulationManager):
        """Resolves the scene entity and converts the joint and body names to indices.

        This function examines the scene entity from the :class:`SimulationManager` and resolves the indices
        and names of the joints and bodies. It is an expensive operation as it resolves regular expressions
        and should be called only once.

        Args:
            scene: The interactive scene instance.

        Raises:
            ValueError: If the scene entity is not found.
            ValueError: If both ``joint_names`` and ``joint_ids`` are specified and are not consistent.
            ValueError: If both ``body_names`` and ``body_ids`` are specified and are not consistent.
        """
        # check if the entity is valid
        asset_uids = scene.asset_uids
        if self.uid not in asset_uids:
            raise ValueError(
                f"The scene entity '{self.uid}' does not exist. Available entities: {asset_uids}."
            )

        # convert joint names to indices based on regex
        self._resolve_joint_names(scene)

        # convert body names to indices based on regex
        self._resolve_body_names(scene)

    def _resolve_joint_names(self, scene: SimulationManager):
        # convert joint names to indices based on regex
        if self.joint_names is not None or self.joint_ids != slice(None):
            entity: Articulation = scene.get_articulation(self.uid)
            # -- if both are not their default values, check if they are valid
            if self.joint_names is not None and self.joint_ids != slice(None):
                if isinstance(self.joint_names, str):
                    self.joint_names = [self.joint_names]
                if isinstance(self.joint_ids, int):
                    self.joint_ids = [self.joint_ids]
                joint_ids, _ = entity.find_joints(
                    self.joint_names, preserve_order=self.preserve_order
                )
                joint_names = [entity.joint_names[i] for i in self.joint_ids]
                if joint_ids != self.joint_ids or joint_names != self.joint_names:
                    raise ValueError(
                        "Both 'joint_names' and 'joint_ids' are specified, and are not consistent."
                        f"\n\tfrom joint names: {self.joint_names} [{joint_ids}]"
                        f"\n\tfrom joint ids: {joint_names} [{self.joint_ids}]"
                        "\nHint: Use either 'joint_names' or 'joint_ids' to avoid confusion."
                    )
            # -- from joint names to joint indices
            elif self.joint_names is not None:
                if isinstance(self.joint_names, str):
                    self.joint_names = [self.joint_names]
                self.joint_ids, _ = entity.find_joints(
                    self.joint_names, preserve_order=self.preserve_order
                )
                # performance optimization (slice offers faster indexing than list of indices)
                # only all joint in the entity order are selected
                if (
                    len(self.joint_ids) == entity.num_joints
                    and self.joint_names == entity.joint_names
                ):
                    self.joint_ids = slice(None)
            # -- from joint indices to joint names
            elif self.joint_ids != slice(None):
                if isinstance(self.joint_ids, int):
                    self.joint_ids = [self.joint_ids]
                self.joint_names = [entity.joint_names[i] for i in self.joint_ids]

    def _resolve_body_names(self, scene: SimulationManager):
        # convert body names to indices based on regex
        if self.body_names is not None or self.body_ids != slice(None):
            entity: RigidObject = scene.get_rigid_object(self.uid)
            # -- if both are not their default values, check if they are valid
            if self.body_names is not None and self.body_ids != slice(None):
                if isinstance(self.body_names, str):
                    self.body_names = [self.body_names]
                if isinstance(self.body_ids, int):
                    self.body_ids = [self.body_ids]
                body_ids, _ = entity.find_bodies(
                    self.body_names, preserve_order=self.preserve_order
                )
                body_names = [entity.body_names[i] for i in self.body_ids]
                if body_ids != self.body_ids or body_names != self.body_names:
                    raise ValueError(
                        "Both 'body_names' and 'body_ids' are specified, and are not consistent."
                        f"\n\tfrom body names: {self.body_names} [{body_ids}]"
                        f"\n\tfrom body ids: {body_names} [{self.body_ids}]"
                        "\nHint: Use either 'body_names' or 'body_ids' to avoid confusion."
                    )
            # -- from body names to body indices
            elif self.body_names is not None:
                if isinstance(self.body_names, str):
                    self.body_names = [self.body_names]
                self.body_ids, _ = entity.find_bodies(
                    self.body_names, preserve_order=self.preserve_order
                )
                # performance optimization (slice offers faster indexing than list of indices)
                # only all bodies in the entity order are selected
                if (
                    len(self.body_ids) == entity.num_bodies
                    and self.body_names == entity.body_names
                ):
                    self.body_ids = slice(None)
            # -- from body indices to body names
            elif self.body_ids != slice(None):
                if isinstance(self.body_ids, int):
                    self.body_ids = [self.body_ids]
                self.body_names = [entity.body_names[i] for i in self.body_ids]


@configclass
class RewardCfg(FunctorCfg):
    """Configuration for a reward functor.

    The reward functor is used to compute rewards for the environment. The `mode` attribute
    determines how the reward is combined with existing rewards.
    """

    mode: Literal["add", "replace"] = "add"
    """The mode for the reward computation.

    - `add`: The reward is added to the existing total reward.
    - `replace`: The reward replaces the total reward (useful for single reward functions).
    """

    weight: float = 1.0
    """The weight multiplier for this reward term.

    This value is used to scale the reward before adding it to the total reward.
    Default is 1.0 (no scaling).
    """


@configclass
class DatasetFunctorCfg(FunctorCfg):
    """Configuration for dataset collection functors.

    Dataset functors are called with mode="save" which handles both:
    - Recording observation-action pairs on every step
    - Auto-saving episodes when dones=True
    """

    mode: Literal["save"] = "save"
