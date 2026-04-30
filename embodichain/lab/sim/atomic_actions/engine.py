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

import torch
from typing import Any, Dict, List, Optional, Type, Union, TYPE_CHECKING

from embodichain.lab.sim.planners import PlanResult
from embodichain.utils import logger
from .core import AtomicAction, ObjectSemantics, ActionCfg

if TYPE_CHECKING:
    from embodichain.lab.sim.planners import MotionGenerator
    from embodichain.lab.sim.objects import Robot


# =============================================================================
# Global Action Registry
# =============================================================================

_global_action_registry: Dict[str, Type[AtomicAction]] = {}
_global_action_configs: Dict[str, Type[ActionCfg]] = {}


def register_action(
    name: str,
    action_class: Type[AtomicAction],
    config_class: Optional[Type[ActionCfg]] = None,
) -> None:
    """Register a custom atomic action class globally.

    This function allows registration of custom action types that can then
    be instantiated by the AtomicActionEngine.

    Args:
        name: Unique identifier for the action type
        action_class: The AtomicAction subclass to register
        config_class: Optional configuration class for the action

    Example:
        >>> class MyCustomAction(AtomicAction):
        ...     def execute(self, target, **kwargs):
        ...         # Implementation
        ...         pass
        ...     def validate(self, target, **kwargs):
        ...         return True
        >>> register_action("my_custom", MyCustomAction)
    """
    _global_action_registry[name] = action_class
    if config_class is not None:
        _global_action_configs[name] = config_class


def unregister_action(name: str) -> None:
    """Unregister an action type.

    Args:
        name: The action type identifier to remove
    """
    _global_action_registry.pop(name, None)
    _global_action_configs.pop(name, None)


def get_registered_actions() -> Dict[str, Type[AtomicAction]]:
    """Get all registered action types.

    Returns:
        Dictionary mapping action names to their classes
    """
    return _global_action_registry.copy()


# =============================================================================
# Semantic Analyzer
# =============================================================================


class SemanticAnalyzer:
    """Analyzes objects and provides ObjectSemantics for atomic actions."""

    def __init__(self):
        self._object_cache: Dict[str, ObjectSemantics] = {}

    def analyze(
        self,
        label: str,
        geometry: Optional[Dict[str, Any]] = None,
        custom_config: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> ObjectSemantics:
        """Analyze object by label and return ObjectSemantics.

        This is a placeholder implementation that should be extended
        with actual object detection and affordance computation.

        Args:
            label: Object category label (e.g., "apple", "bottle")
            geometry: Optional geometry payload. Can include mesh tensors:
                ``mesh_vertices`` [N, 3] and ``mesh_triangles`` [M, 3].
            custom_config: Optional user-defined affordance configuration.
            use_cache: Whether to use cached semantics when available.

        Returns:
            ObjectSemantics containing affordance data
        """
        # Only use cache for default analyze path
        if (
            use_cache
            and geometry is None
            and custom_config is None
            and label in self._object_cache
        ):
            return self._object_cache[label]

        # Create default semantics (placeholder implementation)
        from .core import AntipodalAffordance

        # Generate default grasp poses based on object type
        default_poses = torch.eye(4).unsqueeze(0)
        default_poses[0, 2, 3] = 0.1  # Default offset

        default_geometry: Dict[str, Any] = {"bounding_box": [0.1, 0.1, 0.1]}
        if geometry is not None:
            default_geometry.update(geometry)

        grasp_affordance = AntipodalAffordance(
            object_label=label,
            poses=default_poses,
            grasp_types=["default"],
            custom_config=custom_config or {},
        )

        semantics = ObjectSemantics(
            label=label,
            affordance=grasp_affordance,
            geometry=default_geometry,
            properties={"mass": 1.0, "friction": 0.5},
        )

        # Cache only default path
        if use_cache and geometry is None and custom_config is None:
            self._object_cache[label] = semantics
        return semantics

    def clear_cache(self) -> None:
        """Clear the object semantics cache."""
        self._object_cache.clear()


# =============================================================================
# Atomic Action Engine
# =============================================================================


class AtomicActionEngine:
    """Central engine for managing and executing atomic actions."""

    def __init__(
        self,
        motion_generator: "MotionGenerator",
        actions_cfg_list: Optional[List[ActionCfg]] = None,
    ):
        self.motion_generator = motion_generator
        self.robot = self.motion_generator.robot
        self.device = self.motion_generator.device

        # Semantic analyzer for object understanding
        self._semantic_analyzer = SemanticAnalyzer()

        # Initialize default actions
        self._actions: List[AtomicAction] = self._init_actions(actions_cfg_list)

    def _init_actions(self, actions_cfg_list: Optional[List[ActionCfg]] = None):
        actions: List[AtomicAction] = []
        from .actions import MoveAction, PickUpAction, PlaceAction

        name_action_map = {
            "move": MoveAction,
            "pick_up": PickUpAction,
            "place": PlaceAction,
        }
        if actions_cfg_list is not None:
            for cfg in actions_cfg_list:
                action_class = name_action_map.get(cfg.name)
                if action_class is None:
                    logger.log_error(f"Unknown action name in config: {cfg.name}")
                instance = action_class(motion_generator=self.motion_generator, cfg=cfg)
                actions.append(instance)
        return actions

    def execute_static(
        self,
        target_list: List[Union[torch.Tensor, str, ObjectSemantics, Dict[str, Any]]],
    ) -> tuple[bool, torch.Tensor]:
        """Execute a static move action to target poses without motion planning."""
        if len(target_list) != len(self._actions):
            logger.log_error(
                f"Length of target_list ({len(target_list)}) must match number of actions ({len(self._actions)})."
            )
        start_qpos = self.motion_generator.robot.get_qpos()
        n_envs = start_qpos.shape[0]
        all_dof = self.motion_generator.robot.dof
        all_trajectory = torch.empty(
            size=(n_envs, 0, all_dof), dtype=torch.float32, device=self.device
        )

        for i, target in enumerate(target_list):
            atom_action = self._actions[i]
            target = self._resolve_target(target)
            control_part = atom_action.control_part
            arm_joint_ids = self.motion_generator.robot.get_joint_ids(name=control_part)
            start_qpos_part = start_qpos[:, arm_joint_ids]
            is_success, traj, joint_ids = atom_action.execute(
                target=target, start_qpos=start_qpos_part
            )
            n_waypoints = traj.shape[1]

            traj_full = torch.zeros(
                size=(n_envs, n_waypoints, all_dof),
                dtype=torch.float32,
                device=self.device,
            )
            traj_full[:, :] = start_qpos
            traj_full[:, :, joint_ids] = traj
            all_trajectory = torch.cat((all_trajectory, traj_full), dim=1)
            if is_success:
                # update start qpos
                start_qpos[:, joint_ids] = traj[:, -1, :]
            else:
                return False, all_trajectory
        return True, all_trajectory

    def validate(
        self,
        action_name: str,
        target: Union[torch.Tensor, str, ObjectSemantics, Dict[str, Any]],
        control_part: Optional[str] = None,
        **kwargs,
    ) -> bool:
        """Validate if an action is feasible without executing."""
        if control_part:
            action_key = f"{action_name}_{control_part}"
        else:
            action_key = action_name

        if action_key not in self._actions:
            return False

        action = self._actions[action_key]

        target = self._resolve_target(target)

        return action.validate(target, **kwargs)

    def _resolve_target(
        self,
        target: Union[torch.Tensor, str, ObjectSemantics, Dict[str, Any]],
    ) -> Union[torch.Tensor, ObjectSemantics]:
        """Resolve user target input into tensor pose or ObjectSemantics.

        Supports the convenience dict format in ``execute`` and ``validate``.
        """
        if isinstance(target, torch.Tensor):
            return target

        if isinstance(target, ObjectSemantics):
            return target

        if isinstance(target, str):
            return self._semantic_analyzer.analyze(target)

        if isinstance(target, dict):
            if "pose" in target:
                pose = target["pose"]
                if not isinstance(pose, torch.Tensor):
                    raise TypeError("target['pose'] must be a torch.Tensor")
                return pose

            if "semantics" in target:
                semantics = target["semantics"]
                if not isinstance(semantics, ObjectSemantics):
                    raise TypeError(
                        "target['semantics'] must be an ObjectSemantics instance"
                    )
                return semantics

            label = target.get("label")
            if label is None:
                raise ValueError(
                    "Dict target must provide 'label', or use 'pose'/'semantics'."
                )
            if not isinstance(label, str):
                raise TypeError("target['label'] must be a string")

            geometry = target.get("geometry")
            custom_config = target.get("custom_config")
            use_cache = target.get("use_cache", True)

            semantics = self._semantic_analyzer.analyze(
                label=label,
                geometry=geometry,
                custom_config=custom_config,
                use_cache=use_cache,
            )

            properties = target.get("properties")
            if properties is not None:
                semantics.properties.update(properties)

            uid = target.get("uid")
            if uid is not None:
                semantics.uid = uid

            return semantics

        raise TypeError(
            "target must be torch.Tensor, str, ObjectSemantics, or Dict[str, Any]"
        )

    def get_semantic_analyzer(self) -> SemanticAnalyzer:
        """Get the semantic analyzer for object understanding."""
        return self._semantic_analyzer

    def set_semantic_analyzer(self, analyzer: SemanticAnalyzer) -> None:
        """Set a custom semantic analyzer."""
        self._semantic_analyzer = analyzer
