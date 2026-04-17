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
from typing import Dict, List, Optional, Type, Union, TYPE_CHECKING

from embodichain.lab.sim.planners import PlanResult
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

    def analyze(self, label: str) -> ObjectSemantics:
        """Analyze object by label and return ObjectSemantics.

        This is a placeholder implementation that should be extended
        with actual object detection and affordance computation.

        Args:
            label: Object category label (e.g., "apple", "bottle")

        Returns:
            ObjectSemantics containing affordance data
        """
        # Check cache first
        if label in self._object_cache:
            return self._object_cache[label]

        # Create default semantics (placeholder implementation)
        from .core import GraspPose, InteractionPoints

        # Generate default grasp poses based on object type
        default_poses = torch.eye(4).unsqueeze(0)
        default_poses[0, 2, 3] = 0.1  # Default offset

        grasp_affordance = GraspPose(
            object_label=label,
            poses=default_poses,
            grasp_types=["default"],
        )

        # Default interaction points
        interaction_affordance = InteractionPoints(
            object_label=label,
            points=torch.zeros(1, 3),
            point_types=["contact"],
        )

        semantics = ObjectSemantics(
            label=label,
            affordance=grasp_affordance,
            geometry={"bounding_box": [0.1, 0.1, 0.1]},
            properties={"mass": 1.0, "friction": 0.5},
        )

        # Cache and return
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
        robot: "Robot",
        motion_generator: "MotionGenerator",
        device: torch.device = torch.device("cuda"),
    ):
        self.robot = robot
        self.motion_generator = motion_generator
        self.device = device

        # Registry of action instances
        self._actions: Dict[str, AtomicAction] = {}

        # Semantic analyzer for object understanding
        self._semantic_analyzer = SemanticAnalyzer()

        # Initialize default actions
        self._init_default_actions()

    def _init_default_actions(self):
        """Initialize default atomic action instances."""
        from .actions import ReachAction, GraspAction, MoveAction, ReleaseAction

        control_parts = getattr(self.robot, 'control_parts', None) or ["default"]

        for part in control_parts:
            self.register_action(
                f"reach_{part}",
                ReachAction(
                    motion_generator=self.motion_generator,
                    robot=self.robot,
                    control_part=part,
                    device=self.device,
                )
            )
            self.register_action(
                f"grasp_{part}",
                GraspAction(
                    motion_generator=self.motion_generator,
                    robot=self.robot,
                    control_part=part,
                    device=self.device,
                )
            )
            self.register_action(
                f"move_{part}",
                MoveAction(
                    motion_generator=self.motion_generator,
                    robot=self.robot,
                    control_part=part,
                    device=self.device,
                )
            )
            self.register_action(
                f"release_{part}",
                ReleaseAction(
                    motion_generator=self.motion_generator,
                    robot=self.robot,
                    control_part=part,
                    device=self.device,
                )
            )

        # Register action classes for dynamic instantiation
        for action_name, action_class in _global_action_registry.items():
            # Don't override default actions
            if action_name not in ["reach", "grasp", "move", "release"]:
                for part in control_parts:
                    action_key = f"{action_name}_{part}"
                    if action_key not in self._actions:
                        # Create instance with default config
                        try:
                            instance = action_class(
                                motion_generator=self.motion_generator,
                                robot=self.robot,
                                control_part=part,
                                device=self.device,
                            )
                            self._actions[action_key] = instance
                        except Exception:
                            # Skip if instantiation fails
                            pass

    def register_action(self, name: str, action: AtomicAction):
        """Register a custom atomic action."""
        self._actions[name] = action

    def unregister_action(self, name: str) -> bool:
        """Unregister an action by name.

        Args:
            name: Name of the action to unregister

        Returns:
            True if action was found and removed, False otherwise
        """
        if name in self._actions:
            del self._actions[name]
            return True
        return False

    def get_action_names(self) -> List[str]:
        """Get list of registered action names."""
        return list(self._actions.keys())

    def execute(
        self,
        action_name: str,
        target: Union[torch.Tensor, str, ObjectSemantics],
        control_part: Optional[str] = None,
        **kwargs
    ) -> PlanResult:
        """Execute an atomic action.

        Args:
            action_name: Name of registered action
            target: Target pose, object label, or ObjectSemantics
            control_part: Robot control part to use
            **kwargs: Additional action parameters

        Returns:
            PlanResult with trajectory (positions, velocities, accelerations),
            end-effector poses (xpos_list), and success status.
        """
        # Resolve action
        if control_part:
            action_key = f"{action_name}_{control_part}"
        else:
            action_key = action_name

        if action_key not in self._actions:
            raise ValueError(f"Unknown action: {action_key}")

        action = self._actions[action_key]

        # Resolve target to ObjectSemantics if string label provided
        if isinstance(target, str):
            target = self._semantic_analyzer.analyze(target)

        # Execute action - returns PlanResult directly
        return action.execute(target, **kwargs)

    def validate(
        self,
        action_name: str,
        target: Union[torch.Tensor, str, ObjectSemantics],
        control_part: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Validate if an action is feasible without executing."""
        if control_part:
            action_key = f"{action_name}_{control_part}"
        else:
            action_key = action_name

        if action_key not in self._actions:
            return False

        action = self._actions[action_key]

        if isinstance(target, str):
            target = self._semantic_analyzer.analyze(target)

        return action.validate(target, **kwargs)

    def get_semantic_analyzer(self) -> SemanticAnalyzer:
        """Get the semantic analyzer for object understanding."""
        return self._semantic_analyzer

    def set_semantic_analyzer(self, analyzer: SemanticAnalyzer) -> None:
        """Set a custom semantic analyzer."""
        self._semantic_analyzer = analyzer
