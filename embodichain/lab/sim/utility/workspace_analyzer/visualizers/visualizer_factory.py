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
from typing import Dict, Type, Any
from threading import Lock

from embodichain.lab.sim.utility.workspace_analyzer.configs import (
    VisualizationType,
)

from embodichain.lab.sim.utility.workspace_analyzer.visualizers.base_visualizer import (
    BaseVisualizer,
)

from embodichain.lab.sim.utility.workspace_analyzer.visualizers.point_cloud_visualizer import (
    PointCloudVisualizer,
)

from embodichain.lab.sim.utility.workspace_analyzer.visualizers.voxel_visualizer import (
    VoxelVisualizer,
)

from embodichain.lab.sim.utility.workspace_analyzer.visualizers.sphere_visualizer import (
    SphereVisualizer,
)

from embodichain.lab.sim.utility.workspace_analyzer.visualizers.axis_visualizer import (
    AxisVisualizer,
)

from embodichain.utils import logger

__all__ = [
    "VisualizerFactory",
    "create_visualizer",
]


class VisualizerFactory:
    """Factory class for creating visualizers (Singleton pattern).

    This factory allows registration and creation of visualizers based on
    the visualization type. It uses the singleton pattern to ensure only
    one instance exists throughout the application.

    The factory comes pre-registered with built-in visualizers:
        - POINT_CLOUD: PointCloudVisualizer
        - VOXEL: VoxelVisualizer
        - SPHERE: SphereVisualizer

    Additional visualizers can be registered using register_visualizer().

    Examples:
        >>> factory = VisualizerFactory()
        >>> visualizer = factory.create_visualizer(
        ...     VisualizationType.POINT_CLOUD,
        ...     backend='open3d'
        ... )
        >>> isinstance(visualizer, PointCloudVisualizer)
        True

        >>> # Register custom visualizer
        >>> factory.register_visualizer("custom", CustomVisualizer)
        >>> custom_viz = factory.create_visualizer("custom")
    """

    _instance: VisualizerFactory | None = None
    _lock: Lock = Lock()

    def __new__(cls):
        """Create or return the singleton instance.

        Returns:
            The singleton VisualizerFactory instance.
        """
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking
                if cls._instance is None:
                    cls._instance = super(VisualizerFactory, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the factory with built-in visualizers.

        This method only runs once due to the singleton pattern.
        """
        # Prevent re-initialization
        if self._initialized:
            return

        self._visualizers: Dict[str, Type[BaseVisualizer]] = {}
        self._register_builtin_visualizers()
        self._initialized = True

    def _register_builtin_visualizers(self) -> None:
        """Register the built-in visualizers."""
        self._visualizers[VisualizationType.POINT_CLOUD.value] = PointCloudVisualizer
        self._visualizers[VisualizationType.VOXEL.value] = VoxelVisualizer
        self._visualizers[VisualizationType.SPHERE.value] = SphereVisualizer
        self._visualizers[VisualizationType.AXIS.value] = AxisVisualizer

        logger.log_debug(
            f"Registered built-in visualizers: {list(self._visualizers.keys())}"
        )

    def register_visualizer(
        self, name: str, visualizer_class: Type[BaseVisualizer]
    ) -> None:
        """Register a new visualizer class.

        Args:
            name: String identifier for the visualizer type.
            visualizer_class: The visualizer class to register.
                Must inherit from BaseVisualizer.

        Raises:
            TypeError: If visualizer_class is not a subclass of BaseVisualizer.

        Examples:
            >>> factory = VisualizerFactory()
            >>> factory.register_visualizer("my_viz", MyVisualizerClass)
        """
        if not issubclass(visualizer_class, BaseVisualizer):
            raise TypeError(
                f"visualizer_class must be a subclass of BaseVisualizer, "
                f"got {visualizer_class}"
            )

        if name in self._visualizers:
            logger.log_warning(
                f"Visualizer '{name}' already registered. "
                f"Overwriting with {visualizer_class.__name__}."
            )

        self._visualizers[name] = visualizer_class
        logger.log_info(f"Registered visualizer '{name}': {visualizer_class.__name__}")

    def create_visualizer(
        self, viz_type: VisualizationType | str | None = None, **kwargs: Any
    ) -> BaseVisualizer:
        """Create a visualizer instance based on the type.

        Args:
            viz_type: The visualization type to use. Can be a VisualizationType enum
                or a string identifier. If None, defaults to POINT_CLOUD.
            **kwargs: Additional keyword arguments to pass to the visualizer constructor.
                Common arguments include:
                    - backend: Visualization backend ('open3d', 'matplotlib', 'data')
                              'data' backend returns processed data without visualization
                    - voxel_size: For VoxelVisualizer
                    - sphere_radius: For SphereVisualizer
                    - point_size: For PointCloudVisualizer

        Returns:
            An instance of the requested visualizer.

        Raises:
            ValueError: If the visualization type is not registered.

        Examples:
            >>> factory = VisualizerFactory()
            >>> viz = factory.create_visualizer(
            ...     VisualizationType.POINT_CLOUD,
            ...     backend='open3d'
            ... )
            >>> viz = factory.create_visualizer("voxel", voxel_size=0.02)
            >>> viz = factory.create_visualizer()  # Uses default (POINT_CLOUD)
        """
        # Default to POINT_CLOUD if no type specified
        if viz_type is None:
            viz_type = VisualizationType.POINT_CLOUD

        # Convert enum to string if necessary
        if isinstance(viz_type, VisualizationType):
            type_name = viz_type.value
        else:
            type_name = viz_type

        # Check if type is registered
        if type_name not in self._visualizers:
            available = list(self._visualizers.keys())
            raise ValueError(
                f"Unknown visualization type: '{type_name}'. "
                f"Available types: {available}. "
                f"You can register a custom visualizer using register_visualizer()."
            )

        # Create and return visualizer instance
        visualizer_class = self._visualizers[type_name]
        visualizer = visualizer_class(**kwargs)

        logger.log_info(
            f"Created visualizer: {visualizer_class.__name__} with kwargs: {kwargs}"
        )

        return visualizer

    def list_available_types(self) -> list[str]:
        """List all registered visualization types.

        Returns:
            List of registered type names.
        """
        return list(self._visualizers.keys())

    def is_registered(self, viz_type: VisualizationType | str) -> bool:
        """Check if a visualization type is registered.

        Args:
            viz_type: The visualization type to check.

        Returns:
            True if the type is registered, False otherwise.
        """
        if isinstance(viz_type, VisualizationType):
            type_name = viz_type.value
        else:
            type_name = viz_type

        return type_name in self._visualizers

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (mainly for testing).

        Warning:
            This should only be used in testing scenarios.
        """
        with cls._lock:
            cls._instance = None

    def __repr__(self) -> str:
        """String representation of the factory."""
        types = self.list_available_types()
        return f"VisualizerFactory(types={types})"


# Convenience function for creating visualizers
def create_visualizer(
    viz_type: VisualizationType | str | None = None, **kwargs: Any
) -> BaseVisualizer:
    """Convenience function to create a visualizer.

    This is a shorthand for VisualizerFactory().create_visualizer().

    Args:
        viz_type: The visualization type to use.
        **kwargs: Additional keyword arguments to pass to the visualizer constructor.

    Returns:
        An instance of the requested visualizer.

    Examples:
        >>> viz = create_visualizer(VisualizationType.POINT_CLOUD, backend='open3d')
        >>> viz = create_visualizer("voxel", voxel_size=0.02)
    """
    factory = VisualizerFactory()
    return factory.create_visualizer(viz_type, **kwargs)
