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

import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Protocol, Union, Dict, Any, Tuple
from pathlib import Path

try:
    import open3d as o3d

    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from embodichain.utils import logger
from embodichain.lab.sim.utility.workspace_analyzer.configs.visualization_config import (
    VisualizationConfig,
)


__all__ = [
    "IVisualizer",
    "BaseVisualizer",
]


class IVisualizer(Protocol):
    """Interface for all visualizers.

    This protocol defines the contract that all visualizers must follow.
    """

    def visualize(
        self,
        points: torch.Tensor | np.ndarray,
        colors: torch.Tensor | np.ndarray | None = None,
        **kwargs: Any,
    ) -> Any:
        """Visualize the workspace data.

        Args:
            points: Array of shape (N, 3) containing point positions.
            colors: Optional array of shape (N, 3) or (N, 4) containing colors.
            **kwargs: Additional visualization parameters.

        Returns:
            Visualization object (e.g., Open3D geometry, matplotlib figure).
        """
        ...

    def save(self, filepath: Union[str, Path], **kwargs: Any) -> None:
        """Save visualization to file.

        Args:
            filepath: Path to save the visualization.
            **kwargs: Additional save parameters.
        """
        ...

    def get_type_name(self) -> str:
        """Get the name of the visualization type.

        Returns:
            String identifier for the visualization type.
        """
        ...


class BaseVisualizer(ABC):
    """Abstract base class for all visualizers.

    This class provides common functionality and enforces the implementation
    of the visualization method in all derived classes.

    Attributes:
        backend: Visualization backend ('open3d', 'matplotlib', etc.).
        config: Configuration dictionary for visualization parameters.
    """

    def __init__(self, backend: str = "open3d", config: Dict[str, Any] | None = None):
        """Initialize the base visualizer.

        Args:
            backend: Visualization backend to use. Defaults to "open3d".
            config: Optional configuration dictionary. Defaults to None.
        """
        self.backend = backend
        self.config = config or {}
        self._validate_backend()

        # Store last visualization for reuse
        self._last_visualization = None

    @classmethod
    def from_config(cls, config: VisualizationConfig, backend: str = "open3d"):
        """Create a visualizer instance from a VisualizationConfig.

        Args:
            config: VisualizationConfig instance with visualization settings.
            backend: Visualization backend to use.

        Returns:
            Configured visualizer instance.
        """
        config_dict = {
            "enabled": config.enabled,
            "voxel_size": config.voxel_size,
            "nb_neighbors": config.nb_neighbors,
            "std_ratio": config.std_ratio,
            "is_voxel_down": config.is_voxel_down,
            "color_by_distance": config.color_by_distance,
        }
        return cls(backend=backend, config=config_dict)

    def _validate_backend(self) -> None:
        """Validate the visualization backend is available.

        Raises:
            ImportError: If the required backend is not available.
        """
        if self.backend == "data":
            # Data backend doesn't require any external dependencies
            return
        elif self.backend == "open3d" and not OPEN3D_AVAILABLE:
            raise ImportError(
                "Open3D is not available. Install it with: pip install open3d"
            )
        elif self.backend == "matplotlib" and not MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "Matplotlib is not available. Install it with: pip install matplotlib"
            )

    @abstractmethod
    def visualize(
        self,
        points: torch.Tensor | np.ndarray,
        colors: torch.Tensor | np.ndarray | None = None,
        **kwargs: Any,
    ) -> Any:
        """Visualize the workspace data.

        This method must be implemented by all derived classes.

        Args:
            points: Array of shape (N, 3) containing point positions.
            colors: Optional array of shape (N, 3) or (N, 4) containing colors.
            **kwargs: Additional visualization parameters.

        Returns:
            Visualization object.

        Raises:
            NotImplementedError: If the method is not implemented in the derived class.
        """
        raise NotImplementedError("Subclasses must implement the visualize method")

    @abstractmethod
    def get_type_name(self) -> str:
        """Get the name of the visualization type.

        Returns:
            String identifier for the visualization type.
        """
        raise NotImplementedError("Subclasses must implement get_type_name method")

    def save(self, filepath: Union[str, Path], **kwargs: Any) -> None:
        """Save the last visualization to file.

        Args:
            filepath: Path to save the visualization.
            **kwargs: Additional save parameters.

        Raises:
            RuntimeError: If no visualization has been created yet.
        """
        if self._last_visualization is None:
            raise RuntimeError("No visualization available. Call visualize() first.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        self._save_impl(filepath, **kwargs)
        logger.log_info(f"Visualization saved to {filepath}")

    @abstractmethod
    def _save_impl(self, filepath: Path, **kwargs: Any) -> None:
        """Implementation of save functionality.

        Args:
            filepath: Path to save the visualization.
            **kwargs: Additional save parameters.
        """
        raise NotImplementedError("Subclasses must implement _save_impl method")

    def _to_numpy(self, data: torch.Tensor | np.ndarray) -> np.ndarray:
        """Convert data to numpy array.

        Args:
            data: Input data (torch tensor or numpy array).

        Returns:
            NumPy array.
        """
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return np.asarray(data)

    def _validate_points(self, points: np.ndarray) -> None:
        """Validate the points array.

        Args:
            points: Array of shape (N, 3) containing point positions.

        Raises:
            ValueError: If points array is invalid.
        """
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Points must have shape (N, 3), got {points.shape}")

        if len(points) == 0:
            raise ValueError("Points array is empty")

    def _validate_colors(
        self, colors: torch.Tensor | np.ndarray | None, num_points: int
    ) -> Union[np.ndarray, None]:
        """Validate and normalize the colors array.

        Args:
            colors: Optional array of shape (N, 3) or (N, 4) containing colors.
            num_points: Number of points to validate against.

        Returns:
            Validated colors array or None.

        Raises:
            ValueError: If colors array is invalid.
        """
        if colors is None:
            return None

        colors = self._to_numpy(colors)

        if colors.ndim != 2 or colors.shape[0] != num_points:
            raise ValueError(
                f"Colors must have shape ({num_points}, 3) or ({num_points}, 4), "
                f"got {colors.shape}"
            )

        if colors.shape[1] not in [3, 4]:
            raise ValueError(
                f"Colors must have 3 (RGB) or 4 (RGBA) channels, "
                f"got {colors.shape[1]}"
            )

        # Normalize colors to [0, 1] if needed
        if colors.max() > 1.0:
            colors = colors / 255.0

        return colors

    def _get_default_colors(
        self, num_points: int, color: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    ) -> np.ndarray:
        """Generate default colors for points.

        Args:
            num_points: Number of points.
            color: RGB color tuple. Defaults to green (0.0, 1.0, 0.0).

        Returns:
            Array of shape (num_points, 3) with default colors.
        """
        return np.tile(np.array(color), (num_points, 1))

    def show(self, **kwargs: Any) -> None:
        """Display the visualization interactively.

        Args:
            **kwargs: Backend-specific display parameters.

        Raises:
            RuntimeError: If no visualization has been created yet.
        """
        if self._last_visualization is None:
            raise RuntimeError("No visualization available. Call visualize() first.")

        self._show_impl(**kwargs)

    @abstractmethod
    def _show_impl(self, **kwargs: Any) -> None:
        """Implementation of show functionality.

        Args:
            **kwargs: Backend-specific display parameters.
        """
        raise NotImplementedError("Subclasses must implement _show_impl method")

    def clear(self) -> None:
        """Clear the current visualization."""
        self._last_visualization = None

    def __repr__(self) -> str:
        """String representation of the visualizer."""
        return (
            f"{self.__class__.__name__}("
            f"type={self.get_type_name()}, "
            f"backend={self.backend})"
        )
