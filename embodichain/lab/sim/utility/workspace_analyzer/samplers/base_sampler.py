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
from typing import Protocol, Union, TYPE_CHECKING

from embodichain.utils import logger


__all__ = [
    "ISampler",
    "BaseSampler",
]


class ISampler(Protocol):
    """Interface for all samplers.

    This protocol defines the contract that all samplers must follow.
    """

    def sample(
        self, bounds: Union[torch.Tensor, np.ndarray], num_samples: int
    ) -> torch.Tensor:
        """Generate samples within the given bounds.

        Args:
            bounds: Tensor/Array of shape (n_dims, 2) containing [lower, upper] bounds for each dimension.
            num_samples: Number of samples to generate.

        Returns:
            Tensor of shape (num_samples, n_dims) containing the sampled points.
        """
        ...

    def get_strategy_name(self) -> str:
        """Get the name of the sampling strategy.

        Returns:
            String identifier for the sampling strategy.
        """
        ...


class BaseSampler(ABC):
    """Abstract base class for all samplers.

    This class provides common functionality and enforces the implementation
    of the sampling method in all derived classes.

    Attributes:
        seed: Random seed for reproducibility.
        rng: NumPy random number generator.
        device: PyTorch device for tensor operations.
    """

    def __init__(
        self,
        seed: int = 42,
        device: torch.device | None = None,
    ):
        """Initialize the base sampler.

        Args:
            seed: Random seed for reproducibility. Defaults to 42.
            device: PyTorch device (cpu/cuda). Defaults to cpu.
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.device = device if device is not None else torch.device("cpu")

        # Set torch seed
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(seed)

    def sample(
        self, num_samples: int, bounds: torch.Tensor | np.ndarray | None = None
    ) -> torch.Tensor:
        """Generate samples within the given bounds.

        Args:
            num_samples: Number of samples to generate.
            bounds: Tensor/Array of shape (n_dims, 2) containing [lower, upper] bounds for each dimension.

        Returns:
            Tensor of shape (num_samples, n_dims) containing the sampled points.

        Raises:
            ValueError: If bounds are not provided.
            NotImplementedError: If the method is not implemented in the derived class.
        """
        if bounds is None:
            raise ValueError("bounds parameter is required")
        return self._sample_from_bounds(bounds, num_samples)

    @abstractmethod
    def _sample_from_bounds(
        self, bounds: Union[torch.Tensor, np.ndarray], num_samples: int
    ) -> torch.Tensor:
        """Generate samples within the given bounds.

        This method must be implemented by all derived classes.

        Args:
            bounds: Tensor/Array of shape (n_dims, 2) containing [lower, upper] bounds for each dimension.
            num_samples: Number of samples to generate.

        Returns:
            Tensor of shape (num_samples, n_dims) containing the sampled points.

        Raises:
            NotImplementedError: If the method is not implemented in the derived class.
        """
        raise NotImplementedError(
            "Subclasses must implement the _sample_from_bounds method"
        )

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of the sampling strategy.

        Returns:
            String identifier for the sampling strategy.
        """
        raise NotImplementedError("Subclasses must implement get_strategy_name method")

    def _to_tensor(self, data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Convert data to torch.Tensor.

        Args:
            data: Input data (numpy array or torch tensor).

        Returns:
            PyTorch tensor on the configured device.
        """
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        return torch.tensor(data, dtype=torch.float32, device=self.device)

    def _to_numpy(self, data: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array.

        Args:
            data: PyTorch tensor.

        Returns:
            NumPy array.
        """
        return data.detach().cpu().numpy()

    def _validate_bounds(self, bounds: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Validate the bounds array and convert to tensor.

        Args:
            bounds: Tensor/Array of shape (n_dims, 2) containing [lower, upper] bounds.

        Returns:
            Validated bounds as torch.Tensor.

        Raises:
            ValueError: If bounds are invalid.
        """
        bounds_tensor = self._to_tensor(bounds)

        if bounds_tensor.ndim != 2 or bounds_tensor.shape[1] != 2:
            raise ValueError(
                f"Bounds must have shape (n_dims, 2), got {bounds_tensor.shape}"
            )

        if torch.any(bounds_tensor[:, 0] >= bounds_tensor[:, 1]):
            raise ValueError(
                "Lower bounds must be strictly less than upper bounds. "
                f"Got bounds: {bounds_tensor}"
            )

        return bounds_tensor

    def _validate_samples(self, samples: torch.Tensor, bounds: torch.Tensor) -> None:
        """Validate that samples are within bounds.

        Args:
            samples: Tensor of shape (num_samples, n_dims) containing sampled points.
            bounds: Tensor of shape (n_dims, 2) containing [lower, upper] bounds.

        Raises:
            ValueError: If any sample is outside the bounds.
        """
        lower_bounds = bounds[:, 0]
        upper_bounds = bounds[:, 1]

        # Check if all samples are within bounds (with small tolerance for numerical errors)
        tolerance = 1e-6
        if torch.any(samples < lower_bounds - tolerance) or torch.any(
            samples > upper_bounds + tolerance
        ):
            out_of_bounds = torch.logical_or(
                samples < lower_bounds - tolerance, samples > upper_bounds + tolerance
            )
            num_violations = torch.sum(out_of_bounds).item()
            logger.log_warning(
                f"Found {num_violations} samples outside bounds. "
                "This may be due to numerical precision issues."
            )

    def _scale_samples(
        self, samples: torch.Tensor, bounds: torch.Tensor
    ) -> torch.Tensor:
        """Scale samples from [0, 1] to the given bounds.

        This is a utility method for samplers that generate samples in [0, 1]^n
        and then scale them to the desired bounds.

        Args:
            samples: Tensor of shape (num_samples, n_dims) with values in [0, 1].
            bounds: Tensor of shape (n_dims, 2) containing [lower, upper] bounds.

        Returns:
            Scaled samples within the specified bounds.
        """
        lower_bounds = bounds[:, 0]
        upper_bounds = bounds[:, 1]
        return lower_bounds + samples * (upper_bounds - lower_bounds)

    def __repr__(self) -> str:
        """String representation of the sampler."""
        return f"{self.__class__.__name__}(strategy={self.get_strategy_name()}, seed={self.seed})"
