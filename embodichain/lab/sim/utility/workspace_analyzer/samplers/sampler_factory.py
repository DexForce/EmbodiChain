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
from typing import Dict, Type, Any, Union
from threading import Lock

import torch
import numpy as np

from embodichain.lab.sim.utility.workspace_analyzer.configs.sampling_config import (
    SamplingStrategy,
)
from embodichain.lab.sim.utility.workspace_analyzer.samplers.base_sampler import (
    BaseSampler,
)
from embodichain.lab.sim.utility.workspace_analyzer.samplers.iniform_sampler import (
    UniformSampler,
)
from embodichain.lab.sim.utility.workspace_analyzer.samplers.random_sampler import (
    RandomSampler,
)
from embodichain.lab.sim.utility.workspace_analyzer.samplers.halton_sampler import (
    HaltonSampler,
)
from embodichain.lab.sim.utility.workspace_analyzer.samplers.sobol_sampler import (
    SobolSampler,
)
from embodichain.lab.sim.utility.workspace_analyzer.samplers.lhs_sampler import (
    LatinHypercubeSampler,
)
from embodichain.lab.sim.utility.workspace_analyzer.samplers.importance_sampler import (
    ImportanceSampler,
)
from embodichain.lab.sim.utility.workspace_analyzer.samplers.gaussian_sampler import (
    GaussianSampler,
)

# Note: Constraint imports temporarily disabled

from embodichain.utils import logger


class SamplerFactory:
    """Factory class for creating samplers (Singleton pattern).

    This factory allows registration and creation of samplers based on
    the sampling strategy. It uses the singleton pattern to ensure only
    one instance exists throughout the application.

    The factory comes pre-registered with built-in samplers:
        - UNIFORM: UniformSampler
        - RANDOM: RandomSampler

    Additional samplers can be registered using register_sampler().

    Examples:
        >>> factory = SamplerFactory()
        >>> sampler = factory.create_sampler(SamplingStrategy.UNIFORM, seed=42)
        >>> isinstance(sampler, UniformSampler)
        True

        >>> # Register custom sampler
        >>> factory.register_sampler("custom", CustomSampler)
        >>> custom_sampler = factory.create_sampler("custom", seed=42)
    """

    _instance: SamplerFactory | None = None
    _lock: Lock = Lock()

    def __new__(cls):
        """Create or return the singleton instance.

        Returns:
            The singleton SamplerFactory instance.
        """
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking
                if cls._instance is None:
                    cls._instance = super(SamplerFactory, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the factory with built-in samplers.

        This method only runs once due to the singleton pattern.
        """
        # Prevent re-initialization
        if self._initialized:
            return

        self._samplers: Dict[str, Type[BaseSampler]] = {}
        self._register_builtin_samplers()
        self._initialized = True

    def _register_builtin_samplers(self) -> None:
        """Register the built-in samplers."""
        self._samplers[SamplingStrategy.UNIFORM.value] = UniformSampler
        self._samplers[SamplingStrategy.RANDOM.value] = RandomSampler
        self._samplers[SamplingStrategy.HALTON.value] = HaltonSampler
        self._samplers[SamplingStrategy.SOBOL.value] = SobolSampler
        self._samplers[SamplingStrategy.LATIN_HYPERCUBE.value] = LatinHypercubeSampler
        self._samplers[SamplingStrategy.IMPORTANCE.value] = ImportanceSampler
        self._samplers[SamplingStrategy.GAUSSIAN.value] = GaussianSampler
        self._samplers[SamplingStrategy.SPHERE.value] = UniformSampler

        logger.log_debug(f"Registered built-in samplers: {list(self._samplers.keys())}")

    def register_sampler(self, name: str, sampler_class: Type[BaseSampler]) -> None:
        """Register a new sampler class.

        Args:
            name: String identifier for the sampler strategy.
            sampler_class: The sampler class to register. Must inherit from BaseSampler.

        Raises:
            TypeError: If sampler_class is not a subclass of BaseSampler.
            ValueError: If name already exists and overwrite=False.

        Examples:
            >>> factory = SamplerFactory()
            >>> factory.register_sampler("my_sampler", MySamplerClass)
        """
        if not issubclass(sampler_class, BaseSampler):
            raise TypeError(
                f"sampler_class must be a subclass of BaseSampler, "
                f"got {sampler_class}"
            )

        if name in self._samplers:
            logger.log_warning(
                f"Sampler '{name}' already registered. Overwriting with {sampler_class.__name__}."
            )

        self._samplers[name] = sampler_class
        logger.log_info(f"Registered sampler '{name}': {sampler_class.__name__}")

    def create_sampler(
        self, strategy: SamplingStrategy | str | None = None, **kwargs: Any
    ) -> BaseSampler:
        """Create a sampler instance based on the strategy.

        Args:
            strategy: The sampling strategy to use. Can be a SamplingStrategy enum
                or a string identifier. If None, defaults to RANDOM.
            **kwargs: Additional keyword arguments to pass to the sampler constructor.
                Common arguments include:
                    - seed: Random seed for reproducibility
                    - samples_per_dim: For UniformSampler
                    - device: PyTorch device for tensor operations
                Note: constraint parameter is temporarily disabled

        Returns:
            An instance of the requested sampler.

        Raises:
            ValueError: If the strategy is not registered.

        Examples:
            >>> factory = SamplerFactory()
            >>> # Bounds-based usage
            >>> sampler = factory.create_sampler(SamplingStrategy.UNIFORM, seed=42)
            >>> sampler = factory.create_sampler("random", seed=123)

            Note: Constraint-based sampling examples are temporarily disabled
        """
        # Default to RANDOM if no strategy specified
        if strategy is None:
            strategy = SamplingStrategy.RANDOM

        # Convert enum to string if necessary
        if isinstance(strategy, SamplingStrategy):
            strategy_name = strategy.value
        else:
            strategy_name = strategy

        # Check if strategy is registered
        if strategy_name not in self._samplers:
            available = list(self._samplers.keys())
            raise ValueError(
                f"Unknown sampling strategy: '{strategy_name}'. "
                f"Available strategies: {available}. "
                f"You can register a custom sampler using register_sampler()."
            )

        # Create and return sampler instance
        sampler_class = self._samplers[strategy_name]
        sampler = sampler_class(**kwargs)

        logger.log_info(
            f"Created sampler: {sampler_class.__name__} with strategy '{strategy_name}'"
        )

        return sampler

    def list_available_strategies(self) -> list[str]:
        """List all registered sampling strategies.

        Returns:
            List of registered strategy names.
        """
        return list(self._samplers.keys())

    def is_registered(self, strategy: SamplingStrategy | str) -> bool:
        """Check if a strategy is registered.

        Args:
            strategy: The sampling strategy to check.

        Returns:
            True if the strategy is registered, False otherwise.
        """
        if isinstance(strategy, SamplingStrategy):
            strategy_name = strategy.value
        else:
            strategy_name = strategy

        return strategy_name in self._samplers

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
        strategies = self.list_available_strategies()
        return f"SamplerFactory(strategies={strategies})"


# Convenience function for creating samplers
def create_sampler(
    strategy: SamplingStrategy | str | None = None, **kwargs: Any
) -> BaseSampler:
    """Convenience function to create a sampler.

    This is a shorthand for SamplerFactory().create_sampler().

    Args:
        strategy: The sampling strategy to use.
        **kwargs: Additional keyword arguments to pass to the sampler constructor.

    Returns:
        An instance of the requested sampler.

    Examples:
        >>> sampler = create_sampler(SamplingStrategy.UNIFORM, seed=42)
        >>> sampler = create_sampler("random", seed=123)

        Note: Constraint-based sampling is temporarily disabled
    """
    factory = SamplerFactory()
    return factory.create_sampler(strategy, **kwargs)


# Note: Constraint-based convenience functions have been removed
