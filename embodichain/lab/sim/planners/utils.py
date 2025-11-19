# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

from enum import Enum
from typing import Union
from embodichain.utils import logger


class TrajectorySampleMethod(Enum):
    r"""Enumeration for different trajectory sampling methods.

    This enum defines various methods for sampling trajectories,
    providing meaningful names for different sampling strategies.
    """
    TIME = "time"
    """Sample based on time intervals."""

    QUANTITY = "quantity"
    """Sample based on a specified number of points."""

    DISTANCE = "distance"
    """Sample based on distance intervals."""

    @classmethod
    def from_str(
        cls, value: Union[str, "TrajectorySampleMethod"]
    ) -> "TrajectorySampleMethod":
        if isinstance(value, cls):
            return value
        try:
            return cls[value.upper()]
        except KeyError:
            valid_values = [e.name for e in cls]
            logger.log_error(
                f"Invalid version '{value}'. Valid values are: {valid_values}",
                ValueError,
            )

    def __str__(self):
        """Override string representation for better readability."""
        return self.value.capitalize()
