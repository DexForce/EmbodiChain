import numpy as np

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any
from scipy.spatial.transform import Rotation as R

from embodichain.lab.gym.envs import BaseEnv
from embodichain.utils import logger


class Action(ABC):
    r"""Base class for action terms.

    The action term is responsible for processing the raw actions sent to the environment
    and applying them to the asset managed by the term. The action term is comprised of two
    operations:

    """

    env = None
    scene = None

    def __init__(self, env, **kwargs) -> None:
        self.env: BaseEnv = env
        self.scene = self.env.scene

    # def reset(self, env_ids: Sequence[int] | None = None) -> None:
    #     r"""Resets the manager term.

    #     Args:
    #         env_ids: The environment ids. Defaults to None, in which case
    #             all environments are considered.
    #     """
    #     pass

    # @abstractmethod
    # def process_actions(self, actions: torch.Tensor):
    #     r"""Processes the actions sent to the environment.

    #     Note:
    #         This function is called once per environment step by the manager.

    #     Args:
    #         actions: The actions to process.
    #     """
    #     raise NotImplementedError

    # @abstractmethod
    # def apply_actions(self):
    #     r"""Applies the actions to the asset managed by the term.

    #     Note:
    #         This is called at every simulation step by the manager.
    #     """
    #     raise NotImplementedError

    def __call__(self, *args) -> Any:
        """Returns the value of the term required by the manager.

        In case of a class implementation, this function is called by the manager
        to get the value of the term. The arguments passed to this function are
        the ones specified in the term configuration (see :attr:`ManagerTermBaseCfg.params`).

        .. attention::
            To be consistent with memory-less implementation of terms with functions, it is
            recommended to ensure that the returned mutable quantities are cloned before
            returning them. For instance, if the term returns a tensor, it is recommended
            to ensure that the returned tensor is a clone of the original tensor. This prevents
            the manager from storing references to the tensors and altering the original tensors.

        Args:
            *args: Variable length argument list.

        Returns:
            The value of the term.
        """
        raise NotImplementedError
