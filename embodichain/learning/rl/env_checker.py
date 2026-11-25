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

"""Offline sample checks for lightweight learning environments."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
import random
from typing import Any

from gymnasium import spaces
import numpy as np
import torch
from tensordict import TensorDictBase

from .env import LearningEnvProtocol

__all__ = ["check_learning_env"]


@contextmanager
def _preserve_random_state(seed: int) -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    )
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        # Unlike torch.manual_seed(), this does not queue CUDA initialization work.
        torch.random.default_generator.manual_seed(seed)
        if cuda_states is not None:
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _check_tensor(
    value: Any,
    *,
    shape: tuple[int, ...],
    device: torch.device,
    path: str,
    dtype: torch.dtype | None = None,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{path} must be a torch.Tensor.")
    if tuple(value.shape) != shape:
        raise ValueError(f"{path} must have shape {shape}, got {tuple(value.shape)}.")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{path} must have dtype {dtype}, got {value.dtype}.")
    if value.device.type != device.type or (
        device.index is not None and value.device.index != device.index
    ):
        raise ValueError(f"{path} must be on {device}, got {value.device}.")


def _check_observation(
    value: Any, space: spaces.Space, count: int, device: torch.device, path: str
) -> None:
    if isinstance(space, spaces.Dict):
        if not isinstance(value, (Mapping, TensorDictBase)):
            raise TypeError(f"{path} must be a mapping or TensorDict.")
        if set(value.keys()) != set(space.spaces):
            raise ValueError(f"{path} keys must match the observation space.")
        for name, child_space in space.spaces.items():
            _check_observation(
                value[name], child_space, count, device, f"{path}.{name}"
            )
    elif isinstance(space, spaces.Box):
        dtype = torch.from_numpy(np.empty((), dtype=space.dtype)).dtype
        _check_tensor(
            value, shape=(count, *space.shape), dtype=dtype, device=device, path=path
        )
    else:
        raise TypeError(f"{path} supports only Box and Dict observation spaces.")


def _check_terminal_info(
    info: Mapping[str, Any], space: spaces.Space, count: int, device: torch.device
) -> None:
    terminal = info.get("terminal_observation")
    if terminal is None:
        return
    terminal_count = count
    ids = info.get("terminal_env_ids")
    if ids is not None:
        if not isinstance(ids, torch.Tensor) or ids.ndim != 1:
            raise TypeError("info.terminal_env_ids must be a one-dimensional tensor.")
        _check_tensor(
            ids, shape=(ids.numel(),), device=device, path="info.terminal_env_ids"
        )
        if ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("info.terminal_env_ids must contain integer indices.")
        if (
            bool(((ids < 0) | (ids >= count)).any())
            or ids.unique().numel() != ids.numel()
        ):
            raise ValueError(
                "info.terminal_env_ids must be distinct valid environment indices."
            )
        terminal_count = ids.numel()
    mask = info.get("terminal_observation_mask")
    if mask is not None:
        _check_tensor(
            mask,
            shape=(count,),
            dtype=torch.bool,
            device=device,
            path="info.terminal_observation_mask",
        )
    _check_observation(
        terminal, space, terminal_count, device, "info.terminal_observation"
    )


def check_learning_env(
    factory: Callable[[], LearningEnvProtocol], *, num_steps: int = 3, seed: int = 0
) -> None:
    """Check finite samples from an independently constructed learning environment.

    Supports Box actions and Box or nested Dict observations represented by tensors,
    mappings or TensorDicts. Checks batch shapes, declared observation dtypes,
    devices, boolean termination flags and any supplied terminal observations.
    Terminal observations may be full batches or indexed by ``terminal_env_ids``.
    Missing terminal data is allowed: this does not prove value-bootstrap support,
    autograd correctness, space bounds, or boundaries not encountered in the sample.

    The created environment is closed on success and failure. Python, NumPy and
    Torch CPU global RNG states are restored, along with CUDA states if CUDA was
    already initialized. This checker does not initialize CUDA itself; a factory
    that initializes CUDA for the first time creates states that cannot be restored
    to a pre-existing state. Run offline, without concurrent use of these RNGs.

    Args:
        factory: Zero-argument callable creating a fresh, disposable environment.
        num_steps: Positive number of steps to sample after one reset.
        seed: Non-negative seed for construction, reset and action sampling.

    Raises:
        TypeError: An argument, tensor dtype, or supported space contract is invalid.
        ValueError: A sampled shape, device, or terminal index is invalid.
    """
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
        raise ValueError("num_steps must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an integer in [0, 2**64).")
    with _preserve_random_state(seed):
        env = factory()
        try:
            count = env.num_envs
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError("env.num_envs must be a positive integer.")
            device = torch.device(env.device)
            action_space = env.single_action_space
            if not isinstance(action_space, spaces.Box):
                raise TypeError("check_learning_env supports only Box action spaces.")
            observation, info = env.reset(seed=seed)
            if not isinstance(info, Mapping):
                raise TypeError("reset info must be a mapping.")
            if device.type == "cpu":
                device = torch.device("cpu")
            elif (
                device.type == "cuda"
                and device.index is None
                and torch.cuda.is_initialized()
            ):
                device = torch.device("cuda", torch.cuda.current_device())
            space = env.single_observation_space
            _check_observation(observation, space, count, device, "reset.observation")
            # Spaces can be shared between otherwise independent environments.
            action_space = deepcopy(action_space)
            action_space.seed(seed)
            for _ in range(num_steps):
                action = torch.as_tensor(
                    np.stack([action_space.sample() for _ in range(count)]),
                    device=device,
                )
                observation, reward, terminated, truncated, info = env.step(action)
                _check_observation(
                    observation, space, count, device, "step.observation"
                )
                _check_tensor(reward, shape=(count,), device=device, path="reward")
                if reward.dtype == torch.bool or reward.is_complex():
                    raise TypeError("reward must have a real numeric dtype.")
                for name, flags in (
                    ("terminated", terminated),
                    ("truncated", truncated),
                ):
                    _check_tensor(
                        flags,
                        shape=(count,),
                        dtype=torch.bool,
                        device=device,
                        path=name,
                    )
                if not isinstance(info, Mapping):
                    raise TypeError("step info must be a mapping.")
                _check_terminal_info(info, space, count, device)
        except BaseException as error:
            try:
                env.close()
            except Exception as close_error:
                error.add_note(f"Environment close also failed: {close_error}")
            raise
        else:
            env.close()
