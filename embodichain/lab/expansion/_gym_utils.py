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

"""Native observation schema and identity serialization helpers."""

from __future__ import annotations
from collections.abc import Mapping
import hashlib
import json
import numpy as np
import torch

__all__: list[str] = []


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()
    ).hexdigest()


def _images(
    observation: object, row: int, prefix: tuple[str, ...] = ()
) -> dict[str, np.ndarray]:
    frames = {}
    if not hasattr(observation, "items"):
        return frames
    for key, value in observation.items():
        name = prefix + (
            (str(key),) if not isinstance(key, tuple) else tuple(map(str, key))
        )
        if isinstance(value, torch.Tensor):
            if value.ndim >= 3 and name[-1].split("_", 1)[0] in {
                "rgb",
                "depth",
                "color",
                "mask",
                "normal",
                "position",
                "disparity",
            }:
                frames["/".join(name)] = value[row].detach().cpu().numpy()
        else:
            frames.update(_images(value, row, name))
    return frames


def _observation_schema(space):
    children = getattr(space, "spaces", None)
    if children is not None:
        return {name: _observation_schema(value) for name, value in children.items()}
    if isinstance(space, Mapping):
        return {name: _observation_schema(value) for name, value in space.items()}
    return {
        "type": type(space).__name__,
        "shape": tuple(space.shape),
        "dtype": np.dtype(space.dtype).str,
    }
