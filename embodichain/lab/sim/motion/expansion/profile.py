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

"""Safe loading for source-neutral Generation Profile files."""

from __future__ import annotations

from pathlib import Path

from embodichain.utils.utility import load_config

from .cfg import TrajectoryGenerationJobCfg

__all__ = ["load_generation_profile"]


def load_generation_profile(path: str | Path) -> TrajectoryGenerationJobCfg:
    """Load one callable-free Generation Profile through strict decoding.

    Args:
        path: YAML or JSON profile path. Repository-style task paths retain the
            shared loader's installed-package resolution behavior.

    Returns:
        Strictly decoded, semantically validated Generation Profile.

    Raises:
        TypeError: If the serialized root is not a mapping.
        ValueError: If the profile contains unknown or invalid fields.
    """
    try:
        data = load_config(path)
        return TrajectoryGenerationJobCfg.from_mapping(data)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid Generation Profile {path}: {error}") from error
