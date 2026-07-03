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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["SceneState", "Tempo_SceneState"]


@dataclass
class SceneState:
    raw_input_json: Dict = field(default_factory=dict)
    table_size: Optional[Tuple[float, float]] = None
    table_semantics: str = ""
    coordinate_system: Dict = field(default_factory=dict)
    raw_object_dict: Dict[str, Dict] = field(default_factory=dict)
    init_layout: Dict[str, Dict] = field(default_factory=dict)
    optimization_model: Dict[str, Any] = field(default_factory=dict)
    optimized_layout: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    stack_groups: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    asset_specs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)


@dataclass
class Tempo_SceneState:
    raw_input_json: Dict = field(default_factory=dict)
    table_size: Optional[Tuple[float, float]] = None
    table_semantics: str = ""
    coordinate_system: Dict = field(default_factory=dict)
    raw_object_dict: Dict[str, Dict] = field(default_factory=dict)
    filtered_objects_info: Dict[str, Dict] = field(default_factory=dict)
    filtered_objects: Dict[str, Dict] = field(default_factory=dict)
    init_layout: Dict[str, Dict] = field(default_factory=dict)
    optimization_model: Dict[str, Any] = field(default_factory=dict)
    optimized_layout: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    stack_groups: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    final_layout: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)
