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

"""Official task environments for EmbodiChain.

Importing this package triggers auto-registration of all task environments
via recursive module import. Each task-named module executes its
``@register_env`` decorator, registering the environment in gymnasium's global
registry.
"""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

# Source configs are a sibling of this package; wheels place them inside it.
# Bind the sibling before task auto-registration can consult an old editable map.
_source_root = Path(__file__).resolve().parents[1]
_config_init = _source_root / "configs/__init__.py"
if _source_root.name == __name__ and _config_init.is_file():
    _config_spec = spec_from_file_location(f"{__name__}.configs", _config_init)
    if _config_spec is None or _config_spec.loader is None:
        raise ImportError(f"Cannot load source task configs: {_config_init}")
    configs = module_from_spec(_config_spec)
    sys.modules[_config_spec.name] = configs
    _config_spec.loader.exec_module(configs)

from embodichain.lab.gym.utils.gym_utils import register_manager_modules

from .utils.importer import import_packages

__all__: list[str] = []

_BLACKLIST = ["utils"]

register_manager_modules(
    [
        "embodichain_tasks.locomotion.managers.observations",
        "embodichain_tasks.locomotion.managers.rewards",
    ]
)

import_packages(__name__, _BLACKLIST)
