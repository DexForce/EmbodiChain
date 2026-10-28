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

from importlib import import_module as _import_module
from pathlib import Path as _Path

from embodichain.lab.gym.utils.gym_utils import register_manager_modules

from .utils.importer import import_packages

__all__: list[str] = []

_BLACKLIST = ["utils"]

# Editable installs map the separate configs package to an absolute checkout.
# Prefer the resources beside the selected source package when PYTHONPATH
# selects another worktree. Restore the code-only search path before walking
# task modules, so the extension container cannot be imported recursively.
_source_root = _Path(__file__).resolve().parent.parent
_source_config = _source_root / "configs" / "__init__.py"
if _source_root.parent.name == "extensions" and _source_config.is_file():
    __path__.append(str(_source_root))
    try:
        _config_package = _import_module(f"{__name__}.configs")
        if _Path(_config_package.__file__).resolve() != _source_config.resolve():
            raise ImportError(
                "Task code and configuration resources belong to different checkouts."
            )
    finally:
        __path__.remove(str(_source_root))

register_manager_modules(
    [
        "embodichain_tasks.manipulation.assembly.obs",
        "embodichain.lab.gym.envs.managers.assembly",
        "embodichain_tasks.locomotion.managers.observations",
        "embodichain_tasks.locomotion.managers.rewards",
    ]
)

import_packages(__name__, _BLACKLIST)
