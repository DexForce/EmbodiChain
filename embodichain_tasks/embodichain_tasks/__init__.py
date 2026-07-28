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
via recursive sub-package import. Each task sub-package's ``__init__.py``
calls ``@register_env`` which registers the environment in gymnasium's
global registry.
"""

from __future__ import annotations

from .utils.importer import import_packages

_BLACKLIST = ["utils"]

import_packages(__name__, _BLACKLIST)
