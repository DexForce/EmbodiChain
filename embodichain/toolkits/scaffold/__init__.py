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

"""Scaffold new EmbodiChain task environments (in-repo or external extension)."""

from __future__ import annotations

from embodichain.toolkits.scaffold.cli import main
from embodichain.toolkits.scaffold.generator import generate_task
from embodichain.toolkits.scaffold.spec import TaskSpec

__all__ = ["TaskSpec", "generate_task", "main"]
