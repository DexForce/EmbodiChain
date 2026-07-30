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

"""Build the minimal runtime manifest for one generated Seed graph."""

from __future__ import annotations

from typing import Any

from embodichain.gen_sim.action_agent_pipeline.protocol.artifacts import (
    SEED_TASK_GRAPH_FILENAME,
)

__all__ = ["make_runtime_agent_config"]


def make_runtime_agent_config() -> dict[str, Any]:
    """Return the runtime-only agent configuration.

    Human-readable prompts are review artifacts and are not runtime inputs.
    The historical prompt-bearing manifest remains available through
    ``generation.prompt_builders.make_agent_config``.
    """
    return {
        "TaskAgent": {"seed_task_graph": SEED_TASK_GRAPH_FILENAME},
        "CompileAgent": {},
        "Agent": {},
    }
