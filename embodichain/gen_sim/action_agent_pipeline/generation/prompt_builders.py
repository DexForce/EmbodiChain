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

"""Stable facade for generation config, graph, and diagnostic builders.

The implementation is intentionally split by responsibility. Existing callers
continue importing from this module, while deterministic plan construction and
human-readable diagnostics evolve independently behind the facade.
"""

from __future__ import annotations

from typing import Any

from embodichain.gen_sim.action_agent_pipeline.protocol.artifacts import (
    ATOM_ACTIONS_FILENAME,
    BASIC_BACKGROUND_FILENAME,
    SEED_TASK_GRAPH_FILENAME,
    TASK_GRAPH_FILENAME,
    TASK_PROMPT_FILENAME,
)
from embodichain.gen_sim.action_agent_pipeline.generation.arrangement_diagnostics import (
    make_arrangement_atom_actions_prompt,
    make_arrangement_basic_background,
    make_arrangement_task_prompt,
)
from embodichain.gen_sim.action_agent_pipeline.generation.relative_action_diagnostics import (
    make_relative_atom_actions_prompt,
)
from embodichain.gen_sim.action_agent_pipeline.generation.relative_background_diagnostics import (
    make_relative_basic_background,
)
from embodichain.gen_sim.action_agent_pipeline.generation.relative_task_diagnostics import (
    make_relative_task_prompt,
)
from embodichain.gen_sim.action_agent_pipeline.generation.stacking_diagnostics import (
    make_stacking_atom_actions_prompt,
    make_stacking_basic_background,
    make_stacking_task_prompt,
)
from embodichain.gen_sim.action_agent_pipeline.generation.task_graph_builders import (
    make_arrangement_task_graph,
    make_relative_task_graph,
    make_stacking_task_graph,
)

__all__ = [
    "make_agent_config",
    "make_arrangement_task_graph",
    "make_arrangement_atom_actions_prompt",
    "make_arrangement_basic_background",
    "make_arrangement_task_prompt",
    "make_relative_task_graph",
    "make_relative_atom_actions_prompt",
    "make_relative_basic_background",
    "make_relative_task_prompt",
    "make_stacking_task_graph",
    "make_stacking_atom_actions_prompt",
    "make_stacking_basic_background",
    "make_stacking_task_prompt",
]


def make_agent_config() -> dict[str, Any]:
    """Build the stable agent-config schema for deterministic graph execution.

    ``TaskAgent.precomputed_task_graph`` is the authoritative execution input,
    while ``TaskAgent.seed_task_graph`` identifies the symbolic source checked
    before compilation. ``Agent.prompt_kwargs`` is preserved so existing config
    consumers can find the accompanying diagnostic text files; the runtime does
    not interpret those files.
    """
    return {
        "TaskAgent": {
            "prompt_name": "generate_task_graph",
            "seed_task_graph": SEED_TASK_GRAPH_FILENAME,
            "precomputed_task_graph": TASK_GRAPH_FILENAME,
        },
        "CompileAgent": {},
        "Agent": {
            "prompt_kwargs": {
                "task_prompt": {
                    "type": "text",
                    "name": TASK_PROMPT_FILENAME,
                },
                "basic_background": {
                    "type": "text",
                    "name": BASIC_BACKGROUND_FILENAME,
                },
                "atom_actions": {
                    "type": "text",
                    "name": ATOM_ACTIONS_FILENAME,
                },
            }
        },
    }
