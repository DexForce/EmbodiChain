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

import torch

from embodichain.agents.mllm.prompt import task_prompt as task_prompt_module
from embodichain.agents.mllm.prompt.task_prompt import TaskPrompt


def test_generate_task_graph_formats_legacy_fn_literal(monkeypatch) -> None:
    monkeypatch.setattr(task_prompt_module, "encode_image", lambda observation: "ok")

    prompt = TaskPrompt.generate_task_graph(
        observations={"rgb": torch.zeros((2, 2, 3), dtype=torch.uint8)},
        basic_background="background",
        task_prompt="pour water",
        atom_actions="move, pick_up",
    )

    assert hasattr(prompt, "messages")
    assert len(prompt.messages) == 2
