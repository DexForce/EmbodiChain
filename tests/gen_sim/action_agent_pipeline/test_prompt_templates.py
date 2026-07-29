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

import pytest

from embodichain.gen_sim.action_agent_pipeline.prompts.template_loader import (
    render_prompt_template,
)

_TEMPLATE_NAMES = (
    "task_interpretation.txt",
    "task_router.txt",
    "relative_placement_spec.txt",
    "object_manipulation_spec.txt",
    "arrangement_spec.txt",
    "stacking_spec.txt",
)


@pytest.mark.parametrize("template_name", _TEMPLATE_NAMES)
def test_config_generation_prompt_templates_render_strictly(template_name: str) -> None:
    task_description = "将罐头摆成一排"

    prompt = render_prompt_template(
        template_name,
        project_name="task4_2",
        task_description=task_description,
        scene_summary='[{"source_uid": "can_1"}]',
    )

    assert task_description in prompt
    assert "can_1" in prompt
    assert "$project_name" not in prompt


def test_prompt_template_loader_rejects_directory_escape() -> None:
    with pytest.raises(ValueError, match="must be a file name"):
        render_prompt_template("../task_router.txt")


def test_unified_prompt_preserves_passive_stacking_anchor_rule() -> None:
    prompt = render_prompt_template(
        "task_interpretation.txt",
        project_name="task3_2",
        task_description="把纸杯叠放到爆米花桶上，把蓝色耳机叠放到爆米花桶上",
        scene_summary='[{"source_uid": "interact_popcorn_bucket"}]',
    )

    assert "belongs only in anchor.object" in prompt
    assert "must not be included in objects or bottom_to_top" in prompt
    assert "successive instructions stack A and B onto the same named root C" in prompt
