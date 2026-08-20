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

from pathlib import Path

from embodichain.gen_sim.scene_engine.llms.openai_compatible_client import (
    OpenAICompatibleVLM,
)
from embodichain.gen_sim.scene_engine.pipeline.api import (
    analyze_edit,
    materialize_edit,
)


def edit_scene(
    *,
    output_root: str | Path,
    edit_prompt: str,
) -> None:
    """Apply one text edit instruction to an existing Scene Engine output."""
    resolved_output_root = Path(output_root).expanduser().resolve()
    vlm_client = OpenAICompatibleVLM.from_dotenv()
    blueprint = analyze_edit(
        output_root=resolved_output_root,
        edit_prompt=edit_prompt,
        vlm_client=vlm_client,
    )
    materialize_edit(blueprint, vlm_client=vlm_client)
    return None
