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


def test_prompt2scene_edit_only_returns_existing_gym_config(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from embodichain.gen_sim.prompt2scene.pipeline import runner
    from embodichain.gen_sim.prompt2scene.workflows.request import (
        InputKind,
        Prompt2SceneInput,
    )

    output_root = tmp_path / "prompt2scene"
    gym_config = output_root / "gym_export/gym_config.json"
    scene_state = output_root / "gym_export/scene_state/result.json"
    scene_state.parent.mkdir(parents=True)
    scene_state.write_text("{}", encoding="utf-8")
    gym_config.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        runner,
        "_route_existing_scene_prompt",
        lambda *, request, llm_cfg: {"route": {"route": "scene_randomization"}},
    )
    monkeypatch.setattr(
        runner,
        "_run_routed_existing_scene_workflow",
        lambda *, request, route_result, llm_cfg: output_root
        / "scene_randomization/result.json",
    )

    result = runner.run_prompt2scene(
        Prompt2SceneInput(
            input_kind=InputKind.EDIT,
            output_root=output_root,
            prompt="move the can left",
        ),
        llm_cfg=None,
    )

    assert result.gym_config_path == gym_config.resolve()
