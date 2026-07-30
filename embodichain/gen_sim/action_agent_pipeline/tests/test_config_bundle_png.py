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

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodichain.gen_sim.action_agent_pipeline.generation import config_io
from embodichain.gen_sim.action_agent_pipeline.generation.config_io import (
    raise_if_generated_files_exist,
    write_config_bundle,
)
from embodichain.gen_sim.action_agent_pipeline.generation.seed_task_graph import (
    make_relative_seed_task_graph,
)
from embodichain.gen_sim.action_agent_pipeline.graph_visualization import (
    render_seed_task_graph_png,
)

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def test_config_bundle_publishes_only_seed_graph_artifacts(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    graph_root = tmp_path / "outputs" / "graph"

    paths = write_config_bundle(
        output_dir=config_dir,
        bundle=_bundle("initial"),
        overwrite=False,
        graph_output_root=graph_root,
        graph_renderer=render_seed_task_graph_png,
    )

    assert paths.seed_task_graph.is_file()
    assert paths.seed_task_graph_png.read_bytes().startswith(_PNG_SIGNATURE)
    assert not (config_dir / "task_graph.json").exists()
    assert not (paths.graph_output_dir / "task_graph.png").exists()


@pytest.mark.parametrize("filename", ["seed_task_graph.png", "task_graph.png"])
def test_existing_graph_artifact_requires_overwrite(
    tmp_path: Path,
    filename: str,
) -> None:
    config_dir = tmp_path / "configs"
    graph_root = tmp_path / "outputs" / "graph"
    graph_dir = graph_root / "existing_task"
    graph_dir.mkdir(parents=True)
    (graph_dir / filename).write_bytes(_PNG_SIGNATURE)

    with pytest.raises(FileExistsError, match=filename):
        raise_if_generated_files_exist(
            config_dir,
            overwrite=False,
            task_name="existing_task",
            graph_output_root=graph_root,
        )


def test_config_bundle_does_not_load_or_write_graphs_by_default(
    tmp_path: Path,
) -> None:
    bundle = _bundle("core")
    for key in ("task_prompt", "basic_background", "atom_actions"):
        del bundle[key]
    paths = write_config_bundle(
        output_dir=tmp_path / "configs",
        bundle=bundle,
        overwrite=False,
        graph_output_root=tmp_path / "outputs" / "graph",
    )

    assert paths.graph_output_dir is None
    assert paths.seed_task_graph_png is None
    assert not (tmp_path / "outputs" / "graph").exists()
    assert {path.name for path in paths.output_dir.iterdir()} == {
        "fast_gym_config.json",
        "agent_config.json",
        "seed_task_graph.json",
    }


def test_overwrite_removes_legacy_task_and_compiled_graphs(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    graph_root = tmp_path / "outputs" / "graph"
    graph_dir = graph_root / "bundle_cleanup"
    config_dir.mkdir(parents=True)
    graph_dir.mkdir(parents=True)
    for filename in (
        "seed_task_graph.png",
        "task_graph.json",
        "task_graph.png",
        "agent_compiled_graph.json",
    ):
        (config_dir / filename).write_text("legacy", encoding="utf-8")
    (graph_dir / "task_graph.png").write_bytes(_PNG_SIGNATURE)

    paths = write_config_bundle(
        output_dir=config_dir,
        bundle=_bundle("new", task_name="bundle_cleanup"),
        overwrite=True,
        graph_output_root=graph_root,
        graph_renderer=render_seed_task_graph_png,
    )

    assert paths.seed_task_graph.is_file()
    assert not (config_dir / "seed_task_graph.png").exists()
    assert not (config_dir / "task_graph.json").exists()
    assert not (config_dir / "task_graph.png").exists()
    assert not (config_dir / "agent_compiled_graph.json").exists()
    assert not (graph_dir / "task_graph.png").exists()


def test_seed_png_publication_failure_restores_old_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / "configs"
    graph_root = tmp_path / "outputs" / "graph"
    paths = write_config_bundle(
        output_dir=config_dir,
        bundle=_bundle("old", task_name="bundle_rollback"),
        overwrite=False,
        graph_output_root=graph_root,
        graph_renderer=render_seed_task_graph_png,
    )
    old_contents = {path: path.read_bytes() for path in _artifact_paths(paths)}
    real_replace = os.replace

    def fail_seed_png(source, destination) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if source_path.suffix == ".tmp" and destination_path.name == (
            "seed_task_graph.png"
        ):
            raise OSError("injected seed PNG publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(config_io.os, "replace", fail_seed_png)
    with pytest.raises(OSError, match="injected"):
        write_config_bundle(
            output_dir=config_dir,
            bundle=_bundle("new", task_name="bundle_rollback"),
            overwrite=True,
            graph_output_root=graph_root,
            graph_renderer=render_seed_task_graph_png,
        )

    assert {path: path.read_bytes() for path in _artifact_paths(paths)} == old_contents
    assert not list(tmp_path.rglob(".*.tmp"))
    assert not list(tmp_path.rglob(".*.bak"))


def test_config_bundle_rejects_config_stage_task_graph(tmp_path: Path) -> None:
    bundle = _bundle("invalid")
    bundle["task_graph"] = {"nodes": [], "edges": []}

    with pytest.raises(ValueError, match="must not publish task_graph"):
        write_config_bundle(
            output_dir=tmp_path / "configs",
            bundle=bundle,
            overwrite=False,
            graph_output_root=tmp_path / "outputs" / "graph",
        )


def _bundle(label: str, *, task_name: str | None = None) -> dict:
    task_name = task_name or f"bundle_{label}"
    placement = SimpleNamespace(
        intent="place_relative",
        moved_runtime_uid=f"object_{label}",
        reference_runtime_uid="reference",
        relation="left_of",
        reference_is_initial_pose=False,
        orientation_goal="preserve",
        orientation_axis="none",
        orientation_align_to_runtime_uid=None,
        arm_request="left",
        step_id="s01_place",
        depends_on=(),
    )
    seed = make_relative_seed_task_graph(
        task_name,
        SimpleNamespace(
            intent="place_relative",
            placements=(placement,),
            coordinated_direction=None,
            coordinated_terminal_behavior=None,
        ),
    )
    return {
        "gym_config": {"label": label, "kind": "gym"},
        "agent_config": {"label": label, "kind": "agent"},
        "task_prompt": f"{label} task",
        "seed_task_graph": seed,
        "basic_background": f"{label} background",
        "atom_actions": f"{label} actions",
        "summary": {"label": label},
    }


def _artifact_paths(paths) -> tuple[Path, ...]:
    return tuple(
        path
        for path in (
            paths.gym_config,
            paths.agent_config,
            paths.task_prompt,
            paths.seed_task_graph,
            paths.seed_task_graph_png,
            paths.basic_background,
            paths.atom_actions,
        )
        if path is not None
    )
