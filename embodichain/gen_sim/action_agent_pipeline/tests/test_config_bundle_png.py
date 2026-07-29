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

import pytest

from embodichain.gen_sim.action_agent_pipeline.generation import config_io
from embodichain.gen_sim.action_agent_pipeline.generation.config_io import (
    raise_if_generated_files_exist,
    write_config_bundle,
)
from embodichain.gen_sim.action_agent_pipeline.generation.seed_task_graph import (
    compile_seed_graph_metadata,
)
from embodichain.gen_sim.action_agent_pipeline.protocol.artifacts import (
    TASK_GRAPH_PNG_FILENAME,
)

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def test_config_bundle_writes_matching_graph_png_artifacts(tmp_path: Path) -> None:
    paths = write_config_bundle(
        output_dir=tmp_path,
        bundle=_bundle("initial"),
        overwrite=False,
    )

    assert paths.seed_task_graph_png.read_bytes().startswith(_PNG_SIGNATURE)
    assert paths.task_graph_png.read_bytes().startswith(_PNG_SIGNATURE)
    assert paths.seed_task_graph.is_file()
    assert paths.task_graph.is_file()


@pytest.mark.parametrize(
    "filename",
    ["seed_task_graph.png", "task_graph.png"],
)
def test_existing_graph_png_requires_overwrite(
    tmp_path: Path,
    filename: str,
) -> None:
    (tmp_path / filename).write_bytes(_PNG_SIGNATURE)

    with pytest.raises(FileExistsError, match=filename):
        raise_if_generated_files_exist(tmp_path, overwrite=False)


def test_config_bundle_overwrite_replaces_graph_pngs(tmp_path: Path) -> None:
    paths = write_config_bundle(
        output_dir=tmp_path,
        bundle=_bundle("old"),
        overwrite=False,
    )
    old_seed_png = paths.seed_task_graph_png.read_bytes()
    old_task_png = paths.task_graph_png.read_bytes()

    new_paths = write_config_bundle(
        output_dir=tmp_path,
        bundle=_bundle("new"),
        overwrite=True,
    )

    assert new_paths.seed_task_graph_png.read_bytes() != old_seed_png
    assert new_paths.task_graph_png.read_bytes() != old_task_png


def test_seeded_bundle_rejects_non_renderable_task_graph(tmp_path: Path) -> None:
    bundle = _bundle("invalid")
    bundle["task_graph"].pop("nodes")

    with pytest.raises(RuntimeError, match="task_graph.png"):
        write_config_bundle(
            output_dir=tmp_path,
            bundle=bundle,
            overwrite=False,
        )

    assert not list(tmp_path.iterdir())


def test_binary_publication_failure_restores_entire_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = write_config_bundle(
        output_dir=tmp_path,
        bundle=_bundle("old"),
        overwrite=False,
    )
    old_contents = {path: path.read_bytes() for path in _artifact_paths(paths)}
    real_replace = os.replace

    def fail_while_publishing_task_graph_png(
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
    ) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            source_path.suffix == ".tmp"
            and destination_path.name == TASK_GRAPH_PNG_FILENAME
        ):
            raise OSError("injected task graph PNG publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(
        config_io.os,
        "replace",
        fail_while_publishing_task_graph_png,
    )

    with pytest.raises(OSError, match="injected"):
        write_config_bundle(
            output_dir=tmp_path,
            bundle=_bundle("new"),
            overwrite=True,
        )

    assert {path: path.read_bytes() for path in _artifact_paths(paths)} == old_contents
    assert not list(tmp_path.glob(".*.tmp"))
    assert not list(tmp_path.glob(".*.bak"))


def _bundle(label: str) -> dict:
    seed_graph = {
        "schema_version": "seed_task_graph_v1",
        "task": f"bundle_{label}",
        "route": "object_manipulation",
        "program": "place_relative",
        "steps": [
            {
                "id": "s01_place",
                "operator": "place_relative",
                "object": f"object_{label}",
                "actor": {"mode": "required", "arm": "left_arm"},
                "goal": {
                    "relation": "left_of",
                    "reference_object": "reference",
                    "reference_state": "live",
                    "orientation_goal": "preserve",
                    "orientation_axis": "none",
                },
                "depends_on": [],
                "postcondition": {
                    "type": "semantic_goal",
                    "operator": "place_relative",
                    "relation": "left_of",
                },
            }
        ],
    }
    task_graph = compile_seed_graph_metadata(
        {
            "task": f"bundle_{label}",
            "start": "v0",
            "goal": "v1",
            "nodes": [
                {"id": "v0", "semantic": "Initial state"},
                {"id": "v1", "semantic": f"Move object for {label}"},
            ],
            "edges": [
                {
                    "id": "e01",
                    "source": "v0",
                    "target": "v1",
                    "left_arm_action": {
                        "atomic_action_class": "PickUp",
                        "target_object": {"obj_name": f"object_{label}"},
                    },
                    "right_arm_action": None,
                }
            ],
            "semantic_steps": [
                {
                    "id": "s01_place",
                    "edge_ids": ["e01"],
                }
            ],
        },
        seed_graph,
    )
    return {
        "gym_config": {"label": label, "kind": "gym"},
        "agent_config": {"label": label, "kind": "agent"},
        "task_prompt": f"{label} task",
        "seed_task_graph": seed_graph,
        "task_graph": task_graph,
        "basic_background": f"{label} background",
        "atom_actions": f"{label} actions",
        "summary": {"label": label},
    }


def _artifact_paths(paths) -> tuple[Path, ...]:
    return (
        paths.gym_config,
        paths.agent_config,
        paths.task_prompt,
        paths.seed_task_graph,
        paths.seed_task_graph_png,
        paths.task_graph,
        paths.task_graph_png,
        paths.basic_background,
        paths.atom_actions,
    )
