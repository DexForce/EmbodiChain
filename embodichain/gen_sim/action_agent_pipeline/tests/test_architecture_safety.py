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

import ast
import os
from pathlib import Path
from typing import Any

import pytest

from embodichain.gen_sim.action_agent_pipeline.agents.compile_agent import CompileAgent
from embodichain.gen_sim.action_agent_pipeline.contracts import (
    COMPILED_GRAPH_FILENAME,
    TASK_GRAPH_FILENAME,
)
from embodichain.gen_sim.action_agent_pipeline.env_adapters.tableware.agent_env import (
    AgenticGenSimEnv,
)
from embodichain.gen_sim.action_agent_pipeline.generation import config_io
from embodichain.gen_sim.action_agent_pipeline.generation.builder_protocols import (
    ArrangementSpecLike,
    ArrangementStepLike,
    RelativePlacementLike,
    RelativeSpecLike,
    StackingSpecLike,
    StackingStepLike,
    _ArrangementSpecLike,
    _ArrangementStepLike,
    _RelativePlacementLike,
    _RelativeSpecLike,
    _StackingSpecLike,
    _StackingStepLike,
)
from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    ArrangementLineSpec,
    ArrangementLineStepSpec,
    RelativePlacementSpec,
    RelativePlacementStepSpec,
    SceneObject,
    StackingSpec,
    StackingStepSpec,
    _ArrangementLineSpec,
    _ArrangementLineStepSpec,
    _RelativePlacementSpec,
    _RelativePlacementStepSpec,
    _SceneObject,
    _StackingSpec,
    _StackingStepSpec,
)
from embodichain.gen_sim.action_agent_pipeline.generation.config_io import (
    write_config_bundle,
)
from embodichain.gen_sim.action_agent_pipeline.generation.task_router import (
    TaskRouteSpec,
    _TaskRouteSpec,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.action_runtime_types import (
    CoordinatedGraspPair,
    CoordinatedPayloadRuntimeState,
    ExecutedAtomicAction,
    _CoordinatedGraspPair,
    _CoordinatedPayloadRuntimeState,
    _ExecutedAtomicAction,
)


def test_compiled_graph_defaults_to_owning_config_directory(tmp_path: Path) -> None:
    agent_config_path = tmp_path / "agent_config.json"
    agent = CompileAgent(task_name="shared_name", config_dir=str(agent_config_path))

    artifact_path, _, _ = agent.generate(
        task_graph={"task": "one"},
        regenerate=True,
    )

    assert artifact_path == tmp_path / COMPILED_GRAPH_FILENAME


def test_agent_initialization_wires_config_path_to_compiled_cache(
    tmp_path: Path,
) -> None:
    agent_config_path = tmp_path / "agent_config.json"
    task_graph_path = tmp_path / TASK_GRAPH_FILENAME
    agent_config_path.write_text("{}\n", encoding="utf-8")
    task_graph_path.write_text("{}\n", encoding="utf-8")
    env = AgenticGenSimEnv.__new__(AgenticGenSimEnv)

    env._init_agents(
        agent_config={
            "Agent": {},
            "TaskAgent": {"precomputed_task_graph": TASK_GRAPH_FILENAME},
            "CompileAgent": {},
        },
        task_name="wired_task",
        agent_config_path=str(agent_config_path),
    )

    assert env.precomputed_task_graph_path == task_graph_path
    assert env.compile_agent.config_dir == str(agent_config_path)
    assert (
        env.compile_agent._compiled_graph_path(None)
        == tmp_path / COMPILED_GRAPH_FILENAME
    )


def test_config_bundle_restores_previous_snapshot_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_paths = write_config_bundle(
        output_dir=tmp_path,
        bundle=_bundle("old"),
        overwrite=False,
    )
    old_contents = {
        path: path.read_text(encoding="utf-8") for path in _artifact_paths(old_paths)
    }
    real_replace = os.replace

    def fail_while_publishing_task_graph(
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
    ) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            source_path.suffix == ".tmp"
            and destination_path.name == TASK_GRAPH_FILENAME
        ):
            raise OSError("injected task graph publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(config_io.os, "replace", fail_while_publishing_task_graph)

    with pytest.raises(OSError, match="injected"):
        write_config_bundle(
            output_dir=tmp_path,
            bundle=_bundle("new"),
            overwrite=True,
        )

    assert {
        path: path.read_text(encoding="utf-8") for path in _artifact_paths(old_paths)
    } == old_contents
    assert not list(tmp_path.glob(".*.tmp"))
    assert not list(tmp_path.glob(".*.bak"))


def test_legacy_private_type_names_are_identity_aliases() -> None:
    assert _SceneObject is SceneObject
    assert _RelativePlacementStepSpec is RelativePlacementStepSpec
    assert _RelativePlacementSpec is RelativePlacementSpec
    assert _ArrangementLineStepSpec is ArrangementLineStepSpec
    assert _ArrangementLineSpec is ArrangementLineSpec
    assert _StackingStepSpec is StackingStepSpec
    assert _StackingSpec is StackingSpec
    assert _TaskRouteSpec is TaskRouteSpec
    assert _RelativePlacementLike is RelativePlacementLike
    assert _RelativeSpecLike is RelativeSpecLike
    assert _ArrangementStepLike is ArrangementStepLike
    assert _ArrangementSpecLike is ArrangementSpecLike
    assert _StackingStepLike is StackingStepLike
    assert _StackingSpecLike is StackingSpecLike
    assert _ExecutedAtomicAction is ExecutedAtomicAction
    assert _CoordinatedPayloadRuntimeState is CoordinatedPayloadRuntimeState
    assert _CoordinatedGraspPair is CoordinatedGraspPair


def test_production_modules_do_not_import_legacy_private_types() -> None:
    package_root = Path(__file__).resolve().parents[1]
    forbidden_by_module = {
        "embodichain.gen_sim.action_agent_pipeline.generation.config_types": {
            "_SceneObject",
            "_RelativePlacementStepSpec",
            "_RelativePlacementSpec",
            "_ArrangementLineStepSpec",
            "_ArrangementLineSpec",
            "_StackingStepSpec",
            "_StackingSpec",
        },
        "embodichain.gen_sim.action_agent_pipeline.generation.task_router": {
            "_TaskRouteSpec",
        },
        "embodichain.gen_sim.action_agent_pipeline.generation.builder_protocols": {
            "_RelativePlacementLike",
            "_RelativeSpecLike",
            "_ArrangementStepLike",
            "_ArrangementSpecLike",
            "_StackingStepLike",
            "_StackingSpecLike",
        },
        "embodichain.gen_sim.action_agent_pipeline.runtime.action_runtime_types": {
            "_ExecutedAtomicAction",
            "_CoordinatedPayloadRuntimeState",
            "_CoordinatedGraspPair",
        },
    }
    violations: list[str] = []

    # Compatibility aliases remain in their owner modules, but production
    # consumers must bind only the public names.
    for path in package_root.rglob("*.py"):
        if "tests" in path.relative_to(package_root).parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            forbidden_names = forbidden_by_module.get(node.module or "")
            if forbidden_names is None:
                continue
            for imported_name in node.names:
                bound_name = imported_name.asname or imported_name.name
                if (
                    imported_name.name in forbidden_names
                    or bound_name in forbidden_names
                ):
                    relative_path = path.relative_to(package_root)
                    violations.append(
                        f"{relative_path}:{node.lineno} imports "
                        f"{imported_name.name!r} as {bound_name!r}"
                    )

    assert not violations, "\n".join(violations)


def test_demo_action_list_hook_delegates_to_explicit_execution_method() -> None:
    env = AgenticGenSimEnv.__new__(AgenticGenSimEnv)
    calls: list[dict[str, Any]] = []
    expected = object()

    def execute(*args: Any, **kwargs: Any) -> object:
        calls.append({"args": args, "kwargs": kwargs})
        return expected

    env._execute_precomputed_task_graph = execute

    compatibility_hook = AgenticGenSimEnv.create_demo_action_list.__wrapped__
    result = compatibility_hook(env, True, "legacy", debug_mode=True)

    assert result is expected
    assert calls == [
        {
            "args": (True, "legacy"),
            "kwargs": {"debug_mode": True},
        }
    ]


def _bundle(label: str) -> dict[str, Any]:
    return {
        "gym_config": {"label": label, "kind": "gym"},
        "agent_config": {"label": label, "kind": "agent"},
        "task_prompt": f"{label} task",
        "task_graph": {"label": label, "kind": "graph"},
        "basic_background": f"{label} background",
        "atom_actions": f"{label} actions",
        "summary": {"label": label},
    }


def _artifact_paths(paths: Any) -> tuple[Path, ...]:
    return (
        paths.gym_config,
        paths.agent_config,
        paths.task_prompt,
        paths.task_graph,
        paths.basic_background,
        paths.atom_actions,
    )
