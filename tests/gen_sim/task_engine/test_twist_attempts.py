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

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodichain.gen_sim.task_engine._task_program.twist_attempts import (
    execute_support_candidates,
    support_failure,
)
from embodichain.gen_sim.task_engine._task_program.twist_support import (
    TwistSupportPenetrationError,
)
from embodichain.gen_sim.task_engine.config import TaskEnginePlanningCfg
from embodichain.gen_sim.task_engine.orchestration.execution_resources import (
    ResourceInterruptionError,
)
from embodichain.gen_sim.task_engine.workflow import SubprocessActionExecutor
from embodichain.utils.utility import save_config


def bundle(path: Path, *, height: float = 0.001, twist: bool = True) -> Path:
    path.mkdir(parents=True)
    (path / "task_program").mkdir()
    save_config(
        path / "task_program/program.yaml",
        {
            "program": {
                "items": [
                    {
                        "steps": {
                            "call": {
                                "call_id": (
                                    "gen_sim.twist" if twist else "simulation.pick"
                                )
                            }
                        }
                    }
                ]
            }
        },
    )
    (path / "twist_adaptation.json").write_text(
        json.dumps(
            {
                "policy": "gripper_measured_scale_bounded_height_constrained_relayout",
                "records": [{"support_clearance": {"requested_total_lift_m": height}}],
            }
        )
    )
    return path


@dataclass
class Preparation:
    output_dir: Path
    semantic_task_graph: dict
    adaptation: object
    generated_paths: object = None


@pytest.mark.parametrize(
    "reason,expected",
    [
        ("startup_motion", True),
        ("startup_table_contact", True),
        ("startup_robot_contact", False),
        ("target_qpos_not_reached", False),
        ("invalid_startup_evidence", False),
    ],
)
def test_support_retry_requires_specific_startup_failure(tmp_path, reason, expected):
    (tmp_path / "twist_evidence.json").write_text(
        json.dumps(
            {
                "acceptance": {
                    "phase": "initial_stable",
                    "accepted": False,
                    "startup_check": {"reason": reason},
                }
            }
        )
    )
    assert support_failure(tmp_path) is expected


def test_support_retry_rebuilds_from_same_source_and_preserves_receipts(
    tmp_path, monkeypatch
):
    from embodichain.gen_sim.task_engine import task_program_bundle as generator

    source = object()
    root = bundle(tmp_path / "bundle")
    (root / "static_scene_manifest.json").write_text('{"source": {}}')
    prep = Preparation(
        root,
        {"fingerprint": "original"},
        SimpleNamespace(
            prepared_scene=source, scene_manifest={"robot_profile": "dual_franka"}
        ),
    )
    built, called = [], []

    def generate(graph, scene, output, **kwargs):
        assert scene is source and graph is prep.semantic_task_graph
        built.append(kwargs["twist_support_lift_m"])
        bundle(output, height=kwargs["twist_support_lift_m"])
        return graph, "fresh_paths"

    def execute(path, output, **options):
        assert options == {"seed": 7}
        output.mkdir(parents=True)
        called.append(path)
        accepted = len(called) == 3
        (output / "twist_evidence.json").write_text(
            json.dumps(
                {
                    "acceptance": {
                        "phase": "turned" if accepted else "initial_stable",
                        "accepted": accepted,
                        "startup_check": {"reason": "startup_table_contact"},
                    }
                }
            )
        )
        return {"status": "succeeded" if accepted else "failed"}

    monkeypatch.setattr(generator, "generate_task_program_bundle", generate)
    report, selected, output, attempts = execute_support_candidates(
        prep, tmp_path / "action", execute, {"seed": 7}, TaskEnginePlanningCfg()
    )
    assert report["status"] == "succeeded" and built == [0.002, 0.003]
    assert [attempt["height_m"] for attempt in attempts] == [0.001, 0.002, 0.003]
    assert selected.generated_paths == "fresh_paths"
    assert (selected.output_dir / "static_scene_manifest.json").is_file()
    assert (root / "twist_adaptation.json").is_file()
    assert output.name.endswith("height_03")


def test_geometric_failure_never_erases_last_real_execution(tmp_path, monkeypatch):
    from embodichain.gen_sim.task_engine import task_program_bundle as generator

    root = bundle(tmp_path / "bundle")
    prep = Preparation(
        root,
        {},
        SimpleNamespace(
            prepared_scene=object(), scene_manifest={"robot_profile": "dual_franka"}
        ),
    )
    calls = []

    def generate(*args, **kwargs):
        calls.append(kwargs["twist_support_lift_m"])
        raise TwistSupportPenetrationError("unsafe")

    def execute(path, output, **options):
        output.mkdir(parents=True)
        (output / "twist_evidence.json").write_text(
            json.dumps(
                {
                    "acceptance": {
                        "phase": "initial_stable",
                        "accepted": False,
                        "startup_check": {"reason": "startup_motion"},
                    }
                }
            )
        )
        return {"status": "failed"}

    monkeypatch.setattr(generator, "generate_task_program_bundle", generate)
    report, selected, output, attempts = execute_support_candidates(
        prep, tmp_path / "action", execute, {}, TaskEnginePlanningCfg()
    )
    assert calls == [0.002, 0.003, 0.004]
    assert selected is prep and output == tmp_path / "action"
    assert report["status"] == "failed" and len(attempts) == 4


def test_missed_target_does_not_retry_height(tmp_path, monkeypatch):
    from embodichain.gen_sim.task_engine import task_program_bundle as generator

    root = bundle(tmp_path / "bundle")
    prep = Preparation(root, {}, None)
    monkeypatch.setattr(
        generator,
        "generate_task_program_bundle",
        lambda *a, **kw: pytest.fail("Should not rebuild a missed target"),
    )

    def execute(path, output, **options):
        output.mkdir(parents=True)
        (output / "twist_evidence.json").write_text(
            json.dumps(
                {
                    "acceptance": {
                        "phase": "turned",
                        "accepted": False,
                        "reason": "target_qpos_not_reached",
                    }
                }
            )
        )
        return {"status": "failed"}

    _, selected, _, attempts = execute_support_candidates(
        prep, tmp_path / "action", execute, {}, TaskEnginePlanningCfg()
    )
    assert selected is prep and len(attempts) == 1


def test_resource_retry_keeps_seed_and_never_selects_interrupted_attempt(
    tmp_path, monkeypatch
):
    from embodichain.gen_sim.task_engine.orchestration import (
        execution_resources as resources,
    )

    root = bundle(tmp_path / "bundle")
    executor = SubprocessActionExecutor()
    seen, waits = [], []

    def execute(path, output, **options):
        output.mkdir(parents=True)
        seen.append(options)
        if len(seen) == 1:
            raise ResourceInterruptionError(
                output / "receipt.json",
                {
                    "monitor_path": str(output / "memory.jsonl"),
                    "min_running_available_bytes": 100,
                    "other_sim_processes_at_interrupt": [],
                },
            )
        (output / "twist_evidence.json").write_text(
            '{"acceptance": {"accepted": true}}'
        )
        return {"status": "succeeded"}

    monkeypatch.setattr(executor, "_execute", execute)
    monkeypatch.setattr(
        resources, "wait_for_competitors", lambda error: waits.append(error)
    )
    report = executor(root, tmp_path / "execution", seed=3, num_envs=1)
    assert report["status"] == "succeeded" and seen[0] == seen[1] and len(waits) == 1
    selection = json.loads(
        (tmp_path / "execution/execution_selection.json").read_text()
    )
    assert selection["resource_attempt"] == 2
    assert (tmp_path / "execution/resource_attempts/attempt_0001").is_dir()
    assert (tmp_path / "execution/twist_evidence.json").is_file()


def test_non_e8_executor_does_not_add_resource_attempts(tmp_path, monkeypatch):
    root = bundle(tmp_path / "bundle", twist=False)
    executor = SubprocessActionExecutor()
    called = []
    monkeypatch.setattr(
        executor,
        "_execute",
        lambda path, output, **options: called.append(output)
        or {"status": "succeeded"},
    )
    executor(root, tmp_path / "execution", seed=0, num_envs=1)
    assert called == [tmp_path / "execution"]
    assert not (tmp_path / "execution").exists()


def test_resource_admission_failure_is_not_retried_as_a_physical_attempt(
    tmp_path, monkeypatch
):
    from embodichain.gen_sim.task_engine.orchestration import (
        execution_resources as resources,
    )

    root = bundle(tmp_path / "bundle")
    executor = SubprocessActionExecutor()
    calls = []

    def execute(path, output, **options):
        calls.append(output)
        raise resources.ResourceAdmissionError(
            output / "memory.jsonl",
            "no_running_simulation_and_memory_timeout",
            {"available_bytes": 9 * 1024**3, "total_bytes": 32 * 1024**3},
        )

    monkeypatch.setattr(executor, "_execute", execute)
    with pytest.raises(resources.ResourceAdmissionError):
        executor(root, tmp_path / "execution", seed=0, num_envs=1)
    assert len(calls) == 1
    attempts = json.loads((tmp_path / "execution/resource_attempts.json").read_text())
    assert attempts[0]["status"] == "resource_not_admitted"
