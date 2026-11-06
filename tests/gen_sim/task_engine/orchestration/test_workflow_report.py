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

"""Workflow acceptance consumes fresh public reports and exact attempt identity."""

from __future__ import annotations

from pathlib import Path

import pytest

from embodichain.gen_sim.task_engine.bundle_deployment import prepare_configured_bundle
from embodichain.gen_sim.task_engine.bundle_io import write_bundle_atomic
from embodichain.gen_sim.task_engine.bundle_loading import (
    load_current_task_program_bundle,
)
from embodichain.gen_sim.task_engine.execution import TaskProgramBundleEpisode
from embodichain.gen_sim.task_engine.reporting import build_bundle_execution_report
from embodichain.gen_sim.task_engine.runtime_evidence import TaskEngineEvidenceSession
from embodichain.gen_sim.task_engine.task_program_bundle import (
    task_program_implementation_revisions,
)
from embodichain.gen_sim.task_engine.workflow import _environment_successes
from embodichain.gen_sim.task_engine.orchestration.workflow_artifacts import (
    write_run_json,
)

from ..test_runtime_evidence import _session, _boundary, _episode


@pytest.mark.parametrize("resource_attempt", [None, 2])
def test_workflow_consumes_new_report_with_exact_execution_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resource_attempt: int | None
) -> None:
    # Reuse the public collector fixture with a known upright pose. These are
    # software report/identity checks; no simulated physical success is claimed.
    original, source, env, bridge, binding = _session(tmp_path, monkeypatch)
    original.close()
    from ..test_bundle_loading import fixture as portable_bundle

    inputs_root = tmp_path / "current-inputs"
    inputs_root.mkdir()
    _, _, portable = portable_bundle(inputs_root)
    files = dict(portable.files)
    # Keep this report fixture's exact normative state predicate while using
    # real independently hashed runtime assets for the current-package loader.
    files["task_spec.json"] = source.snapshot.files["task_spec.json"]
    snapshot = prepare_configured_bundle(
        files,
        base_dir=tmp_path,
        external_input_digests=portable.identity.to_dict()["external_inputs"],
        implementation_revisions=task_program_implementation_revisions(),
    )
    bundle_root = write_bundle_atomic(tmp_path / "current-bundle", snapshot)
    loaded = load_current_task_program_bundle(bundle_root)
    session = TaskEngineEvidenceSession(loaded, env, bridge)
    boundary = _boundary(bridge, success=True)
    session.capture_boundary(boundary)
    episode = _episode([boundary], success=True)
    evidence = session.finish(episode, final_bindings=(binding,))
    logical_root = tmp_path / "execution"
    actual_root = logical_root
    if resource_attempt is not None:
        actual_root = (
            logical_root / "resource_attempts" / f"attempt_{resource_attempt:04d}"
        )
        write_run_json(
            logical_root / "execution_selection.json",
            {
                "selected_execution_root": str(actual_root),
                "resource_attempt": resource_attempt,
            },
        )
    actual_root.mkdir(parents=True, exist_ok=True)
    report = build_bundle_execution_report(
        loaded, TaskProgramBundleEpisode(episode, evidence), record_dir=actual_root
    )
    assert _environment_successes(
        report, bundle=bundle_root, execution_root=logical_root
    ) == [True]
    assert _environment_successes(
        report,
        bundle=bundle_root,
        execution_root=logical_root,
        required_semantic_steps=("absent-semantic-step",),
    ) == [False]
    with pytest.raises(ValueError, match="another run output"):
        _environment_successes(
            report, bundle=bundle_root, execution_root=tmp_path / "foreign"
        )
    if resource_attempt is not None:
        write_run_json(
            logical_root / "execution_selection.json",
            {
                "selected_execution_root": str(tmp_path / "foreign"),
                "resource_attempt": resource_attempt,
            },
        )
        with pytest.raises(ValueError, match="another logical attempt"):
            _environment_successes(
                report, bundle=bundle_root, execution_root=logical_root
            )
