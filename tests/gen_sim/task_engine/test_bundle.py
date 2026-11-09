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

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from embodichain.gen_sim.task_engine.bundle_identity import (
    BundleIdentity,
    create_bundle_identity,
    verify_bundle_files,
)
from embodichain.gen_sim.task_engine.bundle_io import (
    BundleSnapshot,
    read_bundle_snapshot,
    write_bundle_atomic,
)
from embodichain.gen_sim.task_engine.bundle_deployment import (
    load_configured_bundle,
    prepare_configured_bundle,
)


def _snapshot():
    files = {
        name: name.encode()
        for name in (
            "task_program_deployment.yaml",
            "task_program/program.yaml",
            "task_program/integration.yaml",
            "components/scene.yaml",
            "components/embodiment.yaml",
            "components/execution_policy.yaml",
            "task_spec.json",
            "semantic_task_graph.json",
        )
    }
    inputs = {"robot.urdf": "4" * 64}
    files["source_input_provenance.json"] = json.dumps(
        {
            "schema_version": "gen_sim.source-input-provenance/v1",
            "groups": {"application": inputs},
        }
    ).encode()
    revisions = {"application": "revision-1"}
    identity = create_bundle_identity(
        integration_id="test",
        integration_fingerprint="1" * 64,
        program_fingerprint="2" * 64,
        task_spec_digest="3" * 64,
        registration_declaration={"profile": "test"},
        component_files={
            name: hashlib.sha256(data).hexdigest() for name, data in files.items()
        },
        external_inputs=inputs,
        implementation_revisions=revisions,
    )
    return BundleSnapshot(identity, files), inputs, revisions


def test_identity_owns_inputs_and_rejects_content_drift():
    snapshot, _, _ = _snapshot()
    data = snapshot.identity.to_dict()
    original = deepcopy(data)
    data["registration_declaration"]["profile"] = "changed"
    assert snapshot.identity.to_dict() == original
    with pytest.raises(ValueError):
        BundleIdentity.from_dict(data)
    with pytest.raises(TypeError):
        snapshot.files["task_spec.json"] = b"changed"
    files = dict(snapshot.files)
    files["task_spec.json"] = b"changed"
    with pytest.raises(ValueError):
        verify_bundle_files(snapshot.identity, files)


def test_atomic_roundtrip_and_no_overwrite(tmp_path):
    snapshot, inputs, revisions = _snapshot()
    root = write_bundle_atomic(tmp_path / "bundle", snapshot)
    loaded = read_bundle_snapshot(
        root, external_input_digests=inputs, implementation_revisions=revisions
    )
    assert loaded.identity.to_dict() == snapshot.identity.to_dict()
    assert loaded.files == snapshot.files
    before = {p.name: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        write_bundle_atomic(root, snapshot)
    with pytest.raises(ValueError, match="sealed"):
        write_bundle_atomic(root / "nested", snapshot)
    assert before == {p.name: p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "damage", ["file", "missing", "extra", "input", "revision", "old_version"]
)
def test_loader_rejects_changed_or_incompatible_bundle(tmp_path, damage):
    snapshot, inputs, revisions = _snapshot()
    root = write_bundle_atomic(tmp_path / "bundle", snapshot)
    if damage == "file":
        (root / "task_spec.json").write_bytes(b"changed")
    elif damage == "missing":
        (root / "task_spec.json").unlink()
    elif damage == "extra":
        (root / "extra.json").write_text("{}")
    elif damage == "input":
        inputs = {"robot.urdf": "5" * 64}
    elif damage == "revision":
        revisions = {"application": "revision-2"}
    else:
        path = root / "integration_fingerprint.json"
        data = json.loads(path.read_text())
        data["adapter_contract"] = "gen_sim.task_program/old"
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        read_bundle_snapshot(
            root, external_input_digests=inputs, implementation_revisions=revisions
        )


@pytest.mark.parametrize("failure", ["input", "write"])
def test_failed_publication_preserves_unrelated_files(tmp_path, monkeypatch, failure):
    snapshot, _, _ = _snapshot()
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("unchanged")

    def changed():
        if failure == "input":
            raise ValueError("Input changed")

    original_open = Path.open

    def failing_open(path, mode="r", *args, **kwargs):
        if path.name == "program.yaml" and mode == "xb":
            raise OSError("Disk write failed")
        return original_open(path, mode, *args, **kwargs)

    if failure == "write":
        monkeypatch.setattr(Path, "open", failing_open)
    with pytest.raises((ValueError, OSError), match="Input changed|Disk write failed"):
        write_bundle_atomic(tmp_path / "bundle", snapshot, validate_inputs=changed)
    assert list(tmp_path.iterdir()) == [sentinel]
    assert sentinel.read_text() == "unchanged"


def _configured_files():
    # Reuse existing official declarations; no copied physical regression corpus.
    from embodichain.lab.task_program.integrations import (
        load_configured_task_program_deployment,
    )
    from embodichain.gen_sim.task_engine.program_declarations import (
        build_program_declaration,
    )

    root = (
        Path(__file__).parents[3]
        / "extensions/embodichain_tasks/configs/tasks/manipulation/rubiks_cube_pick_place"
    )
    original = yaml.safe_load((root / "task.ur5.yaml").read_text())
    scene = yaml.safe_load((root / original["environment"]["component"]).read_text())
    embodiment = yaml.safe_load(
        (root / original["embodiment"]["component"]).read_text()
    )
    integration = yaml.safe_load(
        (root / original["task_program"]["integration"]).read_text()
    )
    policy = yaml.safe_load(
        (root / original["task_program"]["execution_policy"]).read_text()
    )
    call = {"kind": "pick", "object": "rubiks_cube"}
    graph = dict(
        schema_version="semantic_task_graph/v1",
        task_id="bundle_test",
        instruction="Pick the cube",
        planner_route="test",
        integration_fingerprint="0" * 64,
        targets={},
        nodes=[
            dict(
                id="pick_cube",
                call=call,
                depends_on=[],
                task_instance_id="g1",
                task_type="E1",
                role="primary",
            )
        ],
        task_groups=[
            dict(
                id="g1",
                task_type="E1",
                node_ids=["pick_cube"],
                depends_on=[],
                success={},
            )
        ],
        success={},
    )
    program = build_program_declaration(graph, integration["program_id"])
    configured = load_configured_task_program_deployment(
        task_program={
            "program": {"inline": program},
            "integration": {"inline": integration},
            "execution_policy": {"inline": policy},
        },
        skill_profile=embodiment["skill_profile"],
        base_dir=root,
    )
    graph["integration_fingerprint"] = configured.integration.integration_fingerprint
    deployment = dict(original)
    deployment["environment"] = {"component": "components/scene.yaml"}
    deployment["embodiment"] = {"component": "components/embodiment.yaml"}
    deployment["task_program"] = {
        "program": "task_program/program.yaml",
        "integration": "task_program/integration.yaml",
        "execution_policy": "components/execution_policy.yaml",
    }
    values = {
        "task_program_deployment.yaml": deployment,
        "task_program/program.yaml": program,
        "task_program/integration.yaml": integration,
        "components/scene.yaml": scene,
        "components/embodiment.yaml": embodiment,
        "components/execution_policy.yaml": policy,
        "task_spec.json": {"schema_version": "task_spec.absent/v1"},
        "semantic_task_graph.json": graph,
    }
    return {
        name: yaml.safe_dump(value).encode() for name, value in values.items()
    }, root


def test_static_deployment_roundtrip_without_live_resources(tmp_path, monkeypatch):
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.task_program.integrations import (
        SimulationTaskProgramAdapterFactory,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Static bundle touched a live resource")

    monkeypatch.setattr(SimulationManager, "__init__", forbidden)
    monkeypatch.setattr(Robot, "__init__", forbidden)
    monkeypatch.setattr(
        SimulationTaskProgramAdapterFactory, "create_adapter", forbidden
    )
    files, base = _configured_files()
    before = deepcopy(files)
    snapshot = prepare_configured_bundle(
        files,
        base_dir=base,
        external_input_digests={},
        implementation_revisions={"application": "test"},
    )
    assert files == before
    root = write_bundle_atomic(tmp_path / "bundle", snapshot)
    loaded = load_configured_bundle(
        root,
        external_input_digests={},
        implementation_revisions={"application": "test"},
    )
    assert loaded.snapshot.identity.digest == snapshot.identity.digest
    assert loaded.task_spec is None
    assert loaded.program.program_id == "rubiks_cube_pick_place"
    # Sealed components remain usable even when the original files are gone.
    loaded.deployment.load_program()
    env_id, config = loaded.configure_environment()
    assert env_id == "TaskProgramRubiksCubePickPlace-v1"
    assert config.task_program.program_id == "rubiks_cube_pick_place"
