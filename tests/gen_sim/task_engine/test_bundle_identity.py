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
import subprocess
import sys

import pytest

from embodichain.gen_sim.task_engine.bundle_identity import (
    BundleIdentity,
    create_bundle_identity,
    verify_bundle_files,
)


def _identity():
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
    import json

    files["source_input_provenance.json"] = json.dumps(
        {
            "schema_version": "gen_sim.source-input-provenance/v1",
            "groups": {"application": {"asset:robot@revision1": "4" * 64}},
        }
    ).encode()
    values = dict(
        integration_id="integration",
        integration_fingerprint="1" * 64,
        registration_declaration={"services": {"motion": {"sample_count": 140}}},
        program_fingerprint="2" * 64,
        task_spec_digest="3" * 64,
        implementation_revisions={"lab": "commit:abc", "task_engine": "commit:def"},
        component_files={
            name: hashlib.sha256(data).hexdigest() for name, data in files.items()
        },
        external_inputs={"asset:robot@revision1": "4" * 64},
    )
    return create_bundle_identity(**values), values, files


def test_bundle_identity_owns_inputs_and_verifies_complete_declarations():
    identity, values, files = _identity()
    original = identity.to_dict()
    values["registration_declaration"]["services"]["motion"]["sample_count"] = 260
    assert identity.to_dict() == original
    independent = BundleIdentity.from_dict(identity)
    assert independent is not identity and independent.digest == identity.digest
    exported = identity.to_dict()
    exported["external_inputs"].clear()
    assert identity.to_dict() == original
    verify_bundle_files(identity, files)
    for changed in (
        {**files, "undeclared.yaml": b"x"},
        {key: val for key, val in files.items() if key != "task_spec.json"},
        {**files, "task_spec.json": b"changed"},
    ):
        with pytest.raises(ValueError):
            verify_bundle_files(identity, changed)


@pytest.mark.parametrize("via_symlink", [False, True])
@pytest.mark.parametrize("orchestrated", [False, True])
def test_bundle_publication_rejects_sealed_ancestors_without_modifying_history(
    tmp_path, via_symlink, orchestrated
):
    from embodichain.gen_sim.task_engine.bundle_io import (
        BundleSnapshot,
        read_bundle_snapshot,
        write_bundle_atomic,
    )

    identity, values, files = _identity()
    snapshot = BundleSnapshot(identity, files)
    historical = write_bundle_atomic(tmp_path / "historical", snapshot)
    before = {
        str(path.relative_to(historical)): path.read_bytes()
        for path in historical.rglob("*")
        if path.is_file()
    }
    parent = historical
    if via_symlink:
        parent = tmp_path / "historical-link"
        parent.symlink_to(historical, target_is_directory=True)
    with pytest.raises(ValueError, match="sealed"):
        if orchestrated:
            from embodichain.gen_sim.task_engine.orchestration.artifacts import (
                ArtifactTransaction,
            )

            with ArtifactTransaction(parent / "new-parent" / "nested"):
                pytest.fail("Sealed ancestor admitted a staging transaction.")
        else:
            write_bundle_atomic(parent / "new-parent" / "nested", snapshot)
    assert not (historical / "new-parent").exists()
    assert before == {
        str(path.relative_to(historical)): path.read_bytes()
        for path in historical.rglob("*")
        if path.is_file()
    }
    assert (
        read_bundle_snapshot(
            historical,
            external_input_digests=values["external_inputs"],
            implementation_revisions=values["implementation_revisions"],
        ).identity.digest
        == identity.digest
    )


def test_bundle_transaction_preserves_sealed_descendant_paths(tmp_path):
    from embodichain.gen_sim.task_engine.bundle_io import (
        BundleSnapshot,
        write_bundle_atomic,
    )
    from embodichain.gen_sim.task_engine.orchestration.artifacts import (
        ArtifactTransaction,
    )

    identity, _, files = _identity()
    parent = tmp_path / "old-audit"
    historical = write_bundle_atomic(parent / "sealed", BundleSnapshot(identity, files))
    before = (historical / "integration_fingerprint.json").read_bytes()
    with pytest.raises(ValueError, match="sealed"):
        with ArtifactTransaction(parent, overwrite=True):
            pytest.fail("A transaction relocated sealed historical inputs.")
    assert (historical / "integration_fingerprint.json").read_bytes() == before


def test_public_bundle_paths_expand_user_directory(tmp_path, monkeypatch):
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from embodichain.gen_sim.task_engine.bundle_io import (
        BundleSnapshot,
        write_bundle_atomic,
    )
    from embodichain.gen_sim.task_engine.bundle_assembly import TaskProgramBundlePaths

    monkeypatch.chdir(tmp_path)
    identity, _, files = _identity()
    with TemporaryDirectory(prefix=".bundle-path-test-", dir=Path.home()) as directory:
        expected = Path(directory) / "published"
        selected = f"~/{Path(directory).name}/published"
        assert TaskProgramBundlePaths.at(selected).root == expected
        assert (
            write_bundle_atomic(selected, BundleSnapshot(identity, files)) == expected
        )
        assert expected.is_dir() and not (tmp_path / "~").exists()


@pytest.mark.parametrize(
    "field",
    [
        "registration_declaration",
        "external_inputs",
        "implementation_revisions",
        "program_fingerprint",
    ],
)
def test_changes_cannot_reuse_an_existing_identity(field):
    identity, _, _ = _identity()
    changed = identity.to_dict()
    if field == "program_fingerprint":
        changed[field] = "9" * 64
    else:
        changed[field]["injected"] = "9" * 64
    with pytest.raises(ValueError, match="changed under"):
        BundleIdentity.from_dict(changed)


def test_new_effective_inputs_receive_new_bundle_identity():
    identity, values, _ = _identity()
    for field in ("external_inputs", "implementation_revisions", "component_files"):
        changed = deepcopy(values)
        key = next(iter(changed[field]))
        changed[field][key] = "9" * 64
        assert create_bundle_identity(**changed).digest != identity.digest


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "/absolute",
        "a/../b",
        "a//b",
        "a\\b",
        "C:drive",
        "integration_fingerprint.json",
    ],
)
def test_file_manifest_rejects_ambiguous_or_self_referential_paths(name):
    _, values, _ = _identity()
    values["component_files"][name] = "5" * 64
    with pytest.raises(ValueError, match="normalized relative"):
        create_bundle_identity(**values)


def test_old_runtime_contract_is_rejected_without_runtime_imports():
    script = """
import sys
from embodichain.gen_sim.task_engine.bundle_identity import BundleIdentity
try:
    BundleIdentity.from_dict({
        'schema_version':'semantic_integration_fingerprint/v2',
        'adapter_contract':'gen_sim.task_program/2620929c/v16',
    })
except ValueError as error:
    assert 'v16' in str(error) and 'generate_task_program_bundle' in str(error)
else:
    raise AssertionError('old contract accepted')
assert not any(name == 'torch' or name.startswith('embodichain.lab') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_atomic_bundle_roundtrip_owns_bytes_and_checks_current_inputs(tmp_path):
    from embodichain.gen_sim.task_engine.bundle_io import (
        BundleSnapshot,
        read_bundle_snapshot,
        write_bundle_atomic,
    )

    identity, values, files = _identity()
    snapshot = BundleSnapshot(identity, files)
    saved = dict(files)
    files["task_spec.json"] = b"caller mutation"
    destination = tmp_path / "bundle"
    write_bundle_atomic(destination, snapshot)
    kwargs = dict(
        external_input_digests=values["external_inputs"],
        implementation_revisions=values["implementation_revisions"],
    )
    actual = read_bundle_snapshot(destination, **kwargs)
    assert dict(actual.files) == saved and actual.identity.digest == identity.digest
    with pytest.raises(TypeError):
        actual.files["task_spec.json"] = b"mutation"
    with pytest.raises(FileExistsError):
        write_bundle_atomic(destination, snapshot)
    for field in ("external_input_digests", "implementation_revisions"):
        changed = dict(kwargs)
        changed[field] = {}
        with pytest.raises(ValueError, match="differ"):
            read_bundle_snapshot(destination, **changed)
    (destination / "task_spec.json").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="file changed"):
        read_bundle_snapshot(destination, **kwargs)


@pytest.mark.parametrize("failure_point", ["write", "publish"])
def test_failed_bundle_publication_removes_only_its_staging(
    tmp_path, monkeypatch, failure_point
):
    from embodichain.gen_sim.task_engine import bundle_io

    identity, _, files = _identity()
    sentinel = tmp_path / "existing.txt"
    sentinel.write_text("keep")

    def fail(*args, **kwargs):
        raise OSError("injected failure")

    if failure_point == "write":
        monkeypatch.setattr(bundle_io.os, "fsync", fail)
    else:
        monkeypatch.setattr(bundle_io, "_publish_directory", fail)
    with pytest.raises(OSError, match="injected"):
        bundle_io.write_bundle_atomic(
            tmp_path / "bundle", bundle_io.BundleSnapshot(identity, files)
        )
    assert list(tmp_path.iterdir()) == [sentinel]
    assert sentinel.read_text() == "keep"


def test_concurrent_empty_destination_is_not_replaced(tmp_path, monkeypatch):
    from embodichain.gen_sim.task_engine import bundle_io

    identity, _, files = _identity()
    destination = tmp_path / "bundle"
    publish = bundle_io._publish_directory
    observed = []

    def concurrent(staging, target):
        target.mkdir()
        observed.append(target.stat().st_ino)
        publish(staging, target)

    monkeypatch.setattr(bundle_io, "_publish_directory", concurrent)
    with pytest.raises(FileExistsError):
        bundle_io.write_bundle_atomic(
            destination, bundle_io.BundleSnapshot(identity, files)
        )
    assert destination.stat().st_ino == observed[0]
    assert list(destination.iterdir()) == []
    assert list(tmp_path.iterdir()) == [destination]


def test_disk_loader_rejects_legacy_before_reading_components(tmp_path):
    import json
    from embodichain.gen_sim.task_engine.bundle_io import read_bundle_snapshot

    (tmp_path / "integration_fingerprint.json").write_text(
        json.dumps(
            {
                "schema_version": "semantic_integration_fingerprint/v2",
                "adapter_contract": "gen_sim.task_program/2620929c/v16",
            }
        )
    )
    with pytest.raises(ValueError, match="Regenerate"):
        read_bundle_snapshot(
            tmp_path, external_input_digests={}, implementation_revisions={}
        )


@pytest.mark.parametrize(
    "change", ["extra_file", "component_link", "envelope_duplicate"]
)
def test_disk_loader_rejects_ambiguous_declaration_inventory(tmp_path, change):
    from embodichain.gen_sim.task_engine.bundle_io import (
        BundleSnapshot,
        read_bundle_snapshot,
        write_bundle_atomic,
    )

    identity, values, files = _identity()
    root = write_bundle_atomic(tmp_path / "bundle", BundleSnapshot(identity, files))
    if change == "extra_file":
        (root / "extra.yaml").write_text("undeclared")
    elif change == "component_link":
        outside = tmp_path / "outside.json"
        outside.write_bytes(files["task_spec.json"])
        (root / "task_spec.json").unlink()
        (root / "task_spec.json").symlink_to(outside)
    else:
        envelope = root / "integration_fingerprint.json"
        text = envelope.read_text()
        envelope.write_text('{"schema_version":"ignored",' + text[1:])
    with pytest.raises(ValueError):
        read_bundle_snapshot(
            root,
            external_input_digests=values["external_inputs"],
            implementation_revisions=values["implementation_revisions"],
        )


@pytest.mark.parametrize(
    "schema,contract",
    [
        ("semantic_integration_fingerprint/v2", "gen_sim.task_program/2620929c/v16"),
        ("semantic_integration_fingerprint/v3", "gen_sim.task_program/shared/v1"),
    ],
)
def test_predecessor_envelopes_reject_before_loading_lab_or_components(
    tmp_path, schema, contract
):
    import json

    root = tmp_path / "old"
    root.mkdir()
    (root / "integration_fingerprint.json").write_text(
        json.dumps({"schema_version": schema, "adapter_contract": contract})
    )
    script = f"""
import sys
from embodichain.gen_sim.task_engine.bundle_loading import load_current_task_program_bundle
try:
    load_current_task_program_bundle({str(root)!r})
except ValueError as error:
    assert 'Regenerate from frozen task declarations' in str(error)
else:
    raise AssertionError('old bundle accepted')
assert not any(n=='torch' or n.startswith('embodichain.lab') for n in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize("during_write", [False, True])
def test_atomic_publication_rechecks_owner_inputs_without_resealing(
    tmp_path, monkeypatch, during_write
):
    from embodichain.gen_sim.task_engine import bundle_io

    identity, _, files = _identity()
    snapshot = bundle_io.BundleSnapshot(identity, files)
    asset = tmp_path / "source.bin"
    asset.write_bytes(b"original")

    def validate():
        if asset.read_bytes() != b"original":
            raise ValueError("source input drift")

    if during_write:
        fsync = bundle_io.os.fsync

        def change(fd):
            asset.write_bytes(b"changed while staging")
            return fsync(fd)

        monkeypatch.setattr(bundle_io.os, "fsync", change)
    else:
        asset.write_bytes(b"changed before write")
    destination = tmp_path / "bundle"
    with pytest.raises(ValueError, match="source input drift"):
        bundle_io.write_bundle_atomic(destination, snapshot, validate_inputs=validate)
    assert not destination.exists()
    assert not list(tmp_path.glob(".bundle.partial-*"))
    assert snapshot.identity.digest == identity.digest


@pytest.mark.parametrize("boundary", ["destination_race", "source_drift"])
def test_sealed_orchestration_publication_keeps_no_replace_and_source_validation(
    tmp_path, monkeypatch, boundary
):
    import json
    from embodichain.gen_sim.task_engine import bundle_io
    from embodichain.gen_sim.task_engine.orchestration import artifacts

    identity, _, files = _identity()
    destination = tmp_path / "bundle"
    asset = tmp_path / "source.bin"
    asset.write_bytes(b"original")

    def validate():
        if asset.read_bytes() != b"original":
            raise ValueError("source input drift")

    if boundary == "destination_race":
        publish = bundle_io._publish_directory

        def compete(staging, selected):
            selected.mkdir()
            publish(staging, selected)

        monkeypatch.setattr(bundle_io, "_publish_directory", compete)
    else:
        relocate = artifacts._relocate_artifact_paths

        def change(staging, selected):
            relocate(staging, selected)
            asset.write_bytes(b"changed during final staging")

        monkeypatch.setattr(artifacts, "_relocate_artifact_paths", change)
    with pytest.raises((FileExistsError, ValueError)):
        with artifacts.ArtifactTransaction(destination) as transaction:
            staging = transaction.staging_dir
            for name, content in files.items():
                path = staging / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            (staging / "integration_fingerprint.json").write_text(
                json.dumps(identity.to_dict())
            )
            transaction.commit(validate_inputs=validate)
    assert not list(tmp_path.glob(".bundle.staging-*"))
    if boundary == "destination_race":
        assert destination.is_dir() and not list(destination.iterdir())
    else:
        assert not destination.exists()
