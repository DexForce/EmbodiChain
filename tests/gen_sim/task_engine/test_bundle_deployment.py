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
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from embodichain.gen_sim.task_engine.bundle_deployment import (
    prepare_configured_bundle,
    load_configured_bundle,
)
from embodichain.gen_sim.task_engine.bundle_identity import create_bundle_identity
from embodichain.gen_sim.task_engine.bundle_io import (
    BundleSnapshot,
    write_bundle_atomic,
)


def _files():
    fixture = json.loads(
        (Path(__file__).parent / "data/e1_bundle_declarations.json").read_text()
    )
    files = {name: value.encode() for name, value in fixture["files"].items()}
    # Keep the frozen source program, TaskSpec, assets and policy unchanged. Main
    # now derives additional capability descriptors, so bind this graph through
    # the same public composition used for a newly prepared deployment.
    from embodichain.lab.task_program.integrations import (
        load_configured_task_program_deployment,
    )

    configured = load_configured_task_program_deployment(
        task_program={
            "program": {"inline": yaml.safe_load(files["task_program/program.yaml"])},
            "integration": {
                "inline": yaml.safe_load(files["task_program/integration.yaml"])
            },
            "execution_policy": {
                "inline": yaml.safe_load(files["components/execution_policy.yaml"])
            },
        },
        skill_profile=yaml.safe_load(files["components/embodiment.yaml"])[
            "skill_profile"
        ],
        base_dir=Path(__file__).parent,
    )
    graph = json.loads(files["semantic_task_graph.json"])
    graph["integration_fingerprint"] = configured.integration.integration_fingerprint
    files["semantic_task_graph.json"] = json.dumps(graph).encode()
    return files


def _prepare(files, directory):
    return prepare_configured_bundle(
        files,
        base_dir=directory,
        external_input_digests={"fixture-assets": "a" * 64},
        implementation_revisions={"embodichain": "test-revision"},
    )


def _load(directory):
    return load_configured_bundle(
        directory,
        external_input_digests={"fixture-assets": "a" * 64},
        implementation_revisions={"embodichain": "test-revision"},
    )


def test_public_bundle_publish_and_load_share_expanded_user_path(tmp_path, monkeypatch):
    from tempfile import TemporaryDirectory

    monkeypatch.chdir(tmp_path)
    snapshot = _prepare(_files(), tmp_path)
    with TemporaryDirectory(prefix=".bundle-roundtrip-test-", dir=Path.home()) as owner:
        expected = Path(owner) / "published"
        selected = f"~/{Path(owner).name}/published"
        assert write_bundle_atomic(selected, snapshot) == expected
        loaded = _load(selected)
        assert loaded.root == expected
        assert loaded.snapshot.identity.digest == snapshot.identity.digest
        assert not (tmp_path / "~").exists()


def test_frozen_declarations_preflight_publish_and_load_without_simulation(
    tmp_path, monkeypatch
):
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.task_program.integrations import (
        SimulationTaskProgramAdapterFactory,
    )
    from embodichain.lab.gym.utils import _component_composition as physical

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Static bundle preparation touched live runtime or component files"
        )

    monkeypatch.setattr(SimulationManager, "__init__", forbidden)
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.motion.motion_generator import MotionGenerator
    from embodichain.lab.sim.motion.solvers.base_solver import BaseSolver
    from embodichain.lab.sim.atomic_actions import AtomicAction
    from embodichain.toolkits.graspkit.pose_generator import GraspPoseGenerator

    for owner in (
        SimulationManager,
        Robot,
        MotionGenerator,
        BaseSolver,
        AtomicAction,
        GraspPoseGenerator,
    ):
        if "__new__" in owner.__dict__:
            monkeypatch.setattr(owner, "__new__", staticmethod(forbidden))
        else:
            monkeypatch.setattr(owner, "__init__", forbidden)
    monkeypatch.setattr(
        SimulationTaskProgramAdapterFactory, "create_adapter", forbidden
    )
    monkeypatch.setattr(physical, "load_config", forbidden)
    files = _files()
    source = deepcopy(files)
    snapshot = _prepare(files, tmp_path)
    assert not list(tmp_path.iterdir())
    assert (
        snapshot.identity.to_dict()["program_fingerprint"]
        == "6a40b9fce7c7649be2ee1dc4c27b0e0202867b36d5f7844d29a1542b3c5ffb70"
    )
    assert (
        snapshot.identity.to_dict()["integration_fingerprint"]
        == "6ce9aa6db8aad8d9caf6de99526e7c660bb71091446d83730b21d5b28bc5f3c1"
    )
    root = write_bundle_atomic(tmp_path / "bundle", snapshot)
    loaded = _load(root)
    assert loaded.snapshot.identity.digest == snapshot.identity.digest
    assert loaded.deployment.program_path is None
    assert len(tuple(loaded.program.iter_segments())) == 3
    assert files == source


@pytest.mark.parametrize(
    "field",
    [
        "integration_id",
        "integration_fingerprint",
        "program_fingerprint",
        "task_spec_digest",
        "registration_declaration",
    ],
)
def test_load_rejects_semantically_false_but_hash_consistent_envelope(tmp_path, field):
    snapshot = _prepare(_files(), tmp_path)
    values = snapshot.identity.to_dict()
    for key in (
        "schema_version",
        "adapter_contract",
        "registration_fingerprint",
        "bundle_identity",
    ):
        values.pop(key)
    if field == "registration_declaration":
        values[field]["injected"] = "wrong owner declaration"
    else:
        values[field] = "wrong-integration" if field == "integration_id" else "0" * 64
    false_identity = create_bundle_identity(**values)
    root = write_bundle_atomic(
        tmp_path / "bundle", BundleSnapshot(false_identity, snapshot.files)
    )
    with pytest.raises(ValueError, match="differs from its sealed identity"):
        _load(root)


@pytest.mark.parametrize(
    "damage",
    [
        "graph_call",
        "program_target",
        "missing_native_entity",
        "old_constraints",
        "duplicate_yaml",
        "batch",
        "native_backend",
    ],
)
def test_invalid_source_declarations_fail_before_any_write(tmp_path, damage):
    files = _files()
    if damage == "old_constraints":
        files["task_program/constraints.json"] = b"{}"
    elif damage in {"batch", "native_backend"}:
        name = "task_program_deployment.yaml"
        value = yaml.safe_load(files[name])
        if damage == "batch":
            value["num_envs"] = 2
        else:
            value["physics"] = "newton"
        files[name] = yaml.safe_dump(value).encode()
    elif damage == "duplicate_yaml":
        files["task_program/program.yaml"] += b"\nprogram_id: duplicate\n"
    else:
        name = {
            "graph_call": "semantic_task_graph.json",
            "program_target": "task_program/program.yaml",
            "missing_native_entity": "components/scene.yaml",
        }[damage]
        value = yaml.safe_load(files[name])
        if damage == "graph_call":
            value["nodes"][0]["call"]["object"] = "different_subject"
        elif damage == "program_target":
            value["targets"]["extra"] = {
                "kind": "pose",
                "position": [0, 0, 0],
                "quaternion_xyzw": [0, 0, 0, 1],
            }
        else:
            value["simulation"]["rigid_object"] = []
        files[name] = yaml.safe_dump(value).encode()
    with pytest.raises((ValueError, TypeError)):
        _prepare(files, tmp_path)
    assert not list(tmp_path.iterdir())


def test_public_loader_rejects_old_contract_before_importing_runtime(tmp_path):
    (tmp_path / "integration_fingerprint.json").write_text(
        json.dumps(
            {
                "schema_version": "semantic_integration_fingerprint/v2",
                "adapter_contract": "gen_sim.task_program/2620929c/v16",
            }
        )
    )
    script = """
import sys
from embodichain.gen_sim.task_engine.bundle_deployment import load_configured_bundle
try:
    load_configured_bundle(sys.argv[1], external_input_digests={}, implementation_revisions={})
except ValueError as error:
    assert 'Regenerate' in str(error)
else:
    raise AssertionError('legacy bundle accepted')
assert not any(name == 'torch' or name.startswith('embodichain.lab') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=True)


def test_backend_gate_keeps_ordinary_shared_planning_services_available():
    from embodichain.lab.task_program.integrations.simulation.planning_services import (
        SimulationPlanningServicesCfg,
    )

    ordinary = SimulationPlanningServicesCfg()
    ordinary.validate_physics("default")
    ordinary.validate_physics("newton")
    native = SimulationPlanningServicesCfg(capture_native_revision=True)
    native.validate_physics("default")
    with pytest.raises(ValueError, match="requires Default"):
        native.validate_physics("newton")


def test_environment_configuration_uses_sealed_components_after_files_disappear(
    tmp_path, monkeypatch
):
    import shutil
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.gym.utils.registration import REGISTERED_ENVS
    from gymnasium.envs.registration import registry

    files = _files()
    declaration = yaml.safe_load(files["task_program_deployment.yaml"])
    env_id = "SealedBundleConfigurationTest-v1"
    declaration["id"] = env_id
    files["task_program_deployment.yaml"] = yaml.safe_dump(declaration).encode()
    snapshot = _prepare(files, tmp_path)
    root = write_bundle_atomic(tmp_path / "bundle", snapshot)
    loaded = _load(root)
    # Mutable assembly copies are not an alternative configuration authority.
    loaded.physical.config["num_envs"] = 8
    loaded.deployment.scene_binding.clear()
    shutil.rmtree(root)

    def no_simulator(*args, **kwargs):
        raise AssertionError("Configuration construction created a simulator")

    monkeypatch.setattr(SimulationManager, "__init__", no_simulator)
    # This configuration test does not materialize external robot/mesh assets.
    # Native qualification uses the separately hashed frozen asset inventory.
    import embodichain.data as data

    monkeypatch.setattr(data, "get_data_path", lambda value, **kwargs: value)
    assert env_id not in registry and env_id not in REGISTERED_ENVS
    try:
        actual_id, cfg = loaded.configure_environment()
        assert actual_id == env_id
        assert cfg.num_envs == 1
        assert cfg.task_program.program_id == "gen_sim_demo"
        assert env_id in registry
    finally:
        registry.pop(env_id, None)
        REGISTERED_ENVS.pop(env_id, None)


def test_owner_declaration_assembly_resolves_occurrences_before_preflight(
    tmp_path, monkeypatch
):
    from embodichain.gen_sim.task_engine.bundle_assembly import (
        prepare_task_program_bundle,
        publish_task_program_bundle,
    )
    from embodichain.lab.sim.sim_manager import SimulationManager

    def no_simulator(*args, **kwargs):
        raise AssertionError("Bundle generation must not construct simulation.")

    monkeypatch.setattr(SimulationManager, "__init__", no_simulator)
    values = {name: yaml.safe_load(content) for name, content in _files().items()}
    integration = values["task_program/integration.yaml"]
    original = integration["runtime_services"].pop("invocation_policy_bindings")
    program = values["task_program/program.yaml"]
    graph = values["semantic_task_graph.json"]
    presets = {
        segment["name"]: binding["preset_id"]
        for segment, binding in zip(program["program"]["items"], original, strict=True)
    }
    before = deepcopy(values)
    bound, snapshot = prepare_task_program_bundle(
        graph,
        values["task_spec.json"],
        program=program,
        integration=integration,
        scene=values["components/scene.yaml"],
        embodiment=values["components/embodiment.yaml"],
        execution_policy=values["components/execution_policy.yaml"],
        deployment=values["task_program_deployment.yaml"],
        invocation_presets=presets,
        external_input_digests={"fixture-assets": "a" * 64},
        implementation_revisions={"embodichain": "test-revision"},
        base_dir=tmp_path,
    )
    assert values == before
    assert not list(tmp_path.iterdir())
    resolved = yaml.safe_load(snapshot.files["task_program/integration.yaml"])
    assert resolved["runtime_services"]["invocation_policy_bindings"] == original
    assert bound["integration_fingerprint"] == graph["integration_fingerprint"]
    paths = publish_task_program_bundle(snapshot, tmp_path / "bundle")
    loaded = _load(paths.root)
    assert loaded.snapshot.identity.digest == snapshot.identity.digest
    assert paths.integration_fingerprint.is_file()
    with pytest.raises(FileExistsError):
        publish_task_program_bundle(snapshot, paths.root)
    with pytest.raises(ValueError, match="Missing explicit invocation"):
        prepare_task_program_bundle(
            graph,
            values["task_spec.json"],
            program=program,
            integration=integration,
            scene=values["components/scene.yaml"],
            embodiment=values["components/embodiment.yaml"],
            execution_policy=values["components/execution_policy.yaml"],
            deployment=values["task_program_deployment.yaml"],
            invocation_presets={},
            external_input_digests={"fixture-assets": "a" * 64},
            implementation_revisions={"embodichain": "test-revision"},
            base_dir=tmp_path,
        )


def test_absent_spec_is_explicit_executable_bundle_state_not_success(tmp_path):
    from embodichain.gen_sim.task_engine.bundle_spec import (
        bundle_task_spec_payload,
        decode_bundle_task_spec,
    )
    from embodichain.gen_sim.task_spec import canonical_hash

    files = _files()
    files["task_spec.json"] = json.dumps(bundle_task_spec_payload(None)).encode()
    snapshot = _prepare(files, tmp_path)
    write_bundle_atomic(tmp_path / "bundle", snapshot)
    loaded = _load(tmp_path / "bundle")
    assert loaded.task_spec is None
    assert loaded.program.program_id
    assert loaded.snapshot.identity.to_dict()["task_spec_digest"] == canonical_hash(
        {"schema_version": "task_spec.absent/v1"}
    )
    for bad in (
        {"schema_version": "task_spec.absent/v2"},
        {"schema_version": "task_spec.absent/v1", "success": True},
    ):
        with pytest.raises((ValueError, TypeError)):
            decode_bundle_task_spec(bad)
    assert (
        decode_bundle_task_spec({}) is not None
    )  # Existing TaskSpec defaults remain unchanged.
    with pytest.raises(TypeError):
        decode_bundle_task_spec(None)
    # Replacing sealed absence with a supplied specification is a new identity.
    (tmp_path / "bundle/task_spec.json").write_bytes(_files()["task_spec.json"])
    with pytest.raises(ValueError, match="hash|bytes|changed|identity"):
        _load(tmp_path / "bundle")


def test_current_binding_rebuild_keeps_original_program_spec_and_success_declarations():
    original = json.loads(
        (Path(__file__).parent / "data/e1_bundle_declarations.json").read_text()
    )["files"]
    rebuilt = _files()
    assert set(rebuilt) == set(original)
    for name in original:
        if name != "semantic_task_graph.json":
            assert rebuilt[name] == original[name].encode()
    old_graph = json.loads(original["semantic_task_graph.json"])
    new_graph = json.loads(rebuilt["semantic_task_graph.json"])
    old_graph.pop("integration_fingerprint")
    new_graph.pop("integration_fingerprint")
    assert new_graph == old_graph
