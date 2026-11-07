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

"""Current CLI/workflow loading independently verifies input files and package code."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pytest
import yaml
from .test_bundle_deployment import _files
from embodichain.gen_sim.task_engine.bundle_deployment import prepare_configured_bundle
from embodichain.gen_sim.task_engine.bundle_io import write_bundle_atomic
from embodichain.gen_sim.task_engine.bundle_loading import (
    load_current_task_program_bundle,
)
from embodichain.gen_sim.task_engine.task_program_bundle import (
    task_program_implementation_revisions,
)


def fixture(tmp_path, *, revisions=None, omit=None):
    # Portable identity inputs: static preflight deliberately creates no Robot.
    # Runtime geometry validation is covered by the owning execution tests.
    mesh = tmp_path / "mesh.glb"
    mesh.write_bytes(
        (
            Path(__file__).parents[2] / "sim/scene/fixtures/source_ingress_mesh.glb"
        ).read_bytes()
    )
    child = tmp_path / "hand.obj"
    child.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    robot = tmp_path / "robot.urdf"
    robot.write_text(
        '<robot name="identity_fixture"><link name="base"><visual><geometry><mesh filename="hand.obj"/></geometry></visual></link></robot>'
    )
    world = tmp_path / "dependency.usda"
    # Write a real readable USD layer without constructing simulation.
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(world))
    UsdGeom.Xform.Define(stage, "/Dependency")
    stage.GetRootLayer().Save()
    files = _files()

    def paths(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "fpath" and isinstance(item, str):
                    value[key] = str(mesh)
                elif key in {"urdf_path", "xml_path", "mjcf_path"} and isinstance(
                    item, str
                ):
                    value[key] = str(robot)
                else:
                    paths(item)
        elif isinstance(value, list):
            for item in value:
                paths(item)

    for name in ("components/scene.yaml", "components/embodiment.yaml"):
        declaration = yaml.safe_load(files[name])
        paths(declaration)
        if name == "components/scene.yaml":
            declaration["simulation"].setdefault("articulation", []).append(
                {"uid": "dependency_root", "fpath": str(world)}
            )
        files[name] = yaml.safe_dump(declaration, sort_keys=False).encode()
    # Build a new source declaration identity for the portable physical inputs;
    # do not reuse the original fixture's obsolete integration fingerprint.
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
        base_dir=tmp_path,
    )
    graph = json.loads(files["semantic_task_graph.json"])
    graph["integration_fingerprint"] = configured.integration.integration_fingerprint
    files["semantic_task_graph.json"] = json.dumps(graph).encode()
    asset = tmp_path / "independent-input.bin"
    asset.write_bytes(b"frozen owner input")
    snapshot = prepare_configured_bundle(
        files,
        base_dir=tmp_path,
        external_input_digests={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in {
                asset,
                mesh,
                robot,
                child,
                world,
                *configured.integration.adapter_factory.declared_asset_files(),
            }
            if path.name != omit
        },
        implementation_revisions=(
            task_program_implementation_revisions() if revisions is None else revisions
        ),
    )
    path = write_bundle_atomic(tmp_path / "bundle", snapshot)
    return path, asset, snapshot


def test_current_loader_reads_actual_files_and_starts_no_runtime(tmp_path, monkeypatch):
    from embodichain.lab.sim.sim_manager import SimulationManager

    path, asset, snapshot = fixture(tmp_path)
    monkeypatch.setattr(
        SimulationManager,
        "__init__",
        lambda *a, **k: pytest.fail("static loading created simulation"),
    )
    loaded = load_current_task_program_bundle(path)
    assert loaded.snapshot.identity.digest == snapshot.identity.digest
    assert loaded.program.program_id == "gen_sim_demo"
    asset.write_bytes(b"changed owner input")
    with pytest.raises(ValueError, match="external inputs"):
        load_current_task_program_bundle(path)


def test_public_episode_entry_rejects_changed_inputs_before_selecting_segments(
    tmp_path,
):
    from embodichain.gen_sim.task_engine.execution import (
        execute_task_program_bundle_episode,
    )

    path, asset, _ = fixture(tmp_path)
    bundle = load_current_task_program_bundle(path)
    asset.write_bytes(b"input changed after loading")

    class UnenteredEnvironment:
        def create_demo_segments(self):
            pytest.fail("Changed inputs reached segment construction.")

    with pytest.raises(ValueError, match="external input files changed after loading"):
        execute_task_program_bundle_episode(
            bundle,
            UnenteredEnvironment(),
            prepare_evidence=lambda _: pytest.fail("Changed inputs reached evidence."),
            final_evidence=lambda *_: pytest.fail(
                "Changed inputs reached final facts."
            ),
        )


def test_current_loader_never_adopts_a_declared_implementation_revision(tmp_path):
    path, _, _ = fixture(
        tmp_path, revisions={"embodichain": "arbitrary-declared-version"}
    )
    with pytest.raises(ValueError, match="implementation revisions"):
        load_current_task_program_bundle(path)


def test_effective_grasp_model_asset_cannot_be_omitted_from_identity(tmp_path):
    path, _, _ = fixture(tmp_path, omit="robotiq_arg2f_140.urdf")
    with pytest.raises(
        ValueError, match="omits runtime dependencies.*robotiq_arg2f_140"
    ):
        load_current_task_program_bundle(path)


@pytest.mark.parametrize("list_sources", [False, True])
def test_application_robot_preset_requires_complete_model_inventory(
    tmp_path, list_sources
):
    from embodichain.lab.gym.utils.gym_utils import robot_config_asset_files

    _, _, original = fixture(tmp_path)
    files = dict(original.files)
    files.pop("source_input_provenance.json")
    embodiment = yaml.safe_load(files["components/embodiment.yaml"])
    # This is a config-only application declaration, not a physical qualification
    # of the placeholder robot fixture. Its preset owns real implicit UR5 files.
    simulation = embodiment["simulation"]
    simulation.pop("urdf_cfg")
    simulation.update(class_type="URRobot", robot_type="ur5")
    files["components/embodiment.yaml"] = yaml.safe_dump(embodiment).encode()
    identity = original.identity.to_dict()
    inputs = dict(identity["external_inputs"])
    if list_sources:
        inputs.update(
            {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in robot_config_asset_files(simulation, base_dir=tmp_path)
            }
        )
    candidate = prepare_configured_bundle(
        files,
        base_dir=tmp_path,
        external_input_digests=inputs,
        implementation_revisions=identity["implementation_revisions"],
    )
    root = write_bundle_atomic(tmp_path / "preset_bundle", candidate)
    with pytest.raises(
        ValueError, match="requires a model-owned DeclaredAssetFileProvider"
    ):
        load_current_task_program_bundle(root)


def test_application_concrete_cobot_rejects_without_full_model_inventory(tmp_path):
    _, _, original = fixture(tmp_path)
    files = dict(original.files)
    files.pop("source_input_provenance.json")
    embodiment = yaml.safe_load(files["components/embodiment.yaml"])
    # Retain the explicit fixture URDF sources. Concrete model-owned FK files
    # still require that owner's inventory even when serialized inputs agree.
    embodiment["simulation"]["class_type"] = "CobotMagic"
    files["components/embodiment.yaml"] = yaml.safe_dump(embodiment).encode()
    identity = original.identity.to_dict()
    candidate = prepare_configured_bundle(
        files,
        base_dir=tmp_path,
        external_input_digests=identity["external_inputs"],
        implementation_revisions=identity["implementation_revisions"],
    )
    root = write_bundle_atomic(tmp_path / "opaque_model", candidate)
    with pytest.raises(
        ValueError, match="requires a model-owned DeclaredAssetFileProvider"
    ):
        load_current_task_program_bundle(root)


def test_usd_asset_attribute_dependency_cannot_be_omitted_from_identity(tmp_path):
    from pxr import Sdf, Usd

    _, _, original = fixture(tmp_path)
    texture = tmp_path / "albedo.png"
    import base64

    texture.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8A"
            "AwMCAO+jN5sAAAAASUVORK5CYII="
        )
    )
    world = tmp_path / "dependency.usda"
    stage = Usd.Stage.Open(str(world))
    stage.GetPrimAtPath("/Dependency").CreateAttribute(
        "material:texture", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath(texture.name))
    stage.GetRootLayer().Save()
    identity = original.identity.to_dict()
    inputs = dict(identity["external_inputs"])
    inputs[str(world)] = hashlib.sha256(world.read_bytes()).hexdigest()
    files = dict(original.files)
    files.pop("source_input_provenance.json")
    candidate = prepare_configured_bundle(
        files,
        base_dir=tmp_path,
        external_input_digests=inputs,
        implementation_revisions=identity["implementation_revisions"],
    )
    root = write_bundle_atomic(tmp_path / "usd_asset_bundle", candidate)
    with pytest.raises(ValueError, match="omits runtime dependencies.*albedo.png"):
        load_current_task_program_bundle(root)


def test_source_audit_staging_rechecks_inputs_changed_during_final_preflight(
    tmp_path, monkeypatch
):
    from embodichain.gen_sim.task_engine import bundle_deployment
    from embodichain.gen_sim.task_engine.orchestration.coordinator import (
        _stage_prepared_snapshot,
    )

    _, asset, snapshot = fixture(tmp_path)
    prepare = bundle_deployment.prepare_configured_bundle

    def changed(*args, **kwargs):
        result = prepare(*args, **kwargs)
        asset.write_bytes(b"changed during final audit preflight")
        return result

    monkeypatch.setattr(bundle_deployment, "prepare_configured_bundle", changed)
    staging = tmp_path / "audit_staging"
    staging.mkdir()
    with pytest.raises(ValueError, match="Frozen source input changed"):
        _stage_prepared_snapshot(snapshot, staging, tmp_path / "published")
    assert not (staging / "integration_fingerprint.json").exists()


def test_old_envelope_rejects_before_input_hashing_or_component_access(
    tmp_path, monkeypatch
):
    from embodichain.gen_sim.task_engine import bundle_loading as module

    path = tmp_path / "old"
    path.mkdir()
    (path / "integration_fingerprint.json").write_text(
        json.dumps(
            dict(
                schema_version="semantic_integration_fingerprint/v2",
                adapter_contract="old",
            )
        )
    )
    monkeypatch.setattr(
        module,
        "_external_inputs",
        lambda *a: pytest.fail("legacy bundle was traversed"),
    )
    with pytest.raises(ValueError, match="Regenerate"):
        load_current_task_program_bundle(path)


def test_current_loader_rejects_input_drift_during_runtime_declaration_preflight(
    tmp_path, monkeypatch
):
    from embodichain.gen_sim.task_engine import bundle_deployment

    path, asset, _ = fixture(tmp_path)
    original = bundle_deployment.load_configured_bundle

    def drift(*args, **kwargs):
        loaded = original(*args, **kwargs)
        asset.write_bytes(b"changed during preflight")
        return loaded

    monkeypatch.setattr(bundle_deployment, "load_configured_bundle", drift)
    with pytest.raises(ValueError, match="changed during static preflight"):
        load_current_task_program_bundle(path)


def test_current_loader_rejects_current_implementation_drift_during_preflight(
    tmp_path, monkeypatch
):
    from embodichain.gen_sim.task_engine import task_program_bundle

    path, _, _ = fixture(tmp_path)
    original = task_program_bundle.task_program_implementation_revisions()
    observations = iter([original, {**original, "embodichain.source_sha256": "a" * 64}])
    monkeypatch.setattr(
        task_program_bundle,
        "task_program_implementation_revisions",
        lambda: next(observations),
    )
    with pytest.raises(ValueError, match="implementation changed"):
        load_current_task_program_bundle(path)


@pytest.mark.parametrize(
    "distribution,starting_version,upgraded_version",
    [
        ("nvidia-curobo", "0.8.0", "0.9.0"),
        ("newton", "1.6.0", "1.6.1"),
        ("warp-lang", "1.17.0", "1.17.1"),
    ],
    ids=["curobo", "newton", "warp"],
)
@pytest.mark.parametrize("missing", [False, True], ids=["upgraded", "missing"])
def test_current_loader_seals_planning_distribution_versions(
    tmp_path, monkeypatch, distribution, starting_version, upgraded_version, missing
):
    from embodichain.gen_sim.task_engine import task_program_bundle

    actual_version = task_program_bundle.metadata.version
    selected = {"version": starting_version}
    changed_version = None if missing else upgraded_version

    def distribution_version(name):
        if name == distribution:
            if selected["version"] is None:
                raise task_program_bundle.metadata.PackageNotFoundError(name)
            return selected["version"]
        return actual_version(name)

    monkeypatch.setattr(task_program_bundle.metadata, "version", distribution_version)
    path, _, original = fixture(tmp_path)
    loaded = load_current_task_program_bundle(path)
    selected["version"] = changed_version
    with pytest.raises(ValueError, match="implementation"):
        load_current_task_program_bundle(path)
    with pytest.raises(ValueError, match="implementation changed after loading"):
        loaded.configure_environment()

    revisions = task_program_implementation_revisions()
    assert revisions[distribution] == (changed_version or "unavailable")
    envelope = original.identity.to_dict()
    assert {k: v for k, v in revisions.items() if k != distribution} == {
        k: v
        for k, v in envelope["implementation_revisions"].items()
        if k != distribution
    }
    regenerated = prepare_configured_bundle(
        original.files,
        base_dir=tmp_path,
        external_input_digests=envelope["external_inputs"],
        implementation_revisions=revisions,
    )
    assert regenerated.identity.digest != original.identity.digest
    for key in ("program_fingerprint", "task_spec_digest"):
        assert regenerated.identity.to_dict()[key] == envelope[key]
    for name in ("task_program/program.yaml", "task_spec.json"):
        assert regenerated.files[name] == original.files[name]


@pytest.mark.parametrize("omitted", ["mesh.glb", "robot.urdf", "hand.obj"])
def test_current_loader_derives_missing_inputs_from_runtime_declarations(
    tmp_path, omitted, monkeypatch
):
    from embodichain.lab.sim.sim_manager import SimulationManager

    path, _, _ = fixture(tmp_path, omit=omitted)
    monkeypatch.setattr(
        SimulationManager,
        "__init__",
        lambda *a, **kw: pytest.fail("Input-set verification cannot start simulation"),
    )
    with pytest.raises(
        ValueError, match="external input set omits runtime dependencies"
    ) as failure:
        load_current_task_program_bundle(path)
    assert omitted in str(failure.value)


def test_current_loader_checks_dependency_set_again_after_preflight(
    tmp_path, monkeypatch
):
    from embodichain.gen_sim.task_engine import bundle_loading

    path, _, _ = fixture(tmp_path)
    original = bundle_loading._external_inputs
    calls = []

    def add_source_sidecar(names):
        values = original(names)
        calls.append(1)
        if len(calls) == 2:
            # USD source semantics are a real optional Data dependency. Adding
            # one does not change the already-hashed root layer bytes.
            (tmp_path / "dependency.twist.json").write_text("{}")
        return values

    monkeypatch.setattr(bundle_loading, "_external_inputs", add_source_sidecar)
    with pytest.raises(
        ValueError, match="external input set omits runtime dependencies"
    ) as failure:
        load_current_task_program_bundle(path)
    assert "dependency.twist.json" in str(failure.value)


def test_current_loaded_bundle_rechecks_external_input_before_environment_configuration(
    tmp_path,
):
    root, asset, _ = fixture(tmp_path)
    loaded = load_current_task_program_bundle(root)
    asset.write_bytes(b"changed after loading")
    with pytest.raises(ValueError, match="external input|input file"):
        loaded.configure_environment()


def test_current_input_guard_uses_sealed_sources_not_mutable_physical_view(tmp_path):
    path, _, _ = fixture(tmp_path)
    loaded = load_current_task_program_bundle(path)
    loaded.physical.config.clear()
    (tmp_path / "dependency.twist.json").write_text("{}")
    with pytest.raises(ValueError, match="omits runtime dependencies"):
        loaded.assert_current_inputs()


def test_current_input_guard_rechecks_late_dependency_after_second_hash(
    tmp_path, monkeypatch
):
    from embodichain.gen_sim.task_engine import bundle_loading

    path, _, _ = fixture(tmp_path)
    loaded = load_current_task_program_bundle(path)
    original = bundle_loading._external_inputs
    calls = []

    def add_late(names):
        value = original(names)
        calls.append(1)
        if len(calls) == 2:
            (tmp_path / "dependency.twist.json").write_text("{}")
        return value

    monkeypatch.setattr(bundle_loading, "_external_inputs", add_late)
    with pytest.raises(ValueError, match="omits runtime dependencies"):
        loaded.assert_current_inputs()


def test_current_input_guard_rejects_implementation_drift_after_load(
    tmp_path, monkeypatch
):
    from embodichain.gen_sim.task_engine import task_program_bundle

    path, _, _ = fixture(tmp_path)
    loaded = load_current_task_program_bundle(path)
    monkeypatch.setattr(
        task_program_bundle,
        "task_program_implementation_revisions",
        lambda: {"changed": "owner"},
    )
    with pytest.raises(ValueError, match="implementation changed after loading"):
        loaded.configure_environment()


def test_current_configuration_rechecks_inputs_changed_by_parser(tmp_path, monkeypatch):
    from embodichain.lab.gym.utils import gym_utils

    path, asset, _ = fixture(tmp_path)
    loaded = load_current_task_program_bundle(path)
    original = gym_utils.config_to_cfg

    def changed(*args, **kwargs):
        value = original(*args, **kwargs)
        asset.write_bytes(b"changed by configuration")
        return value

    monkeypatch.setattr(gym_utils, "config_to_cfg", changed)
    with pytest.raises(ValueError, match="input files changed after loading"):
        loaded.configure_environment()


def test_current_loaded_snapshot_keeps_component_bytes_after_disk_files_disappear(
    tmp_path,
):
    from gymnasium.envs.registration import registry
    from embodichain.lab.gym.utils.registration import REGISTERED_ENVS

    path, _, _ = fixture(tmp_path)
    loaded = load_current_task_program_bundle(path)
    for name in loaded.snapshot.files:
        (path / name).unlink()
    env_id = yaml.safe_load(loaded.snapshot.files["task_program_deployment.yaml"])["id"]
    try:
        actual, cfg = loaded.configure_environment()
        assert actual == env_id and cfg.num_envs == 1
    finally:
        registry.pop(env_id, None)
        REGISTERED_ENVS.pop(env_id, None)


def test_current_runner_configuration_rechecks_inputs_before_parser(
    tmp_path, monkeypatch
):
    from embodichain.gen_sim.task_engine import _bundle_runner
    from embodichain.lab.gym.utils import gym_utils

    root, asset, _ = fixture(tmp_path)
    loaded = load_current_task_program_bundle(root)
    args = _bundle_runner._runner_parser().parse_args([])
    asset.write_bytes(b"changed before launcher configuration")
    monkeypatch.setattr(
        gym_utils,
        "config_to_cfg",
        lambda *a, **k: pytest.fail("Drift reached environment parser"),
    )
    with pytest.raises(ValueError, match="input files changed after loading"):
        _bundle_runner._configured_environment(loaded, args, tmp_path / "run")
