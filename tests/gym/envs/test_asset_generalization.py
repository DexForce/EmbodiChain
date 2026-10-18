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

"""Bounded asset orchestration and recording lifecycle tests with CPU ports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodichain.data.asset_catalog import (
    AssetCandidate,
    AssetDescriptor,
    AssetQuery,
    Bounds3D,
    PreparedAsset,
)
from embodichain.lab.gym.envs import asset_generalization as module
from embodichain.lab.sim.motion.expansion import ValidationCheck, ValidationResult
from embodichain.lab.task_program.integrations.asset_materialization import (
    MaterializedAssetDeployment,
)
from embodichain.lab.task_program.semantics.asset_requirements import (
    AssetRequirements,
    AssetRoleRequirement,
)
from embodichain.lab.task_program.semantics.asset_matching import (
    AssetResolutionError,
    ResolvedAssetSet,
)


def _artifact(root: Path, name: str, resolved: ResolvedAssetSet | None = None):
    folder = root / name
    folder.mkdir(parents=True)
    task = folder / "task.yaml"
    task.write_text("id: Fake-v0\n")
    for filename in (
        "program.yaml",
        "integration.yaml",
        "embodiment.yaml",
        "policy.yaml",
    ):
        (folder / filename).write_text("{}\n")
    if resolved is None:
        descriptor = AssetDescriptor(
            name,
            category="cube",
            bounds_m=Bounds3D((0, 0, 0), (0.1, 0.1, 0.1)),
            normalized=True,
        )
        asset = PreparedAsset(
            descriptor,
            "/cache/model.obj",
            "/cache/model.obj",
            hashlib.sha256(name.encode()).hexdigest(),
        )
        resolved = ResolvedAssetSet({"source": asset}, 0, "b" * 64)
    manifest = folder / "assets.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "asset_set_id": resolved.set_id,
                "files": {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in folder.glob("*.yaml")
                },
                "resolved_assets": resolved.to_dict(),
            }
        )
    )
    return MaterializedAssetDeployment(
        task,
        resolved.set_id,
        manifest,
        hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )


def _passed(env=None):
    return ValidationResult((ValidationCheck("measured", "passed"),))


class _Env:
    num_envs = 1

    def __init__(self, events, name):
        self.events = events
        self.name = name
        self._demo_episode_metadata = [{}]
        self.reset_error = None
        self.close_error = None
        self.snapshots = []

    def reset(self, *, seed=None, options):
        self.events.append(("reset", self.name, seed, dict(options)))
        self.snapshots.append(dict(self._demo_episode_metadata[0]))
        if self.reset_error is not None and "commit_env_ids" in options:
            raise self.reset_error
        self._demo_episode_metadata = [{}]

    def close(self, *, exit_process):
        assert exit_process is False
        self.events.append(("close", self.name))
        if self.close_error:
            raise self.close_error


@pytest.fixture
def ports(monkeypatch):
    events = []
    envs = []

    def factory(deployment, *, seed):
        events.append(("create", deployment.asset_set_id, seed))
        env = _Env(events, deployment.asset_set_id)
        envs.append(env)
        return env

    class Host:
        def __init__(self, env, *, initial_validator, settle_steps):
            self.validator = initial_validator
            self.env = env
            events.append(("host", env.name, settle_steps))

        def __call__(self, variant):
            assert variant.pose_changes == ()
            return self.validator()

    def execute(env, variant, *, prepare_scene, measured_validator, episode_index):
        events.append(("execute", env.name, episode_index))
        initial = prepare_scene(variant)
        measured = measured_validator() if initial.accepted else ValidationResult(())
        accepted = initial.accepted and measured.accepted
        return SimpleNamespace(
            accepted=accepted, to_metadata=lambda: {"accepted": accepted}
        )

    monkeypatch.setattr(module, "SimulationSceneExpansionHost", Host)
    monkeypatch.setattr(module, "execute_scene_variant", execute)
    monkeypatch.setattr(module, "_flush_cleanup", lambda: events.append(("flush",)))
    return SimpleNamespace(events=events, envs=envs, factory=factory)


def test_batches_close_and_flush_before_next_and_reset_before_repeat(tmp_path, ports):
    deployments = tuple(_artifact(tmp_path, f"batch_{i}") for i in range(2))
    outcomes = module.collect_asset_batches(
        deployments,
        cfg=module.AssetGeneralizationCfg(max_batches=2, episodes_per_batch=2, seed=17),
        initial_validator=_passed,
        measured_validator=_passed,
        environment_factory=ports.factory,
        report_path=tmp_path / "report.json",
    )
    assert all(item.close_finalized for item in outcomes)
    assert all(
        ep.accepted and ep.reset_submitted for item in outcomes for ep in item.episodes
    )
    names = [event[0] for event in ports.events]
    assert (
        names
        == [
            "create",
            "reset",
            "host",
            "execute",
            "reset",
            "execute",
            "reset",
            "close",
            "flush",
        ]
        * 2
    )
    assert [event[2] for event in ports.events if event[0] == "create"] == [17, 18]
    for env in ports.envs:
        for snapshot in env.snapshots[1:]:
            assert (
                snapshot["asset_generalization"]["manifest"]["asset_set_id"] == env.name
            )
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["status"] == "completed"
    assert report["batches"][0]["episodes"][1]["reset_submitted"] is True


@pytest.mark.parametrize(
    "initial,measured",
    [
        (ValidationResult(()), _passed()),
        (_passed(), ValidationResult(())),
        (_passed(), ValidationResult((ValidationCheck("task", "failed"),))),
    ],
)
def test_rejected_checks_always_discard(tmp_path, ports, initial, measured):
    outcome = module.collect_asset_batches(
        [_artifact(tmp_path, "batch")],
        cfg=module.AssetGeneralizationCfg(),
        initial_validator=lambda env: initial,
        measured_validator=lambda env: measured,
        environment_factory=ports.factory,
    )[0]
    assert outcome.episodes[0].accepted is False
    resets = [event for event in ports.events if event[0] == "reset"]
    assert resets[-1][-1] == {"save_data": False, "commit_env_ids": []}


def test_tampered_manifest_file_prevents_constructor(tmp_path, ports):
    deployment = _artifact(tmp_path, "batch")
    deployment.deployment_path.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        module.collect_asset_batches(
            [deployment],
            cfg=module.AssetGeneralizationCfg(),
            initial_validator=_passed,
            measured_validator=_passed,
            environment_factory=ports.factory,
        )
    assert ports.events == []


def test_constructor_failure_never_flushes_live_traceback(tmp_path, ports):
    failure = RuntimeError("constructor failed")

    def factory(deployment, *, seed):
        raise failure

    with pytest.raises(RuntimeError) as caught:
        module.collect_asset_batches(
            [_artifact(tmp_path, "batch")],
            cfg=module.AssetGeneralizationCfg(),
            initial_validator=_passed,
            measured_validator=_passed,
            environment_factory=factory,
            report_path=tmp_path / "report.json",
        )
    assert caught.value is failure
    assert ports.events == []
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["batches"][0]["close_finalized"] is False


def test_execution_error_remains_primary_when_close_also_fails(
    tmp_path, ports, monkeypatch
):
    failure = RuntimeError("execution failed")

    def execute(*args, **kwargs):
        raise failure

    def factory(deployment, *, seed):
        env = ports.factory(deployment, seed=seed)
        env.close_error = ValueError("close failed")
        return env

    monkeypatch.setattr(module, "execute_scene_variant", execute)
    with pytest.raises(RuntimeError) as caught:
        module.collect_asset_batches(
            [_artifact(tmp_path, "batch")],
            cfg=module.AssetGeneralizationCfg(),
            initial_validator=_passed,
            measured_validator=_passed,
            environment_factory=factory,
            report_path=tmp_path / "report.json",
        )
    assert caught.value is failure
    assert any("close failed" in note for note in caught.value.__notes__)
    assert ("flush",) not in ports.events
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["batches"][0]["close_error"]["detail"] == "close failed"


def test_reset_submission_failure_does_not_become_persistence_success(tmp_path, ports):
    failure = RuntimeError("submit failed")

    def factory(deployment, *, seed):
        env = ports.factory(deployment, seed=seed)
        env.reset_error = failure
        return env

    with pytest.raises(RuntimeError) as caught:
        module.collect_asset_batches(
            [_artifact(tmp_path, "batch")],
            cfg=module.AssetGeneralizationCfg(),
            initial_validator=_passed,
            measured_validator=_passed,
            environment_factory=factory,
            report_path=tmp_path / "report.json",
        )
    assert caught.value is failure
    report = json.loads((tmp_path / "report.json").read_text())
    episode = report["batches"][0]["episodes"][0]
    assert episode["accepted"] is True
    assert episode["reset_submitted"] is False
    assert report["batches"][0]["close_finalized"] is True
    assert ("flush",) not in ports.events


def test_close_failure_propagates_and_stops_next_batch(tmp_path, ports):
    failure = RuntimeError("durability failed")

    def factory(deployment, *, seed):
        env = ports.factory(deployment, seed=seed)
        env.close_error = failure
        return env

    with pytest.raises(RuntimeError) as caught:
        module.collect_asset_batches(
            [_artifact(tmp_path, "one"), _artifact(tmp_path, "two")],
            cfg=module.AssetGeneralizationCfg(max_batches=2),
            initial_validator=_passed,
            measured_validator=_passed,
            environment_factory=factory,
        )
    assert caught.value is failure
    assert len(ports.envs) == 1
    assert ("flush",) not in ports.events


def test_reporting_failure_does_not_replace_original_error(
    tmp_path, ports, monkeypatch
):
    failure = RuntimeError("factory failed")

    def factory(*args, **kwargs):
        raise failure

    def write(*args):
        raise OSError("report failed")

    monkeypatch.setattr(module, "_write_report", write)
    with pytest.raises(RuntimeError) as caught:
        module.collect_asset_batches(
            [_artifact(tmp_path, "batch")],
            cfg=module.AssetGeneralizationCfg(),
            initial_validator=_passed,
            measured_validator=_passed,
            environment_factory=factory,
            report_path=tmp_path / "report.json",
        )
    assert caught.value is failure
    assert any("report failed" in note for note in failure.__notes__)


def test_non_singleton_custom_factory_is_closed_without_execution(tmp_path, ports):
    def factory(deployment, *, seed):
        env = ports.factory(deployment, seed=seed)
        env.num_envs = 2
        return env

    with pytest.raises(ValueError, match="B=1"):
        module.collect_asset_batches(
            [_artifact(tmp_path, "batch")],
            cfg=module.AssetGeneralizationCfg(),
            initial_validator=_passed,
            measured_validator=_passed,
            environment_factory=factory,
        )
    assert [event[0] for event in ports.events] == ["create", "close"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": True},
        {"max_batches": 0},
        {"episodes_per_batch": -1},
        {"candidates_per_role": 1.2},
        {"max_search_nodes": 0},
        {"settle_steps": -1},
        {"seed": 2**32 - 1, "max_batches": 2},
    ],
)
def test_invalid_configuration_budgets(kwargs):
    with pytest.raises(ValueError):
        module.AssetGeneralizationCfg(**kwargs)


class _Provider:
    def __init__(self, candidate_ids):
        self.ids = candidate_ids
        self.queries = []
        self.materialized = []

    def candidates(self, request, *, limit):
        self.queries.append((request, limit))
        return [AssetCandidate(name, name) for name in self.ids]

    def materialize(self, candidate, *, context):
        self.materialized.append(candidate.candidate_id)
        if candidate.candidate_id == "bad":
            raise ValueError("missing bounds")
        if candidate.candidate_id == "invalid_return":
            return "wrong"
        descriptor = AssetDescriptor(
            candidate.candidate_id,
            category="cup",
            bounds_m=Bounds3D((0, 0, 0), (0.1, 0.1, 0.1)),
            normalized=True,
        )
        return PreparedAsset(
            descriptor, "/cache/model.obj", "/cache/model.obj", "a" * 64
        )


@pytest.fixture
def prepare_ports(monkeypatch, tmp_path):
    requirements = AssetRequirements(
        {"source": AssetRoleRequirement("source", AssetQuery("cup"))}
    )
    monkeypatch.setattr(
        module, "load_asset_requirements_from_deployment", lambda path: requirements
    )
    selected = []

    def materialize(task, contract, resolved, *, output_dir, adaptations):
        selected.append(resolved)
        return _artifact(Path(output_dir).parent, Path(output_dir).name, resolved)

    monkeypatch.setattr(module, "materialize_asset_deployment", materialize)
    return SimpleNamespace(requirements=requirements, selected=selected)


def test_preparation_retains_bad_candidate_diagnostics_and_reuses_snapshot(
    tmp_path, prepare_ports
):
    provider = _Provider(["bad", "invalid_return", "good"])
    deployments = module.prepare_asset_batches(
        "task.yaml",
        {"source": provider},
        cfg=module.AssetGeneralizationCfg(max_batches=2, seed=9),
        output_dir=tmp_path / "out",
    )
    assert len(deployments) == 2
    assert len(provider.queries) == 1
    assert provider.materialized == ["bad", "invalid_return", "good"]
    assert [result.seed for result in prepare_ports.selected] == [9, 10]
    report = json.loads((tmp_path / "out/preparation.report.json").read_text())
    assert report["roles"]["source"]["prepared"] == 1
    assert len(report["roles"]["source"]["rejections"]) == 2
    assert report["status"] == "prepared"


def test_empty_prepared_pool_preserves_role_and_matching_diagnostics(
    tmp_path, prepare_ports
):
    with pytest.raises(AssetResolutionError):
        module.prepare_asset_batches(
            "task.yaml",
            {"source": _Provider(["bad"])},
            cfg=module.AssetGeneralizationCfg(),
            output_dir=tmp_path / "out",
        )
    report = json.loads((tmp_path / "out/preparation.report.json").read_text())
    assert report["status"] == "failed"
    assert report["roles"]["source"]["rejections"][0]["detail"] == "missing bounds"
    assert report["matching_rejections"][0]["role"] == "source"


def test_provider_over_budget_does_not_materialize_unbounded_candidates(
    tmp_path, prepare_ports
):
    provider = _Provider(["a", "b"])
    with pytest.raises(AssetResolutionError):
        module.prepare_asset_batches(
            "task.yaml",
            {"source": provider},
            cfg=module.AssetGeneralizationCfg(candidates_per_role=1),
            output_dir=tmp_path / "out",
        )
    assert provider.materialized == []


def test_alias_query_broadens_category_without_relaxing_final_match(
    tmp_path, prepare_ports, monkeypatch
):
    requirements = AssetRequirements(
        prepare_ports.requirements.roles, category_aliases={"mug": "cup"}
    )
    monkeypatch.setattr(
        module, "load_asset_requirements_from_deployment", lambda path: requirements
    )
    provider = _Provider(["good"])
    module.prepare_asset_batches(
        "task.yaml",
        {"source": provider},
        cfg=module.AssetGeneralizationCfg(),
        output_dir=tmp_path / "out",
    )
    assert provider.queries[0][0].category is None
    assert prepare_ports.selected[0].assets["source"].descriptor.category == "cup"


def test_default_factory_isolates_recorders_across_batches_and_repeated_runs(
    tmp_path, monkeypatch
):
    from copy import deepcopy

    from embodichain.lab.gym.utils import gym_utils, registration
    from embodichain.lab.gym.envs.managers.record import record_camera_data

    template = SimpleNamespace(
        num_envs=1,
        seed=0,
        record_trajectory=True,
        trajectory_auto_save=True,
        trajectory_save_dir="/shared/trajectories",
        dataset=SimpleNamespace(
            recorder=SimpleNamespace(mode="save", params={"save_path": "/shared/data"}),
            loader=SimpleNamespace(mode="load", params={"save_path": "/read/data"}),
        ),
        events=SimpleNamespace(
            camera=SimpleNamespace(
                func=record_camera_data, params={"save_path": "/shared/video"}
            )
        ),
    )
    constructed = []
    monkeypatch.setattr(
        gym_utils, "config_to_cfg", lambda *args, **kwargs: deepcopy(template)
    )

    def build(env_id, cfg):
        constructed.append(cfg)
        return SimpleNamespace(cfg=cfg)

    monkeypatch.setattr(registration, "build_env", build)
    one = _artifact(tmp_path, "one")
    two = _artifact(tmp_path, "two")
    envs = [module._create_environment(item, seed=13) for item in (one, two, one)]
    assert len({env._asset_generalization_recording_root for env in envs}) == 3
    assert len({cfg.trajectory_save_dir for cfg in constructed}) == 3
    assert len({cfg.dataset.recorder.params["save_path"] for cfg in constructed}) == 3
    assert len({cfg.events.camera.params["save_path"] for cfg in constructed}) == 3
    assert all(
        cfg.dataset.loader.params["save_path"] == "/read/data" for cfg in constructed
    )
    assert template.trajectory_save_dir == "/shared/trajectories"
    assert template.dataset.recorder.params["save_path"] == "/shared/data"


def test_default_factory_rejects_configured_batch_size_before_constructing(
    tmp_path, monkeypatch
):
    from embodichain.lab.gym.utils import gym_utils, registration

    monkeypatch.setattr(
        gym_utils, "config_to_cfg", lambda *args, **kwargs: SimpleNamespace(num_envs=2)
    )
    constructed = []
    monkeypatch.setattr(
        registration, "build_env", lambda *args: constructed.append(args)
    )
    with pytest.raises(ValueError, match="num_envs=1"):
        module._create_environment(_artifact(tmp_path, "batch"), seed=0)
    assert constructed == []


def test_default_factory_applies_recording_flag_after_production_config_loading(
    tmp_path, monkeypatch
):
    import yaml

    from embodichain.lab.gym.utils import registration
    from embodichain.utils.config_paths import resolve_config_path

    repository = Path(__file__).resolve().parents[3]
    source = (
        repository
        / "extensions/embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml"
    )
    config = yaml.safe_load(source.read_text())
    config["id"] = "AssetFactoryRecordingFlagTest-v0"
    config.pop("expansion", None)
    config["environment"] = {"component": config["environment"]["default"]}
    for key in ("environment", "embodiment"):
        config[key]["component"] = str(
            resolve_config_path(config[key]["component"], base_dir=source.parent)
        )
    for key in ("program", "integration", "execution_policy"):
        config["task_program"][key] = str(
            resolve_config_path(config["task_program"][key], base_dir=source.parent)
        )
    deployment = _artifact(tmp_path, "batch")
    deployment.deployment_path.write_text(yaml.safe_dump(config))
    constructed = []

    def build(env_id, env_cfg):
        constructed.append(env_cfg)
        return SimpleNamespace(cfg=env_cfg)

    monkeypatch.setattr(registration, "build_env", build)
    module._create_environment(deployment, seed=18, record_trajectory=True)
    assert constructed[0].record_trajectory is True
    assert constructed[0].trajectory_auto_save is True
    assert Path(constructed[0].trajectory_save_dir).is_relative_to(
        deployment.deployment_path.parent / "recordings"
    )
    module._create_environment(deployment, seed=18)
    assert constructed[1].record_trajectory is False


def test_recording_flag_rejects_truthy_non_boolean():
    with pytest.raises(TypeError, match="record_trajectory"):
        module.AssetGeneralizationCfg(record_trajectory=1)
