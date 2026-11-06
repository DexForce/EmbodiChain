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
from dataclasses import replace
import json
import queue
from types import SimpleNamespace
import pytest
import yaml

from embodichain.gen_sim.task_engine import _bundle_runner as runner
from .test_runtime_evidence import _session, _boundary, _episode


def test_old_bundle_rejects_before_launcher_and_output_creation(tmp_path, monkeypatch):
    bundle = tmp_path / "old"
    bundle.mkdir()
    (bundle / "integration_fingerprint.json").write_text(
        json.dumps(dict(schema_version="semantic_integration_fingerprint/v2"))
    )
    monkeypatch.setattr(
        runner,
        "_runner_parser",
        lambda: pytest.fail("Old envelope reached the launcher"),
    )
    output = tmp_path / "run"
    with pytest.raises(
        ValueError, match="rebuild|Rebuild|regenerate|unsupported|Unsupported|version"
    ):
        runner.execute_bundle(bundle, execution_output=output)
    assert not output.exists()


def test_launcher_uses_sealed_components_without_runtime_creation(
    tmp_path, monkeypatch
):
    from embodichain.lab.sim import SimulationManager
    import embodichain.data as data
    from gymnasium.envs.registration import registry
    from embodichain.lab.gym.utils.registration import REGISTERED_ENVS

    session, bundle, env, bridge, binding = _session(tmp_path, monkeypatch)
    session.close()
    original = dict(bundle.snapshot.files)
    monkeypatch.setattr(data, "get_data_path", lambda value, **kwargs: value)
    monkeypatch.setattr(
        SimulationManager,
        "__init__",
        lambda *a, **k: pytest.fail("Launcher constructed simulation"),
    )
    args = runner._runner_parser().parse_args(
        ["--seed", "7", "--headless", "--filter_dataset_saving"]
    )
    env_id = yaml.safe_load(bundle.snapshot.files["task_program_deployment.yaml"])["id"]
    try:
        actual, cfg = runner._configured_environment(bundle, args, tmp_path / "run")
        assert actual == env_id and cfg.seed == 7 and cfg.num_envs == 1
        assert cfg.sim_cfg.headless and cfg.filter_dataset_saving
        assert cfg.task_program.program_id == bundle.program.program_id
        assert bundle.snapshot.files == original and not (tmp_path / "run").exists()
        args.num_envs = 2
        with pytest.raises(ValueError, match="num_envs=1"):
            runner._configured_environment(bundle, args, tmp_path / "batch")
    finally:
        registry.pop(env_id, None)
        REGISTERED_ENVS.pop(env_id, None)


def _worker_fixture(
    tmp_path, monkeypatch, *, accepted=True, fault=None, bind_goal=True
):
    import gymnasium
    from embodichain.gen_sim.task_engine import runtime_bindings
    from embodichain.lab.gym.envs import demo
    from embodichain.lab.sim import SimulationManager

    session, bundle, env, bridge, binding = _session(tmp_path, monkeypatch)
    session.close()
    events = []
    state = SimpleNamespace(pending=False, committed=0, discarded=0, destroyed=0)
    observers = []
    monkeypatch.setattr(SimulationManager, "_cleanup_queue", queue.Queue())
    monkeypatch.setattr(SimulationManager, "wait_scene_destruction", lambda: None)
    original_flush = SimulationManager.flush_cleanup_queue

    def forbidden_simulation(*args, **kwargs):
        pytest.fail("Worker software contract constructed simulation.")

    monkeypatch.setattr(SimulationManager, "__init__", forbidden_simulation)

    def flush():
        events.append(("flush",))
        if fault == "flush":
            raise RuntimeError("controlled deferred cleanup failure")
        original_flush()

    monkeypatch.setattr(SimulationManager, "flush_cleanup_queue", flush)

    def deferred_destroy():
        state.destroyed += 1
        events.append(("deferred_destroy",))

    def queue_destroy():
        # Exercise the actual public deferred-destruction handoff without a world.
        manager = SimpleNamespace(
            stop_visualization=lambda: None, _deferred_destroy=deferred_destroy
        )
        SimulationManager.destroy(manager, exit_process=False)

    def reset(**kwargs):
        events.append(("reset", kwargs))
        if state.pending:
            if kwargs.get("options", {}).get("save_data", True):
                if fault == "commit":
                    raise RuntimeError("controlled episode commit failure")
                state.committed += 1
            else:
                state.discarded += 1
            state.pending = False

    def close(**kwargs):
        events.append(("close", kwargs))
        if state.pending:
            state.discarded += 1
            state.pending = False
        queue_destroy()
        if fault == "close":
            raise RuntimeError("controlled recorder finalization failure")

    def save_trajectory(path, **kwargs):
        events.append(("trajectory", path))
        if fault == "capture":
            raise RuntimeError("controlled terminal capture failure")

    env.device = "cpu"
    env.step_dt = 0.04
    env.robot = SimpleNamespace(uid="DualFrankaPanda")
    env.reset, env.close, env.save_trajectory = reset, close, save_trajectory
    camera = SimpleNamespace(
        save_and_clear=lambda **kwargs: events.append(("camera", kwargs))
    )
    env.event_manager = SimpleNamespace(
        get_functor_cfg=lambda name: SimpleNamespace(func=camera)
    )

    def segments(**kwargs):
        observers.append(kwargs.get("initial_plan_observer"))
        return SimpleNamespace(close=lambda: events.append(("segments_closed",)))

    env.create_demo_segments = segments

    def make(**kwargs):
        if fault == "construct":
            queue_destroy()
            raise RuntimeError("controlled partial environment construction failure")
        return SimpleNamespace(unwrapped=env)

    monkeypatch.setattr(gymnasium, "make", make)
    monkeypatch.setattr(runner, "load_current_task_program_bundle", lambda path: bundle)
    monkeypatch.setattr(
        runner, "_configured_environment", lambda *args: ("test", SimpleNamespace())
    )
    if bind_goal:

        class Projector:
            control_operation_ids = ()
            unavailable = {}

            def __init__(self, *args, **kwargs):
                pass

            def prepare(self, session):
                return []

            def finish(self, session, result):
                return [binding.to_dict()]

        monkeypatch.setattr(runtime_bindings, "BundleEvidenceProjector", Projector)

    def execute(actual, **kwargs):
        assert actual is env
        assert len(observers) == 1 and callable(observers[0])
        observers[0](
            dict(
                scope="initial_call",
                planning_source="execution",
                plan_success=[accepted],
                task_success=None,
            )
        )
        events.append(("installed_plan_observed",))
        boundaries = []
        state.pending = True
        segment_count = len(tuple(bridge.program.iter_segments()))
        for index in range(segment_count if accepted else 1):
            bridge.clock.advance_after_env_step()
            boundary = _boundary(bridge, index, success=accepted)
            metadata = boundary.to_metadata()["metadata"]
            metadata["segment_count"] = segment_count
            call = metadata["runtime"]["calls"][0]
            call["status"] = "completed" if accepted else "failed"
            call["effects"][0]["decision"]["success_mask"] = [accepted]
            call["plan_attempts"] = [dict(plan_success_mask=[accepted])]
            boundary = replace(boundary, metadata=metadata)
            kwargs["on_segment_completed"](boundary)
            boundaries.append(boundary)
            if fault == "cancel":
                raise KeyboardInterrupt("controlled operator cancellation")
        return _episode(boundaries, success=accepted)

    monkeypatch.setattr(demo, "execute_demo_episode", execute)
    return SimpleNamespace(
        env=env,
        bundle=bundle,
        events=events,
        state=state,
        observers=observers,
        output=tmp_path / "run",
        cleanup_queue=SimulationManager._cleanup_queue,
    )


def test_normal_worker_preserves_program_unavailable_and_fresh_output(
    tmp_path, monkeypatch
):
    case = _worker_fixture(tmp_path, monkeypatch, bind_goal=False)
    assert runner.execute_bundle(case.bundle.root, execution_output=case.output) == 2
    report = json.loads((case.output / "execution_report.json").read_text())
    physical = report["runtime_result"]["physical_run_report"]
    assert physical["program_success"] == [True]
    assert physical["strict_task_spec"][0]["status"] == "unavailable"
    assert report["environments"][0]["success"] is False
    assert report["environments"][0]["terminal_reason"] == "task_spec_unverified"
    assert report["record_dir"] == str(case.output)
    assert (
        json.loads((case.output / "execution_attempt.json").read_text())["run_id"]
        == physical["run_id"]
    )
    archived = json.loads((case.output / "raw_evidence.json").read_text())
    assert archived["schema_version"] == "task_engine.raw_evidence/v1"
    assert archived["bundle_digest"] == case.bundle.snapshot.identity.digest
    assert archived["run_id"] == physical["run_id"]
    assert archived["bindings"] == []
    assert archived["measurements"] == json.loads(
        (case.output / "recorded_measurements.json").read_text()
    )
    assert not (case.bundle.root / "execution_report.json").exists()
    assert [name for name, *_ in case.events].count("close") == 1
    before = (case.output / "execution_report.json").read_bytes()
    with pytest.raises(FileExistsError):
        runner.execute_bundle(case.bundle.root, execution_output=case.output)
    assert (case.output / "execution_report.json").read_bytes() == before
    with pytest.raises(ValueError, match="sealed bundle"):
        runner.execute_bundle(
            case.bundle.root, execution_output=case.bundle.root / "execution"
        )


@pytest.mark.parametrize(
    "accepted,fault",
    [
        (True, None),
        (False, None),
        (True, "commit"),
        (True, "close"),
        (True, "capture"),
        (True, "flush"),
        (True, "construct"),
    ],
)
def test_worker_run_transaction_keeps_exit_report_and_saved_facts_consistent(
    tmp_path, monkeypatch, accepted, fault
):
    case = _worker_fixture(tmp_path, monkeypatch, accepted=accepted, fault=fault)
    succeeded = accepted and fault is None
    assert runner.execute_bundle(case.bundle.root, execution_output=case.output) == (
        0 if succeeded else 2
    )
    report = json.loads((case.output / "execution_report.json").read_text())
    attempt = json.loads((case.output / "execution_attempt.json").read_text())
    summary = json.loads((case.output / "program_completion_summary.json").read_text())
    assert attempt["status"] == ("succeeded" if succeeded else "failed")
    assert attempt["exit_code"] == (0 if succeeded else 2)
    assert report["status"] == (
        "succeeded" if succeeded else "rejected" if not accepted else "failed"
    )
    assert report["environments"][0]["success"] is succeeded
    assert summary["strict_execution_success_by_env"] == [succeeded]
    assert summary["experimental_program_completed_by_env"] == [succeeded]
    if fault != "construct":
        physical = json.loads((case.output / "physical_run_report.json").read_text())
        assert report["runtime_result"]["physical_run_report"] == physical
        assert physical["program_success"] == [accepted]
        assert physical["run_id"] == attempt["run_id"]
        archived = json.loads((case.output / "raw_evidence.json").read_text())
        assert archived["run_id"] == physical["run_id"]
        assert archived["episode_ids"] == physical["episode_ids"]
        assert len(archived["bindings"]) == 1
        assert archived["bindings"][0]["predicate_id"] == "now"
        assert archived["bindings"][0]["binding_hash"]
        assert archived["measurements"] == json.loads(
            (case.output / "recorded_measurements.json").read_text()
        )
        assert (
            report["failure"] is None
            if fault is None
            else report["failure"] is not None
        )
        if fault != "capture":
            exported = next(
                i for i, event in enumerate(case.events) if event[0] == "trajectory"
            )
            finalized = next(
                i for i, event in enumerate(case.events[1:], 1) if event[0] == "reset"
            )
            assert exported < finalized
    if not accepted:
        assert report["environments"][0]["terminal_reason"] == "initial_plan_rejected"
    assert case.state.committed == int(accepted and fault in (None, "close", "flush"))
    assert case.state.discarded == int(
        fault != "construct" and not case.state.committed
    )
    assert case.state.destroyed == (0 if fault == "flush" else 1)
    assert case.cleanup_queue.qsize() == (1 if fault == "flush" else 0)


@pytest.mark.parametrize(
    "filename", ["raw_evidence.json", "recorded_measurements.json"]
)
def test_existing_raw_archive_rejects_before_starting_another_run(
    tmp_path, monkeypatch, filename
):
    case = _worker_fixture(tmp_path, monkeypatch)
    case.output.mkdir()
    original = b'{"original_run":true}\n'
    (case.output / filename).write_bytes(original)
    with pytest.raises(FileExistsError, match="Historical execution artifacts"):
        runner.execute_bundle(case.bundle.root, execution_output=case.output)
    assert (case.output / filename).read_bytes() == original
    assert not any(name == "construct" for name, *_ in case.events)


def test_worker_cancellation_discards_partial_run_and_finishes_owned_cleanup(
    tmp_path, monkeypatch
):
    case = _worker_fixture(tmp_path, monkeypatch, fault="cancel")
    assert runner.execute_bundle(case.bundle.root, execution_output=case.output) == 2
    report = json.loads((case.output / "execution_report.json").read_text())
    attempt = json.loads((case.output / "execution_attempt.json").read_text())
    assert report["status"] == "failed"
    assert report["failure"]["type"] == "KeyboardInterrupt"
    assert report["environments"][0]["success"] is False
    assert attempt["status"] == "failed" and attempt["exit_code"] == 2
    assert case.state.committed == 0 and case.state.discarded == 1
    assert any(event[0] == "trajectory" for event in case.events)
    assert case.state.destroyed == 1 and case.cleanup_queue.empty()


def _bundle_with_native_trial_preset(tmp_path, *, used):
    from .test_bundle_deployment import _files, _prepare, _load
    from embodichain.gen_sim.task_engine.bundle_io import write_bundle_atomic
    from embodichain.lab.gym.utils.composition import resolve_gym_components
    from embodichain.lab.task_program.integrations import (
        load_configured_task_program_deployment,
    )

    parsed = {name: yaml.safe_load(value) for name, value in _files().items()}
    integration = parsed["task_program/integration.yaml"]
    policy = parsed["components/execution_policy.yaml"]
    options = deepcopy(integration["profile"]["action_options"]["pick"])
    options["native_scene_trial"] = True
    policy["presets"]["unused_trial"] = {}
    integration["profile"].setdefault("preset_options", {})["unused_trial"] = {
        "action_options": {"pick": options}
    }
    if used:
        first = next(iter(policy["presets"]))
        integration["profile"]["preset_options"][first] = {
            "action_options": {"pick": deepcopy(options)}
        }
    physical = resolve_gym_components(
        parsed["task_program_deployment.yaml"],
        base_dir=tmp_path,
        component_sources=parsed,
    )
    deployment = load_configured_task_program_deployment(
        task_program={
            key: {"inline": parsed[name]}
            for key, name in parsed["task_program_deployment.yaml"][
                "task_program"
            ].items()
        },
        skill_profile=physical.embodiment_skill_profile,
        base_dir=tmp_path,
    )
    parsed["semantic_task_graph.json"][
        "integration_fingerprint"
    ] = deployment.integration.integration_fingerprint
    files = {
        name: (
            yaml.safe_dump(value, sort_keys=False)
            if name.endswith(".yaml")
            else json.dumps(value)
        ).encode()
        for name, value in parsed.items()
    }
    return _load(write_bundle_atomic(tmp_path / "bundle", _prepare(files, tmp_path)))


@pytest.mark.parametrize("used", [False, True])
def test_worker_admission_checks_effective_native_trial_occurrences_only(
    tmp_path, monkeypatch, used
):
    bundle = _bundle_with_native_trial_preset(tmp_path, used=used)
    monkeypatch.setattr(runner, "load_current_task_program_bundle", lambda path: bundle)
    configured = []

    def stop_before_runtime(*args):
        configured.append(True)
        raise ValueError("software static configuration boundary")

    monkeypatch.setattr(runner, "_configured_environment", stop_before_runtime)
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="experiment" if used else "software static"):
        runner.execute_bundle(
            bundle.root, execution_output=output, acceptance_mode="strict"
        )
    assert configured == ([] if used else [True])
    assert not output.exists()


@pytest.mark.parametrize("outcome", [True, False, "error"])
def test_probe_only_preserves_proposal_exit_without_executing_or_exporting_empty_run(
    tmp_path, monkeypatch, outcome
):
    from embodichain.lab.task_program.integrations.simulation import planning_probe

    case = _worker_fixture(tmp_path, monkeypatch)
    probes = []

    def probe(environment, deployment):
        probes.append(True)
        if outcome == "error":
            raise RuntimeError("controlled explicit probe failure")
        return dict(
            scope="initial_call",
            planning_source="probe",
            task_success=None,
            plan_success=[outcome],
        )

    monkeypatch.setattr(
        planning_probe, "probe_configured_task_program_initial_plan", probe
    )
    expected = 0 if outcome is True else 2
    assert (
        runner.execute_bundle(
            case.bundle.root, ("--plan-probe-only",), execution_output=case.output
        )
        == expected
    )
    assert probes == [True] and case.observers == []
    resets = [event for event in case.events if event[0] == "reset"]
    assert len(resets) == 1 and resets[0][1]["options"] == {"save_data": False}
    assert not any(
        event[0] in {"trajectory", "camera", "installed_plan_observed"}
        for event in case.events
    )
    for filename in (
        "execution_report.json",
        "physical_run_report.json",
        "recorded_measurements.json",
        "raw_evidence.json",
        "trajectory.pt",
    ):
        assert not (case.output / filename).exists()
    saved = json.loads((case.output / "planning_probe.json").read_text())
    assert saved["scope"] == "initial_call" and saved["task_success"] is None
    if outcome != "error":
        assert saved["planning_source"] == "probe"
    assert saved["plan_success"] == ([] if outcome == "error" else [outcome])
    assert bool(saved.get("failure")) is (outcome == "error")
    attempt = json.loads((case.output / "execution_attempt.json").read_text())
    assert attempt["exit_code"] == expected
    assert case.state.destroyed == 1 and case.cleanup_queue.empty()


@pytest.mark.parametrize("drift_boundary", ["reset", "probe"])
def test_probe_cannot_succeed_under_the_old_identity_after_actual_input_drift(
    tmp_path, monkeypatch, drift_boundary
):
    import torch

    from .test_bundle_loading import fixture
    from embodichain.lab.task_program.integrations.simulation import planning_probe

    root, _, _ = fixture(tmp_path)
    mesh = tmp_path / "mesh.glb"
    probes = []

    def drift():
        mesh.write_bytes(mesh.read_bytes() + b"changed during runtime preparation")

    def reset(**kwargs):
        if drift_boundary == "reset":
            drift()

    def probe(environment, deployment):
        probes.append(True)
        if drift_boundary == "probe":
            drift()
        return dict(scope="initial_call", plan_success=[True], task_success=None)

    environment = SimpleNamespace(
        device=torch.device("cpu"),
        num_envs=1,
        reset=reset,
        close=lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "gymnasium.make", lambda **kwargs: SimpleNamespace(unwrapped=environment)
    )
    monkeypatch.setattr(
        planning_probe, "probe_configured_task_program_initial_plan", probe
    )
    output = tmp_path / "drifted_probe"
    assert (
        runner.execute_bundle(root, ("--plan-probe-only",), execution_output=output)
        == 2
    )
    assert probes == ([] if drift_boundary == "reset" else [True])
    saved = json.loads((output / "planning_probe.json").read_text())
    assert "external input files changed after loading" in saved["failure"]["message"]
    assert not (output / "physical_run_report.json").exists()
