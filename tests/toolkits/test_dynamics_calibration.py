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

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

import embodichain.toolkits.dynamics_calibration.cli as calibration_cli
from embodichain.toolkits.dynamics_calibration import (
    CalibrationConfig,
    EvaluationError,
    QualificationThresholds,
    build_drive_overlay,
    compute_tracking_metrics,
    load_overlay,
    qualify,
    resolve_control_schedule,
    run_candidate,
    tune_drive,
    write_overlay,
)


def _write_asset(path: Path) -> Path:
    path.write_text(
        """\
<robot name="synthetic">
  <link name="body">
    <inertial>
      <mass value="1"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>
</robot>
""",
        encoding="utf-8",
    )
    return path


def _config_dict(asset: Path, evaluator_target: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "assets": [str(asset)],
        "backend": "default",
        "physics_dt": 1.0 / 240.0,
        "control_frequency_hz": 60.0,
        "seed": 7,
        "candidate_count": 5,
        "evaluator": {
            "target": evaluator_target,
            "timeout_seconds": 5.0,
            "payload": {},
        },
        "parameters": [
            {
                "name": "arm_stiffness",
                "field": "stiffness",
                "selector": "arm",
                "lower": 0.0,
                "upper": 10.0,
                "initial": 0.0,
            }
        ],
        "qualification": {
            "aggregate_rmse_max": 0.25,
            "per_joint_rmse_max": 0.25,
            "control_frequency_relative_error_max": 0.0,
        },
    }


def _write_evaluator(path: Path, body: str) -> str:
    path.write_text(body, encoding="utf-8")
    return f"{path}:evaluate"


def test_control_frequency_requires_an_exact_physics_schedule() -> None:
    """Control timing cannot silently round to a different frequency."""
    schedule = resolve_control_schedule(physics_dt=1.0 / 240.0, requested_hz=60.0)

    assert schedule.physics_steps_per_control == 4
    assert schedule.actual_hz == pytest.approx(60.0)

    with pytest.raises(ValueError, match="cannot be represented"):
        resolve_control_schedule(physics_dt=1.0 / 240.0, requested_hz=100.0)

    approximate = resolve_control_schedule(
        physics_dt=1.0 / 240.0,
        requested_hz=100.0,
        allow_approximate=True,
    )
    assert approximate.physics_steps_per_control == 2
    assert approximate.actual_hz == pytest.approx(120.0)

    closer_ceiling = resolve_control_schedule(
        physics_dt=1.0 / 240.0,
        requested_hz=96.0,
        allow_approximate=True,
    )
    assert closer_ceiling.physics_steps_per_control == 3
    assert closer_ceiling.actual_hz == pytest.approx(80.0)


def test_configuration_rejects_values_that_yaml_would_otherwise_coerce(
    tmp_path: Path,
) -> None:
    """Boolean strings, fractional counts, and malformed parameters fail early."""
    asset = _write_asset(tmp_path / "robot.urdf")
    data = _config_dict(asset, "example.module:evaluate")
    data["allow_approximate_control_frequency"] = "false"
    with pytest.raises(TypeError, match="must be a boolean"):
        CalibrationConfig.from_dict(data, base_dir=tmp_path)

    data = _config_dict(asset, "example.module:evaluate")
    data["candidate_count"] = 1.5
    with pytest.raises(TypeError, match="must be an integer"):
        CalibrationConfig.from_dict(data, base_dir=tmp_path)

    data = _config_dict(asset, "example.module:evaluate")
    data["parameters"] = ["not-a-mapping"]
    with pytest.raises(TypeError, match=r"parameters\[0\] must be a mapping"):
        CalibrationConfig.from_dict(data, base_dir=tmp_path)


def test_one_bad_joint_fails_even_when_aggregate_gate_passes() -> None:
    """Per-joint hard gates prevent aggregate metrics from hiding failure."""
    target = np.zeros((100, 10), dtype=np.float64)
    actual = target.copy()
    actual[:, -1] = 1.0
    metrics = compute_tracking_metrics(
        {
            "joint_names": [f"j{index}" for index in range(10)],
            "target_qpos": target.tolist(),
            "actual_qpos": actual.tolist(),
            "requested_control_hz": 60.0,
            "actual_control_hz": 60.0,
            "target_qvel_write_count": 0,
        }
    )

    result = qualify(
        metrics,
        QualificationThresholds(
            aggregate_rmse_max=0.5,
            per_joint_rmse_max=0.5,
            control_frequency_relative_error_max=0.0,
        ),
    )

    assert metrics.aggregate_rmse < 0.5
    assert result.status == "fail"
    gate = next(item for item in result.gates if item.name == "per_joint_rmse")
    assert not gate.passed
    assert gate.entity == "j9"


def test_control_group_metrics_and_nonfinite_results_remain_hard_failures() -> None:
    """Group regressions and divergence remain explicit and JSON serializable."""
    metrics = compute_tracking_metrics(
        {
            "joint_names": ["left", "right"],
            "target_qpos": [[0.0, 0.0], [0.0, 0.0]],
            "actual_qpos": [[0.0, float("nan")], [0.0, 1.0]],
            "control_groups": {"arm": ["left", "right"]},
            "requested_control_hz": 60.0,
            "actual_control_hz": 60.0,
            "target_qvel_write_count": 0,
        }
    )
    result = qualify(
        metrics,
        QualificationThresholds(per_control_group_rmse_max=0.5),
    )

    assert not metrics.stable
    assert result.status == "fail"
    assert (
        next(
            gate for gate in result.gates if gate.name == "per_control_group_rmse"
        ).entity
        == "arm"
    )
    json.dumps(metrics.to_dict(), allow_nan=False)


def test_optional_application_metrics_and_velocity_saturation_are_gated() -> None:
    """Optional observations use the same strict, centralized hard-gate path."""
    metrics = compute_tracking_metrics(
        {
            "joint_names": ["joint"],
            "target_qpos": [[0.0], [0.1]],
            "actual_qpos": [[0.0], [0.1]],
            "qvel": [[0.9], [1.0]],
            "qvel_limits": [1.0],
            "overshoot": 0.2,
            "settling_time_seconds": 0.4,
            "requested_control_hz": 60.0,
            "actual_control_hz": 60.0,
            "target_qvel_write_count": 0,
        }
    )

    result = qualify(
        metrics,
        QualificationThresholds(
            velocity_saturation_fraction_max=0.25,
            overshoot_max=0.1,
            settling_time_seconds_max=0.5,
        ),
    )

    assert metrics.velocity_saturation_fraction == pytest.approx(0.5)
    assert result.status == "fail"
    assert {gate.name: gate.passed for gate in result.gates}["settling_time_seconds"]


def test_missing_qvel_instrumentation_is_not_assumed_to_be_zero() -> None:
    """The default qpos-only gate needs evidence, not an omitted field."""
    metrics = compute_tracking_metrics(
        {
            "joint_names": ["joint"],
            "target_qpos": [[0.0]],
            "actual_qpos": [[0.0]],
            "requested_control_hz": 60.0,
            "actual_control_hz": 60.0,
        }
    )

    result = qualify(metrics, QualificationThresholds())

    gate = next(item for item in result.gates if item.name == "target_qvel_write_count")
    assert metrics.target_qvel_write_count is None
    assert gate.observed is None
    assert not gate.passed
    assert result.status == "fail"


def test_overlay_round_trip_does_not_modify_source_asset(tmp_path: Path) -> None:
    """A tuned candidate is emitted as a reviewable overlay only."""
    asset = _write_asset(tmp_path / "robot.urdf")
    source_before = asset.read_bytes()
    config = CalibrationConfig.from_dict(
        _config_dict(asset, "example.module:evaluate"),
        base_dir=tmp_path,
    )
    overlay = build_drive_overlay(config, {"arm_stiffness": 5.0})
    output = tmp_path / "drive_overlay.yaml"

    write_overlay(output, overlay)

    assert load_overlay(output) == overlay
    assert yaml.safe_load(output.read_text(encoding="utf-8"))["drive_pros"] == {
        "stiffness": {"arm": 5.0}
    }
    assert asset.read_bytes() == source_before


def test_worker_exception_and_timeout_are_not_reported_as_success(
    tmp_path: Path,
) -> None:
    """Candidate infrastructure failures propagate instead of becoming metrics."""
    asset = _write_asset(tmp_path / "robot.urdf")
    raising_target = _write_evaluator(
        tmp_path / "raising.py",
        "def evaluate(overlay, context):\n    raise RuntimeError('boom')\n",
    )
    config = CalibrationConfig.from_dict(
        _config_dict(asset, raising_target), base_dir=tmp_path
    )

    with pytest.raises(EvaluationError, match="RuntimeError: boom"):
        run_candidate(
            config.evaluator,
            build_drive_overlay(config, {"arm_stiffness": 1.0}),
            config.evaluation_context("training"),
            cache_dir=tmp_path / "cache",
        )

    sleeping_target = _write_evaluator(
        tmp_path / "sleeping.py",
        "import time\ndef evaluate(overlay, context):\n    time.sleep(1)\n    return {}\n",
    )
    timeout_data = _config_dict(asset, sleeping_target)
    timeout_data["evaluator"]["timeout_seconds"] = 0.05  # type: ignore[index]
    timeout_config = CalibrationConfig.from_dict(timeout_data, base_dir=tmp_path)
    with pytest.raises(EvaluationError, match="timed out"):
        run_candidate(
            timeout_config.evaluator,
            build_drive_overlay(timeout_config, {"arm_stiffness": 1.0}),
            timeout_config.evaluation_context("training"),
            cache_dir=tmp_path / "timeout-cache",
        )


def test_tuning_improves_a_known_synthetic_perturbation(tmp_path: Path) -> None:
    """The deterministic search recovers a held-out synthetic drive optimum."""
    asset = _write_asset(tmp_path / "robot.urdf")
    evaluator_target = _write_evaluator(
        tmp_path / "synthetic.py",
        """\
def evaluate(overlay, context):
    value = overlay["drive_pros"]["stiffness"]["arm"]
    error = abs(value - 5.0) / 5.0
    target = [[0.0], [0.0], [0.0], [0.0]]
    return {
        "joint_names": ["joint"],
        "target_qpos": target,
        "actual_qpos": [[error] for _ in target],
        "requested_control_hz": context["requested_control_hz"],
        "actual_control_hz": context["actual_control_hz"],
        "target_qvel_write_count": 0,
    }
""",
    )
    config = CalibrationConfig.from_dict(
        _config_dict(asset, evaluator_target), base_dir=tmp_path
    )

    result = tune_drive(config, cache_dir=tmp_path / "cache")

    assert result.best_candidate == {"arm_stiffness": 5.0}
    assert result.best_objective < result.baseline_objective * 0.1
    repeated = tune_drive(config, cache_dir=tmp_path / "cache")
    assert [trial.candidate for trial in repeated.trials] == [
        trial.candidate for trial in result.trials
    ]
    assert repeated.best_candidate == result.best_candidate
    assert all(trial.cache_hit for trial in repeated.trials)
    held_out = run_candidate(
        config.evaluator,
        result.overlay,
        config.evaluation_context("qualification"),
        cache_dir=tmp_path / "cache",
    )
    assert qualify(held_out.metrics, config.qualification).status == "pass"


def test_candidate_cache_is_keyed_by_inputs(tmp_path: Path) -> None:
    """An identical asset/config/candidate reuses its isolated result."""
    asset = _write_asset(tmp_path / "robot.urdf")
    evaluator_target = _write_evaluator(
        tmp_path / "constant.py",
        """\
def evaluate(overlay, context):
    return {
        "joint_names": ["joint"],
        "target_qpos": [[0.0]],
        "actual_qpos": [[0.0]],
        "requested_control_hz": context["requested_control_hz"],
        "actual_control_hz": context["actual_control_hz"],
        "target_qvel_write_count": 0,
    }
""",
    )
    config = CalibrationConfig.from_dict(
        _config_dict(asset, evaluator_target), base_dir=tmp_path
    )
    overlay = build_drive_overlay(config, {"arm_stiffness": 5.0})
    cache_dir = tmp_path / "cache"

    first = run_candidate(
        config.evaluator,
        overlay,
        config.evaluation_context("training"),
        cache_dir=cache_dir,
    )
    second = run_candidate(
        config.evaluator,
        overlay,
        config.evaluation_context("training"),
        cache_dir=cache_dir,
    )

    assert not first.cache_hit
    assert second.cache_hit
    assert first.metrics.to_dict() == second.metrics.to_dict()
    assert len(list(cache_dir.glob("*.json"))) == 1


def test_cli_writes_overlay_and_complete_qualification_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public workflow joins audit, search, held-out gates, and artifacts."""
    asset = _write_asset(tmp_path / "robot.urdf")
    evaluator_target = _write_evaluator(
        tmp_path / "synthetic.py",
        """\
def evaluate(overlay, context):
    value = overlay["drive_pros"]["stiffness"]["arm"]
    error = abs(value - 5.0) / 5.0
    return {
        "joint_names": ["joint"],
        "target_qpos": [[0.0], [0.0]],
        "actual_qpos": [[error], [error]],
        "requested_control_hz": context["requested_control_hz"],
        "actual_control_hz": context["actual_control_hz"],
        "target_qvel_write_count": 0,
        "metadata": {"phase": context["phase"]},
    }
""",
    )
    config_path = tmp_path / "calibration.yaml"
    config_path.write_text(
        yaml.safe_dump(_config_dict(asset, evaluator_target)), encoding="utf-8"
    )

    class ReviewAudit:
        ready = True
        status = "review"

        def to_dict(self) -> dict[str, object]:
            return {
                "status": "review",
                "source": str(asset),
                "asset_sha256": "test-digest",
            }

        def to_markdown(self) -> str:
            return "# review audit\n"

    monkeypatch.setattr(
        calibration_cli,
        "audit_assets",
        lambda *_args, **_kwargs: (ReviewAudit(),),
    )
    output_dir = tmp_path / "output"

    calibration_cli.main(
        ["tune-drive", "--config", str(config_path), "--output-dir", str(output_dir)]
    )

    overlay = load_overlay(output_dir / "drive_overlay.yaml")
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert overlay["drive_pros"] == {"stiffness": {"arm": 5.0}}
    assert report["status"] == "review"
    assert report["claim"] == "effective_drive_tuning"
    assert report["qualification_evaluation"]["evaluator_metadata"] == {
        "phase": "qualification"
    }
    assert "does not claim" in (output_dir / "report.md").read_text(encoding="utf-8")


@pytest.mark.slow
@pytest.mark.requires_sim
@pytest.mark.subprocess_sim
def test_builtin_tracking_evaluator_runs_in_a_real_isolated_simulator(
    tmp_path: Path,
) -> None:
    """The shipped evaluator drives qpos only and returns simulator evidence."""
    asset = Path(__file__).parent / "assets" / "dynamics_calibration_pendulum.urdf"
    data = _config_dict(
        asset,
        "embodichain.toolkits.dynamics_calibration.tracking_evaluator:evaluate",
    )
    data["candidate_count"] = 1
    data["evaluator"] = {
        "target": "embodichain.toolkits.dynamics_calibration.tracking_evaluator:evaluate",
        "timeout_seconds": 30.0,
        "payload": {
            "control_part": "arm",
            "robot_cfg": {"control_parts": {"arm": ["joint1"]}},
            "trajectory": {
                "duration_seconds": 0.05,
                "warmup_seconds": 0.0,
                "amplitude": 0.05,
                "frequencies_hz": 1.0,
            },
        },
    }
    data["parameters"] = [
        {
            "name": "arm_stiffness",
            "field": "stiffness",
            "selector": "arm",
            "lower": 10.0,
            "upper": 100.0,
            "initial": 50.0,
        }
    ]
    config = CalibrationConfig.from_dict(data, base_dir=tmp_path)

    evaluation = run_candidate(
        config.evaluator,
        build_drive_overlay(config, {"arm_stiffness": 50.0}),
        config.evaluation_context("training"),
        cache_dir=tmp_path / "cache",
    )

    assert evaluation.metrics.sample_count == 3
    assert evaluation.metrics.stable
    assert evaluation.metrics.velocity_saturation_fraction is not None
    assert evaluation.metrics.target_qvel_write_count == 0
    assert evaluation.metadata["target_qvel_instrumentation"] == (
        "Articulation.set_qvel"
    )
