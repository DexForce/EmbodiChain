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

"""Evidence-based comparison reports with seed-level confidence intervals."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from ._comparison import _read_plan, _json
from ._qualification import paired_noninferiority
from ._reference import OFFICIAL_COMMIT, _hash

__all__ = ["write_comparison_report", "annotate_rollout_evidence"]

_PARITY_REFERENCES = {
    OFFICIAL_COMMIT: "official_warp_main",
    "476dc67fccc6a99be293bf611f897b1c06f13e30": "newton_preview",
}
_HORIZONS = (1, 10, 100, 1000)


def _finite_nonnegative(value: Any, maximum: float | None = None) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
        and (maximum is None or value <= maximum)
    )


def _sha_identity(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _numerical_evidence(path: str | Path, robot: str) -> dict[str, Any]:
    """Read independent runner measurements instead of accepting bare pass flags."""
    file = Path(path)
    result = {
        **json.loads(file.read_text()),
        "path": str(file.resolve()),
        "sha256": _hash(file),
    }
    groups = {
        "preprocessing": (
            "max_codec_feature_error",
            "max_codec_target_error",
            "max_codec_decode_error",
        ),
        "forward": ("max_forward_error",),
        "parameter_gradients": ("max_parameter_gradient_error",),
        "input_gradients": ("max_input_gradient_error",),
        "adam_update": ("max_adam_parameter_error",),
    }
    required = {field for fields in groups.values() for field in fields}
    required.update(
        {
            "robot",
            "reference_commit",
            "checkpoint_sha256",
            "codec_fixtures_sha256",
            "history_lengths",
            "adam_updates",
            "precision",
            "atol",
            "rtol",
        }
    )
    if not required.issubset(result):
        return {
            **result,
            "status": "pending",
            "reason": "Numerical report lacks measured values or reference identity",
        }
    identity_matches = (
        str(result["robot"]).lower() == robot
        and isinstance(result["reference_commit"], str)
        and result["reference_commit"] in _PARITY_REFERENCES
        and _sha_identity(result["checkpoint_sha256"])
        and _sha_identity(result["codec_fixtures_sha256"])
        and result["precision"] == "float32"
        and isinstance(result["history_lengths"], list)
        and {1, 9, 10}.issubset(result["history_lengths"])
        and isinstance(result["adam_updates"], int)
        and result["adam_updates"] >= 1
        and _finite_nonnegative(result["atol"], 1e-5)
        and _finite_nonnegative(result["rtol"], 1e-4)
    )
    if not identity_matches:
        return {
            **result,
            "status": "failed",
            "reason": "Numerical reference, robot, fixtures or tolerance differs from qualification contract",
        }
    checks = {}
    for name, fields in groups.items():
        finite = all(_finite_nonnegative(result[field]) for field in fields)
        # A maximum absolute error below atol is sufficient for every reference
        # magnitude. Larger residuals need per-element relative-error evidence;
        # a caller-supplied Boolean is not that evidence.
        checks[name] = {
            "passed": finite
            and all(result[field] <= result["atol"] for field in fields),
            "max_abs": max(result[field] for field in fields) if finite else None,
        }
    if any(item["max_abs"] is None for item in checks.values()):
        status, reason = "failed", "Nonfinite or invalid numerical measurements"
    elif all(item["passed"] for item in checks.values()):
        status, reason = (
            "passed",
            "Measured absolute errors satisfy the declared tolerance",
        )
    else:
        status, reason = (
            "pending",
            "Residuals need per-element relative-tolerance evidence",
        )
    return {
        **result,
        "reference_kind": _PARITY_REFERENCES[result["reference_commit"]],
        "checks": checks,
        "status": status,
        "reason": reason,
    }


def _rollout_evidence(
    record: dict[str, Any] | None,
    cell: dict[str, Any],
    robot: str,
    *,
    minimum_trajectories: int = 2048,
    required_horizons: tuple[int, ...] = _HORIZONS,
) -> tuple[str, str]:
    """Validate an evidence scope; formal reports always use the fixed defaults."""
    required = {
        "robot",
        "motion_mode",
        "state_dtype",
        "teacher_origin",
        "teacher_split",
        "selected_teacher_sha256",
        "trajectories",
        "horizons",
        "failure_rate",
        "nonfinite_rate",
        "errors",
        "model_sha256",
        "model_manifest_sha256",
        "source_report",
        "source_sha256",
        "trace_evidence",
        "trace_evidence_sha256",
        "trajectory_evidence",
        "trajectory_evidence_sha256",
        "selected_teacher_hash_schema",
        "runtime_identity",
    }
    if record is None or not required.issubset(record):
        return "pending", "Missing rollout measurements or provenance"
    if "output_dir" not in cell:
        return "pending", "Training cell has no exported model"
    if (
        record["robot"] != robot
        or record["state_dtype"] != "float32"
        or not isinstance(record["teacher_origin"], dict)
        or record["teacher_origin"].get("kind") != cell["teacher"]
        or record["teacher_split"] != "test"
    ):
        return "failed", "Rollout robot, teacher domain or test split differs"
    if (
        not isinstance(record["trajectories"], int)
        or isinstance(record["trajectories"], bool)
        or record["trajectories"] < minimum_trajectories
        or not isinstance(record["horizons"], list)
        or not set(required_horizons).issubset(record["horizons"])
    ):
        return "pending", "Required rollout population or horizons incomplete"
    if not _sha_identity(record["selected_teacher_sha256"]):
        return "failed", "Missing selected-trajectory content identity"
    if record["selected_teacher_hash_schema"] != "ordered-teacher-content-v1":
        return "failed", "Unsupported selected-trajectory content schema"
    runtime = record["runtime_identity"]
    if (
        not isinstance(runtime, dict)
        or not isinstance(runtime.get("device"), str)
        or runtime.get("state_dtype") != "float32"
        or not runtime.get("python")
        or not isinstance(runtime.get("versions"), dict)
        or not all(runtime["versions"].get(name) for name in ("torch", "warp-lang"))
    ):
        return "failed", "Actual neural execution identity is incomplete"
    if cell["teacher"] == "official_warp":
        expected_warp = "1.5.1" if robot == "ant" else "1.8.0"
        if (
            runtime.get("engine") != "official_warp"
            or runtime.get("reference_commit") != OFFICIAL_COMMIT
            or runtime["versions"]["warp-lang"] != expected_warp
            or record["teacher_origin"].get("source_commit") != OFFICIAL_COMMIT
            or record["teacher_origin"].get("warp_version") != expected_warp
        ):
            return (
                "failed",
                "Historical rollout requires the pinned original Warp runtime",
            )
        physical = record.get("physical_verification")
        if (
            not isinstance(physical, dict)
            or physical.get("schema") != "official-warp-physical-arrays-v2"
            or physical.get("complete") is not True
            or physical.get("matched_recorded_fields") is not True
            or physical.get("missing_recorded_fields") != []
            or "gravity" not in physical.get("matched_fields", [])
            or not _sha_identity(physical.get("sha256"))
        ):
            return (
                "pending",
                "Historical training physics must include verified gravity and recorded parameters",
            )
    elif runtime.get("engine") != "current_newton" or not runtime["versions"].get(
        "newton"
    ):
        return "failed", "Current-domain rollout requires recorded Newton execution"
    if not all(
        _finite_nonnegative(record[name], 1.0)
        for name in ("failure_rate", "nonfinite_rate")
    ):
        return "failed", "Invalid failure/nonfinite rates"
    metrics = ["q_mae", "qd_mae"]
    if robot == "ant":
        metrics.extend(("base_position_error", "base_orientation_error"))
    if not isinstance(record["errors"], dict):
        return "failed", "Invalid horizon error mapping"
    for horizon in required_horizons:
        values = record["errors"].get(str(horizon), {})
        if not isinstance(values, dict) or any(name not in values for name in metrics):
            return "pending", "Missing predeclared horizon errors"
        if not all(_finite_nonnegative(values[name]) for name in metrics):
            return "failed", "Undefined or nonfinite horizon errors"
    if record["motion_mode"] == "fixed_policy":
        if not _sha_identity(record.get("policy_sha256")):
            return "pending", "Fixed-policy checkpoint/config identity missing"
        policy_return = record.get("policy_return")
        if (
            not isinstance(policy_return, (int, float))
            or isinstance(policy_return, bool)
            or not math.isfinite(policy_return)
        ):
            return "failed", "Undefined or nonfinite policy return"
    model = Path(cell["output_dir"]) / "best-export"
    artifacts = (
        (model / "weights.pt", record["model_sha256"]),
        (model / "manifest.json", record["model_manifest_sha256"]),
        (Path(record["source_report"]), record["source_sha256"]),
        (Path(record["trace_evidence"]), record["trace_evidence_sha256"]),
        (Path(record["trajectory_evidence"]), record["trajectory_evidence_sha256"]),
    )
    if any(not file.is_file() or _hash(file) != digest for file, digest in artifacts):
        return "failed", "Rollout source, model or trajectory evidence changed"
    raw = json.loads(Path(record["source_report"]).read_text())
    if raw.get("horizons") != record["errors"] or any(
        raw.get(name) != record.get(name)
        for name in (
            "robot",
            "motion_mode",
            "state_dtype",
            "trajectories",
            "selected_teacher_sha256",
            "teacher_origin",
            "teacher_split",
            "model_sha256",
            "model_manifest_sha256",
            "failure_rate",
            "nonfinite_rate",
            "policy_return",
            "policy_sha256",
            "trace_evidence",
            "trace_evidence_sha256",
            "trajectory_evidence",
            "trajectory_evidence_sha256",
            "selected_teacher_hash_schema",
            "runtime_identity",
            "physical_verification",
        )
    ):
        return "failed", "Annotated rollout differs from measured source"
    return "available", "Measured rollout and source identities verified"


def _matched_rollouts(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Paired errors require the same actual teacher traces and fixed policy."""
    return all(
        left.get(name) == right.get(name)
        for name in (
            "selected_teacher_sha256",
            "teacher_origin",
            "trajectories",
            "horizons",
            "policy_sha256",
            "state_dtype",
        )
    )


def _supported_split_sources(metadata: dict[str, Any]) -> bool:
    """Verify raw split receipts generated by the isolated supported environment."""
    splits = {"train", "validation", "test"}
    sources = metadata.get("legacy_sources", {})
    identities = metadata.get("source_split_identities", {})
    if (
        not isinstance(sources, dict)
        or not isinstance(identities, dict)
        or set(sources) != splits
        or set(identities) != splits
    ):
        return False
    robot = metadata.get("codec", {}).get("robot")
    if robot not in {"cartpole", "ant"}:
        return False
    warp_version = "1.5.1" if robot == "ant" else "1.8.0"
    for split in sorted(splits):
        source, identity = sources[split], identities[split]
        if not isinstance(source, dict) or not isinstance(identity, dict):
            return False
        torch_version = identity.get("torch")
        if (
            identity.get("source_commit") != OFFICIAL_COMMIT
            or identity.get("robot") != robot
            or identity.get("python") != "3.8.20"
            or identity.get("numpy") != "1.24.4"
            or not isinstance(torch_version, str)
            or torch_version.split("+", 1)[0] != "2.2.2"
            or identity.get("warp") != warp_version
            or identity.get("runtime_imports_embodichain") is not False
            or not _sha_identity(source.get("sha256"))
            or identity.get("raw_sha256") != source["sha256"]
            or not isinstance(source.get("path"), str)
        ):
            return False
        raw = Path(source["path"])
        if not raw.is_file() or _hash(raw) != source["sha256"]:
            return False
    return True


def _dataset_evidence(teacher: str, declaration: dict[str, Any]) -> dict[str, Any]:
    """Check provenance in immutable data, rather than trusting a matrix row label."""
    if declaration.get("status") != "available":
        return {"status": "pending", "reason": "Teacher dataset unavailable"}
    from embodichain_models.dynamics.nerd.data import (
        dataset_fingerprint,
        load_manifest,
    )

    manifest = load_manifest(declaration["path"], verify=True)
    if dataset_fingerprint(manifest) != declaration.get("fingerprint"):
        return {
            "status": "failed",
            "reason": "Teacher dataset changed after preparation",
        }
    metadata = manifest["metadata"]
    if teacher == "official_warp":
        origin = metadata.get("teacher_origin", {})
        verified = (
            origin.get("kind") == teacher
            and origin.get("source_commit") == OFFICIAL_COMMIT
            and bool(metadata.get("source_physical_arrays"))
            and (
                (
                    metadata.get("source", {}).get("commit") == OFFICIAL_COMMIT
                    and bool(metadata.get("source_raw_hdf5"))
                )
                or _supported_split_sources(metadata)
            )
        )
    else:
        native = metadata.get("teacher", {})
        verified = (
            _sha_identity(metadata.get("physics_signature"))
            and bool(native.get("versions", {}).get("newton"))
            and bool(native.get("versions", {}).get("warp-lang"))
            and bool(native.get("solver", {}).get("solver_type"))
            and isinstance(native.get("num_substeps"), int)
            and native["num_substeps"] > 0
        )
        origin = {"kind": teacher, **native} if verified else {}
    counts = {split: 0 for split in ("train", "validation", "test")}
    transitions = counts.copy()
    for shard in manifest["shards"]:
        for row in shard["trajectories"]:
            counts[row["split"]] += 1
            transitions[row["split"]] += row["length"]
    return {
        "status": "available" if verified else "pending",
        "reason": (
            "Recorded teacher provenance verified"
            if verified
            else "Matrix label lacks recorded teacher provenance"
        ),
        "origin": origin,
        "fingerprint": declaration["fingerprint"],
        "trajectories": counts,
        "transitions": transitions,
    }


def _full_profile_issues(plan: dict[str, Any]) -> list[str]:
    """Check effective initial artifacts, preserving feature order lost in sorted JSON."""
    import torch

    robot = "ant" if plan["model"]["output_dim"] == 29 else "cartpole"
    inputs = (
        [
            ("states_embedding", 29),
            ("contact_normals", 75),
            ("contact_points_1", 75),
            ("contact_depths", 25),
            ("joint_f", 8),
            ("gravity_dir", 3),
        ]
        if robot == "ant"
        else [("states_embedding", 4), ("joint_f", 2)]
    )
    expected_model = {
        "output_dim": 29 if robot == "ant" else 4,
        "n_embd": 384 if robot == "ant" else 192,
        "n_layer": 6,
        "n_head": 12,
        "history_length": 10,
        "block_size": 32,
        "head_hidden_dims": [64],
        "bias": False,
        "dropout": 0.0,
        "normalize_input": True,
        "normalize_output": True,
        "output_tanh": False,
    }
    expected_train = {
        "total_steps": 4995000,
        "batch_size": 512,
        "learning_rate": 1e-3,
        "final_learning_rate": 1e-4,
        "max_grad_norm": 1.0,
        "lr_update_interval": 5000,
        "lr_offset": 1,
        "validation_interval": 5000,
        "validation_batches": 100,
    }
    issues = [
        f"train.{name} differs from the matched official profile"
        for name, value in expected_train.items()
        if plan["train"].get(name) != value
    ]
    seen = set()
    for cell in plan["cells"]:
        if cell.get("status") != "prepared":
            continue
        spec = json.loads(Path(cell["spec"]).read_text())
        path = Path(spec["initial_model"]) / "manifest.json"
        if path in seen:
            continue
        seen.add(path)
        model = json.loads(path.read_text())["model_config"]
        weights = torch.load(
            path.with_name("weights.pt"), map_location="cpu", weights_only=True
        )
        floating = [value for value in weights.values() if value.is_floating_point()]
        if not floating or any(value.dtype != torch.float32 for value in floating):
            issues.append(
                f"{cell['id']}: actual initial-model weights/normalizers are not FP32"
            )
        if list(model.get("input_dims", {}).items()) != inputs:
            issues.append(
                f"{cell['id']}: actual initial-model feature order/dimensions differ"
            )
        issues.extend(
            f"{cell['id']}: actual initial-model {name} differs"
            for name, value in expected_model.items()
            if model.get(name) != value
        )
    if not seen:
        issues.append(
            "No hash-verified initial model is available to verify the full profile"
        )
    return issues


def _curve(
    path: Path, budget: int, max_points: int = 1000
) -> tuple[list[int], list[float]]:
    """Keep the first/final observation and bounded update-spaced samples.

    Sparse logging need not divide the plotting stride. Selecting observations
    by modulo can otherwise discard nearly the entire measured learning curve.
    """
    if max_points < 2:
        raise ValueError("Learning curves require at least two plot points")
    steps = []
    losses = []
    stride = max(1, (budget + max_points - 2) // (max_points - 1))
    next_step = 0
    last = None
    if path.exists():
        with path.open() as stream:
            for line in stream:
                record = json.loads(line)
                last = record
                if not steps or record["step"] >= next_step:
                    steps.append(record["step"])
                    losses.append(record["loss"])
                    next_step = record["step"] + stride
    if last is not None and steps[-1] != last["step"]:
        steps.append(last["step"])
        losses.append(last["loss"])
    return steps, losses


def annotate_rollout_evidence(
    source: str | Path, run_id: str, output: str | Path
) -> dict[str, Any]:
    """Attach a measured simulation report to a matrix cell without changing errors.

    The simulation report must already identify the exact model weights and
    motion scenario. This adapter never infers a scenario or model hash from a
    filename and never turns missing stability/return measurements into zeros.
    """
    source = Path(source)
    raw = json.loads(source.read_text())
    for key in ("model_sha256", "motion_mode", "trajectories", "horizons"):
        if key not in raw:
            raise ValueError(f"Rollout source lacks measured identity/metadata: {key}")
    errors = raw["horizons"]
    if not isinstance(errors, dict) or not errors:
        raise ValueError(
            "Expected per-horizon metric mapping from simulation evaluator"
        )
    last = errors[str(max(map(int, errors)))]
    result = {
        "run_id": run_id,
        "model_sha256": raw["model_sha256"],
        "motion_mode": raw["motion_mode"],
        "trajectories": raw["trajectories"],
        "horizons": sorted(map(int, errors)),
        "errors": errors,
        "source_report": str(source.resolve()),
        "source_sha256": _hash(source),
        "teacher_dataset_fingerprint": raw.get("teacher_dataset_fingerprint"),
        "teacher_origin": raw.get("teacher_origin"),
    }
    for name in (
        "robot",
        "seed",
        "state_dtype",
        "teacher_split",
        "selected_teacher_sha256",
        "model_manifest_sha256",
        "policy_sha256",
        "trace_evidence",
        "trace_evidence_sha256",
        "trajectory_evidence",
        "trajectory_evidence_sha256",
        "selected_teacher_hash_schema",
        "teacher_source_motion_mode",
        "runtime_identity",
        "physical_verification",
    ):
        if name in raw:
            result[name] = raw[name]
    for name in ("failure_rate", "nonfinite_rate", "policy_return"):
        if name in raw:
            result[name] = raw[name]
        elif name in last:
            result[name] = last[name]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _json(output, result, exclusive=True)
    return result


def write_comparison_report(
    experiment: str | Path,
    output: str | Path | None = None,
    *,
    numerical_parity_report: str | Path | None = None,
    rollout_reports: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Report verified run receipts; missing evidence always remains pending.

    Full rollout attachments use one JSON per cell/scenario with run_id,
    model_sha256, trajectories, horizons, motion_mode, failure_rate,
    nonfinite_rate and (for fixed_policy) policy_return. Attachments are measured
    evidence, not substitutes for the required training matrix or numerical gate.
    Their source/model/trace hashes, teacher domain, selected trajectory identity
    and common fixed-policy identity must also verify before paired statistics.
    """
    root, plan = _read_plan(experiment)
    destination = Path(output).resolve() if output else root / "report"
    destination.mkdir(parents=True, exist_ok=True)
    cells = []
    completed = {}
    pending = []
    for cell in plan["cells"]:
        receipt_path = root / (cell["id"] + ".receipt.json")
        if not receipt_path.exists():
            cells.append(
                {
                    "id": cell["id"],
                    "status": "pending",
                    "reason": cell.get("reason", "not executed"),
                }
            )
            continue
        receipt = json.loads(receipt_path.read_text())
        result_path = Path(cell["output_dir"]) / "result.json"
        if (
            receipt["status"] != "completed"
            or not result_path.exists()
            or _hash(result_path) != receipt["result_sha256"]
        ):
            cells.append(
                {
                    "id": cell["id"],
                    "status": "failed",
                    "reason": "run failed or result receipt mismatch",
                }
            )
            continue
        result = json.loads(result_path.read_text())
        spec = json.loads(Path(cell["spec"]).read_text())
        if (
            result.get("completed_updates") != spec["train"]["total_steps"]
            or result.get("dataset_fingerprint") != spec["dataset_fingerprint"]
            or result.get("sample_order_sha256") != spec["sample_order_sha256"]
        ):
            cells.append(
                {
                    "id": cell["id"],
                    "status": "failed",
                    "reason": "budget/data/order evidence mismatch",
                }
            )
            continue
        weights = Path(cell["output_dir"]) / "best-export" / "weights.pt"
        if not weights.exists() or _hash(weights) != result.get("best_model_sha256"):
            cells.append(
                {
                    "id": cell["id"],
                    "status": "failed",
                    "reason": "exported model identity missing or changed",
                }
            )
            continue
        completed[cell["id"]] = result
        cells.append(
            {
                "id": cell["id"],
                "status": "completed",
                "completed_updates": result["completed_updates"],
                "test": result["test"],
                "result": str(result_path),
            }
        )
    acceptance = plan["acceptance"]
    dataset_evidence = {
        teacher: _dataset_evidence(teacher, declaration)
        for teacher, declaration in plan["datasets"].items()
    }
    comparisons = []
    metrics = list(acceptance["primary_metrics"])
    if any(
        "base_orientation_error_rad" in value["test"] for value in completed.values()
    ):
        metrics.extend(["base_position_error", "base_orientation_error_rad"])
    for teacher in ("official_warp", "current_newton"):
        for metric in metrics:
            reference = []
            candidate = []
            paired_seeds = []
            for seed in plan["seeds"]:
                left = completed.get(f"{teacher}-official-{seed}")
                right = completed.get(f"{teacher}-embodichain-{seed}")
                if (
                    left
                    and right
                    and metric in left["test"]
                    and metric in right["test"]
                ):
                    reference.append(float(left["test"][metric]))
                    candidate.append(float(right["test"][metric]))
                    paired_seeds.append(seed)
            gate = (
                paired_noninferiority(
                    reference,
                    candidate,
                    margin=acceptance["relative_margin"],
                    absolute_tolerance=acceptance["absolute_tolerance"],
                    resamples=acceptance["resamples"],
                    seed=acceptance["bootstrap_seed"],
                )
                if reference
                else {"status": "pending", "reason": "no completed seed pairs"}
            )
            comparisons.append(
                {
                    "teacher": teacher,
                    "metric": metric,
                    "seeds": paired_seeds,
                    "official": reference,
                    "embodichain": candidate,
                    "gate": gate,
                }
            )
    if len(plan["seeds"]) != acceptance["required_seeds"]:
        pending.append("Five predeclared paired training seeds are required.")
    if (
        plan["scope"] != "full"
        or plan["train"]["total_steps"] != acceptance["official_updates"]
    ):
        pending.append(
            "This is a shortened/smoke training budget, not full official-budget qualification."
        )
    profile_issues = _full_profile_issues(plan) if plan["scope"] == "full" else []
    if profile_issues:
        pending.append(
            "Effective architecture/optimizer/validation settings differ from the predeclared full matched profile."
        )
    if any(
        value["status"] != "available"
        or (value.get("origin") or {}).get("kind") != teacher
        for teacher, value in plan["datasets"].items()
    ):
        pending.append(
            "Both teacher datasets require recorded origin evidence; a row label is not teacher provenance."
        )
    if any(value["status"] != "available" for value in dataset_evidence.values()):
        pending.append(
            "Teacher data provenance must be verified from committed dataset metadata."
        )
    if any(cell["status"] == "pending" for cell in cells):
        pending.append("Some cells have not completed verified training.")
    numerical = {
        "status": "pending",
        "reason": "No independent numerical/Adam parity report attached",
    }
    robot = "ant" if plan["model"]["output_dim"] == 29 else "cartpole"
    if numerical_parity_report:
        numerical = _numerical_evidence(numerical_parity_report, robot)
    if numerical["status"] != "passed":
        pending.append(
            "Independent preprocessing, forward, gradient and Adam numerical parity must pass."
        )
    rollout = {}
    valid_keys = {
        (cell["id"], scenario)
        for cell in plan["cells"]
        for scenario in ("passive", "random_effort", "fixed_policy")
    }
    for path in rollout_reports or []:
        path = Path(path)
        record = json.loads(path.read_text())
        key = (record.get("run_id"), record.get("motion_mode"))
        if key not in valid_keys:
            raise ValueError(f"Unknown rollout cell/scenario: {key}")
        if key in rollout:
            raise ValueError(f"Duplicate rollout attachment: {key}")
        rollout[key] = {**record, "source": str(path.resolve()), "sha256": _hash(path)}
    qualified_rollouts = []
    verified_rollouts = {}
    for cell in plan["cells"]:
        for scenario in ("passive", "random_effort", "fixed_policy"):
            record = rollout.get((cell["id"], scenario))
            status, reason = _rollout_evidence(record, cell, robot)
            if status == "available":
                verified_rollouts[(cell["id"], scenario)] = record
            qualified_rollouts.append(
                {
                    "id": cell["id"],
                    "scenario": scenario,
                    "status": status,
                    "reason": reason,
                    "evidence": record,
                }
            )
    if any(record["status"] != "available" for record in qualified_rollouts):
        pending.append(
            "Matched passive/random-effort/fixed-policy rollouts, stability and return evidence are incomplete."
        )
    rollout_gates = []
    pair_mismatches = []
    rollout_metrics = list(
        acceptance.get("rollout_primary_metrics", ["q_mae", "qd_mae"])
    )
    if plan["model"]["output_dim"] == 29:
        rollout_metrics.extend(["base_position_error", "base_orientation_error"])
    for teacher in ("official_warp", "current_newton"):
        for scenario in ("passive", "random_effort", "fixed_policy"):
            pairs = []
            for seed in plan["seeds"]:
                left = verified_rollouts.get((f"{teacher}-official-{seed}", scenario))
                right = verified_rollouts.get(
                    (f"{teacher}-embodichain-{seed}", scenario)
                )
                if left and right:
                    if _matched_rollouts(left, right):
                        pairs.append((left, right))
                    else:
                        pair_mismatches.append(
                            {
                                "teacher": teacher,
                                "seed": seed,
                                "scenario": scenario,
                                "reason": "Paired rollout teacher trajectories, population or policy differ",
                            }
                        )
            for horizon in (1, 10, 100, 1000):
                for metric in rollout_metrics:
                    values = [
                        (
                            left["errors"][str(horizon)][metric],
                            right["errors"][str(horizon)][metric],
                        )
                        for left, right in pairs
                        if metric in left.get("errors", {}).get(str(horizon), {})
                        and metric in right.get("errors", {}).get(str(horizon), {})
                    ]
                    gate = (
                        paired_noninferiority(
                            [x[0] for x in values],
                            [x[1] for x in values],
                            margin=acceptance["relative_margin"],
                            absolute_tolerance=acceptance["absolute_tolerance"],
                            resamples=acceptance["resamples"],
                            seed=acceptance["bootstrap_seed"],
                        )
                        if values
                        else {"status": "pending"}
                    )
                    rollout_gates.append(
                        {
                            "teacher": teacher,
                            "scenario": scenario,
                            "horizon": horizon,
                            "metric": metric,
                            "gate": gate,
                        }
                    )
            for metric, threshold_name in (
                ("failure_rate", "failure_rate_increase"),
                ("nonfinite_rate", "nonfinite_rate_increase"),
            ):
                values = [
                    (left[metric], right[metric])
                    for left, right in pairs
                    if metric in left and metric in right
                ]
                gate = (
                    {"status": "pending"}
                    if len(values) < 5
                    else {
                        "status": (
                            "passed"
                            if all(
                                candidate - reference <= acceptance[threshold_name]
                                for reference, candidate in values
                            )
                            else "failed"
                        ),
                        "threshold": acceptance[threshold_name],
                    }
                )
                rollout_gates.append(
                    {
                        "teacher": teacher,
                        "scenario": scenario,
                        "metric": metric,
                        "gate": gate,
                    }
                )
            if scenario == "fixed_policy":
                import numpy as np

                values = np.asarray(
                    [
                        (left["policy_return"], right["policy_return"])
                        for left, right in pairs
                        if "policy_return" in left and "policy_return" in right
                    ],
                    dtype=float,
                )
                gate = {"status": "pending"}
                if len(values) >= 5:
                    if not np.isfinite(values).all():
                        gate = {"status": "failed", "reason": "nonfinite_returns"}
                    else:
                        absolute = bool(
                            (
                                np.abs(values[:, 0]) <= acceptance["absolute_tolerance"]
                            ).any()
                        )
                        degradation = values[:, 0] - values[:, 1]
                        if not absolute:
                            degradation = degradation / np.abs(values[:, 0])
                        threshold = (
                            acceptance["absolute_tolerance"]
                            if absolute
                            else acceptance["policy_return_relative_margin"]
                        )
                        rng = np.random.default_rng(acceptance["bootstrap_seed"])
                        upper = float(
                            np.quantile(
                                degradation[
                                    rng.integers(
                                        0,
                                        len(values),
                                        size=(acceptance["resamples"], len(values)),
                                    )
                                ].mean(1),
                                0.95,
                            )
                        )
                        gate = {
                            "status": ("passed" if upper <= threshold else "failed"),
                            "upper_95": upper,
                            "threshold": threshold,
                            "difference_mode": "absolute" if absolute else "relative",
                        }
                rollout_gates.append(
                    {
                        "teacher": teacher,
                        "scenario": scenario,
                        "metric": "policy_return",
                        "gate": gate,
                    }
                )
    if any(item["gate"]["status"] == "pending" for item in rollout_gates):
        pending.append(
            "Paired seed-level rollout error/stability/policy-return gates lack measurements."
        )
    failed = (
        any(cell["status"] == "failed" for cell in cells)
        or any(row["gate"]["status"] == "failed" for row in comparisons)
        or numerical["status"] == "failed"
        or any(row["status"] == "failed" for row in qualified_rollouts)
        or bool(pair_mismatches)
        or any(value["status"] == "failed" for value in dataset_evidence.values())
        or any(item["gate"]["status"] == "failed" for item in rollout_gates)
    )
    result = {
        "status": "failed" if failed else "pending" if pending else "passed",
        "scope": plan["scope"],
        "qualification_kind": "controlled_matched_data_noninferiority",
        "paper_reproduction": {
            "status": "not_established",
            "reason": "Shared initialization, statistics, batch order and validation selection control this comparison; literal upstream data/validation recipe reproduction is separate.",
        },
        "full_profile_issues": profile_issues,
        "comparison_protocol": plan.get(
            "comparison_protocol",
            {
                "kind": "controlled_matched_data_noninferiority",
                "note": "Older manifest: shared statistics/order/checkpoint controls are described in dataset and run artifacts; this is not literal paper-recipe reproduction.",
            },
        ),
        "source_commit": plan["source"]["commit"],
        "cells": cells,
        "dataset_evidence": dataset_evidence,
        "comparisons": comparisons,
        "numerical_parity": numerical,
        "rollouts": qualified_rollouts,
        "rollout_gates": rollout_gates,
        "rollout_pair_mismatches": pair_mismatches,
        "pending": pending,
        "experiment_sha256": _hash(Path(experiment)),
        "training_matrix_status": (
            "completed" if len(completed) == len(plan["cells"]) else "pending"
        ),
    }
    _json(destination / "comparison.json", result)
    _plots(plan, completed, comparisons, destination)
    lines = [
        "# NeRD official implementation and training comparison",
        "",
        f"Overall qualification: **{result['status']}**. Training matrix: **{result['training_matrix_status']}**.",
        "",
        f"Source: `{plan['source']['commit']}`. Scope: `{plan['scope']}`; configured updates per run: {plan['train']['total_steps']}.",
        "",
        "This qualification measures controlled matched-data noninferiority. Shared statistics, sampling order and checkpoint selection are experiment controls; literal paper data/validation-recipe reproduction is not established.",
        "",
        "| Run | Status | Completed updates |",
        "|---|---|---|",
    ]
    lines += [
        f"| {cell['id']} | {cell['status']} | {cell.get('completed_updates','—')} |"
        for cell in cells
    ]
    lines += ["", "## Dataset origins", "", "| Row | Recorded origin |", "|---|---|"]
    lines += [
        f"| {teacher} | {json.dumps(value.get('origin', {'status': 'unavailable'}), sort_keys=True)} |"
        for teacher, value in plan["datasets"].items()
    ]
    lines += [
        "",
        "## Paired seed-level error gates",
        "",
        "| Teacher | Metric | Paired seeds | Upper 95% difference | Status |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {row['teacher']} | {row['metric']} | {len(row['seeds'])} | {row['gate'].get('upper_95','—')} | {row['gate']['status']} |"
        for row in comparisons
    ]
    lines += [
        "",
        "## Outstanding evidence",
        "",
        *[f"- {reason}" for reason in pending],
        "",
        "![Training curves](training-curves.png)",
        "",
        "![Paired offline errors](paired-errors.png)",
        "",
        "Commands and stdout/stderr are retained in each run directory. Source files remain unmodified; the transport adapter binds common initialization, stats and data to original trainer methods. Offline errors alone do not establish long-rollout or paper equivalence.",
    ]
    (destination / "REPORT.md").write_text("\n".join(lines) + "\n")
    return result


def _plots(
    plan: dict[str, Any],
    completed: dict[str, Any],
    comparisons: list[dict[str, Any]],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    synthetic = any(
        (value.get("origin") or {}).get("kind") == "synthetic_fixture_only"
        for value in plan["datasets"].values()
    )
    label = "synthetic fixtures; infrastructure smoke" if synthetic else plan["scope"]
    figure, axis = plt.subplots(figsize=(10, 5))
    for cell in plan["cells"]:
        if cell["id"] in completed:
            steps, loss = _curve(
                Path(cell["output_dir"]) / "metrics.jsonl", plan["train"]["total_steps"]
            )
            axis.plot(steps, loss, label=cell["id"], alpha=0.65)
    axis.set(
        xlabel="Committed optimizer update",
        ylabel="Normalized training MSE",
        title=f"NeRD matched-data training ({label})",
    )
    if completed:
        axis.legend(fontsize=6, ncol=2)
    else:
        axis.text(
            0.5,
            0.5,
            "No completed training runs",
            ha="center",
            transform=axis.transAxes,
        )
    figure.tight_layout()
    figure.savefig(output / "training-curves.png", dpi=140)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(10, 5))
    available = [row for row in comparisons if row["official"]]
    for index, row in enumerate(available):
        axis.scatter(
            [index - 0.12] * len(row["official"]),
            row["official"],
            color="tab:blue",
            label="Official" if index == 0 else None,
        )
        axis.scatter(
            [index + 0.12] * len(row["embodichain"]),
            row["embodichain"],
            color="tab:orange",
            label="EmbodiChain" if index == 0 else None,
        )
    axis.set_xticks(
        range(len(available)),
        [row["teacher"] + "\n" + row["metric"] for row in available],
        rotation=15,
        fontsize=7,
    )
    axis.set(
        ylabel="Per-seed held-out error",
        title=f"Paired offline metrics ({label})",
    )
    if available:
        axis.legend()
    else:
        axis.text(
            0.5,
            0.5,
            "No completed matched seed pairs",
            ha="center",
            transform=axis.transAxes,
        )
    figure.tight_layout()
    figure.savefig(output / "paired-errors.png", dpi=140)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--numerical-parity-report", type=Path)
    parser.add_argument("--rollout-report", type=Path, action="append", default=[])
    args = parser.parse_args()
    result = write_comparison_report(
        args.experiment,
        args.output,
        numerical_parity_report=args.numerical_parity_report,
        rollout_reports=args.rollout_report,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
