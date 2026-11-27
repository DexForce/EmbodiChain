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

"""Compile checked scene-bound Pickup fits into a sealed Task Program deployment."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
import yaml

from embodichain.data_pipeline.skill_fitting import file_digest

if TYPE_CHECKING:
    from embodichain.data_pipeline.real2sim import DemoSkillArtifact
    from embodichain.lab.task_program.integrations.invocation_policy import (
        InvocationPolicyBinding,
    )

__all__ = ["export_task_program", "load_invocation_bindings"]

_BINDING_SCHEMA = "fitted-task-program/v1"


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _read_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path.name}.")
    return value


def _compile(root: Path):
    from embodichain.lab.task_program.integrations.composition import (
        load_configured_task_program_deployment,
    )
    from embodichain.lab.task_program.semantics.integration import (
        SemanticIntegrationManifest,
    )

    deployment = _read_yaml(root / "deployment.yaml")
    embodiment = _read_yaml(root / deployment["embodiment"]["component"])
    cfg = load_configured_task_program_deployment(
        task_program=deployment["task_program"],
        skill_profile=embodiment["skill_profile"],
        base_dir=root,
    )
    catalog = cfg.integration.registration.catalog
    program = catalog.preflight(cfg.load_program())
    semantic = SemanticIntegrationManifest(
        scene=catalog.scene,
        robot_profile=catalog.robot_profile,
        call_catalog=catalog.call_catalog,
        runtime_preset=cfg.selection.runtime_preset,
    )
    return program, semantic, cfg.integration.integration_fingerprint


def _execution_timeline(timeline: list[dict], size: int) -> list[dict]:
    """Keep the sample ownership of annotation shoulders outside the fit core."""
    result = []
    cursor = 0
    for span in timeline:
        start, stop = span["start"], span["stop"]
        if start != cursor or not cursor < stop <= size:
            raise ValueError("Fit does not partition all original control samples.")
        cursor = stop
        if span["kind"] == "demo_skill":
            result.append(dict(span))
            continue
        if span["kind"] != "atomic_skill":
            raise ValueError(f"Unknown fitted timeline span kind: {span['kind']}.")
        left, right = span["fit_start"], span["fit_stop"]
        if not start <= left < right <= stop:
            raise ValueError("Fit core lies outside its source annotation.")
        for begin, end, fitted in (
            (start, left, False),
            (left, right, True),
            (right, stop, False),
        ):
            if begin == end:
                continue
            if fitted:
                result.append({**span, "start": begin, "stop": end})
            else:
                result.append(
                    {
                        "kind": "demo_skill",
                        "label": "DemoSkill",
                        "skill_id": "execute_demo_skill",
                        "start": begin,
                        "stop": end,
                        "source": "annotation_transition",
                        "annotation": span["annotation"],
                    }
                )
    if cursor != size:
        raise ValueError("Fit dropped source samples.")
    return result


def _bound_demos(
    root: Path,
    segments: Mapping[str, tuple[int, dict]],
    integration: dict,
    spans: list[dict],
    pickup: dict,
    bindings: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    dt: float,
) -> list[dict]:
    """Verify every retained interval against a registered, measured DemoSkill."""
    from embodichain.data_pipeline.real2sim import load_demo_skill

    if "retained_demos" in bindings:
        declarations = bindings["retained_demos"]
    elif len(spans) == 1 and spans[0]["start"] == bindings["continuation_sample"]:
        declarations = [
            {
                **bindings["continuation"],
                "start": spans[0]["start"],
                "stop": spans[0]["stop"],
            }
        ]
    else:
        raise ValueError(
            "Every retained source span needs its own qualified DemoSkill binding."
        )
    expected = [(span["start"], span["stop"]) for span in spans]
    if [(item["start"], item["stop"]) for item in declarations] != expected:
        raise ValueError(
            "Retained DemoSkill bindings must cover every retained interval in source order."
        )
    registry = integration["runtime_services"]["demo_skills"]
    demos_by_call = {entry["call_id"]: entry for entry in registry.values()}
    pickup_order = segments[bindings["segment"]][0]
    previous = -1
    result = []
    for item in declarations:
        segment = item["program_segment"]
        if segment not in segments:
            raise ValueError(f"Unknown retained DemoSkill segment: {segment}.")
        order, call = segments[segment]
        start, stop = item["start"], item["stop"]
        if order <= previous or (stop <= pickup["start"]) != (order < pickup_order):
            raise ValueError(
                "Retained DemoSkill calls do not follow the source execution order."
            )
        previous = order
        declaration = demos_by_call.get(call.get("call_id"))
        if call.get("kind") != "registered" or declaration is None:
            raise ValueError(
                f"Retained segment {segment!r} must bind a registered DemoSkill."
            )
        if "source_slice" in item:
            path, asset = _build_source_slice(
                root, start, stop, item["source_slice"], dt
            )
            declaration["asset_manifest"] = str(path.relative_to(root))
            declaration["asset_sha256"] = asset.content_sha256
            item = {**item, "asset_sha256": asset.content_sha256}
        else:
            path = (root / declaration["asset_manifest"]).resolve(strict=True)
            if not path.is_relative_to(root):
                raise ValueError(
                    "Retained DemoSkill asset leaves the execution bundle."
                )
            asset = load_demo_skill(path)
        if (
            asset.content_sha256 != item["asset_sha256"]
            or declaration["asset_sha256"] != item["asset_sha256"]
            or len(asset.execution_qpos) != stop - start
            or not np.allclose(
                np.diff(asset.execution_timestamps), dt, atol=1e-8, rtol=0
            )
        ):
            raise ValueError(
                "Retained DemoSkill asset disagrees with its source interval or clock."
            )
        if "retained_demos" in bindings and not np.array_equal(
            asset.execution_qpos, np.asarray(arrays["original_qpos"])[start:stop]
        ):
            raise ValueError(
                "Retained DemoSkill must preserve the qualified commands for its interval."
            )
        result.append(
            {
                **item,
                "asset_manifest": str(path.relative_to(root)),
                "source_sample_offset": asset.demonstration.source_sample_offset,
            }
        )
    return result


def _build_source_slice(
    root: Path, start: int, stop: int, binding: dict, dt: float
) -> tuple[Path, DemoSkillArtifact]:
    """Build a DemoSkill on an authoritative measured post-step control clock."""
    from embodichain.data_pipeline.real2sim import (
        DemoSkillConstraint,
        build_demo_skill,
        load_demo_skill,
        save_demo_skill,
    )
    from embodichain.data_pipeline.real2sim.demo_skill import decode_demo_skill_spec

    source, contract = (
        (root / binding[key]).resolve(strict=True)
        for key in ("reference_asset", "contract")
    )
    if not source.is_relative_to(root) or not contract.is_relative_to(root):
        raise ValueError(
            "Retained source and its semantic contract must be bundled resources."
        )
    reference = load_demo_skill(source)
    recording = reference.demonstration
    if (
        reference.content_sha256 != binding["reference_asset_sha256"]
        or reference.reference is not None
        or len(reference.execution_qpos) != len(recording.qpos)
        or recording.command_alignment != "post_step"
        or recording.time_unit != "s"
        or not np.allclose(np.diff(recording.timestamps), dt, atol=1e-9, rtol=0)
        or (start == 0 and recording.initial_observed_qpos is None)
        or not 0 <= start < stop <= len(recording.timestamps)
    ):
        raise ValueError(
            "Source slices require pinned commands and initial state on the control grid."
        )
    payload = _read_yaml(contract)
    original = reference.execution_spec
    phases = tuple(
        replace(
            phase,
            start=max(phase.start, start) - start,
            stop=min(phase.stop, stop) - start,
        )
        for phase in original.phases
        if phase.start < stop and phase.stop > start
    )
    events = tuple(
        replace(event, sample_index=event.sample_index - start)
        for event in original.events
        if start <= event.sample_index < stop
    )
    event_ids = {event.event_id for event in events}
    phase_names = {phase.name for phase in phases}
    requested = tuple(DemoSkillConstraint(**item) for item in payload["constraints"])
    checks = {check.check_id: check for check in requested}
    if len(checks) != len(requested):
        raise ValueError("Retained source contract has duplicate evidence checks.")
    for check in original.constraints:
        if (
            check.scope == "entry"
            and start != 0
            or check.scope == "terminal"
            and stop != len(recording.qpos)
        ):
            continue
        if check.scope == "event" and check.event_id not in event_ids:
            continue
        if check.phase_names:
            overlap = tuple(name for name in check.phase_names if name in phase_names)
            if not overlap:
                continue
            check = replace(check, phase_names=overlap)
        if check.check_id in checks and checks[check.check_id] != check:
            raise ValueError(
                f"Retained source contract changes evidence check {check.check_id}."
            )
        checks[check.check_id] = check
    payload["phases"] = [asdict(phase) for phase in phases]
    payload["events"] = [asdict(event) for event in events]
    payload["constraints"] = [asdict(check) for check in checks.values()]
    spec = decode_demo_skill_spec(payload)
    if {(r.role_id, r.geometry_sha256, r.reference_kind) for r in spec.roles} != {
        (r.role_id, r.geometry_sha256, r.reference_kind) for r in original.roles
    }:
        raise ValueError(
            "A retained source slice must keep its declared physical role geometry."
        )
    clip = replace(
        recording,
        qpos=recording.qpos[start:stop],
        commands=recording.commands[start:stop],
        timestamps=recording.timestamps[start:stop],
        qvel=None if recording.qvel is None else recording.qvel[start:stop],
        eef_poses=(
            None if recording.eef_poses is None else recording.eef_poses[start:stop]
        ),
        initial_observed_qpos=(
            recording.qpos[start - 1] if start else recording.initial_observed_qpos
        ),
        source_sample_offset=recording.source_sample_offset + start,
    )
    asset = build_demo_skill(clip, spec, joints=reference.joint_specs, control_dt=dt)
    path = save_demo_skill(asset, root / "demos" / asset.content_sha256)
    return path, asset


def export_task_program(
    arrays: Mapping[str, np.ndarray],
    report: Mapping[str, Any],
    *,
    execution_context: Path,
    segment_bindings: Mapping[str, Any],
    output_dir: Path,
) -> Path:
    """Bind one checked Pickup to an explicitly authored hybrid task context.

    The caller owns the source-reference hash, qualified continuation, robot
    endpoints, and explicit approval of deviations from fitted native options.
    The copied context retains deployment components and measured effect
    contracts. Only the named Pickup occurrence receives the fitted preset.

    Args:
        arrays: Fitting output arrays on the qualified control clock.
        report: Accepted fitting report with the source digest and timeline.
        execution_context: Directory containing the qualified deployment.
        segment_bindings: Explicit object, resource, and demonstration bindings.
        output_dir: New directory for the compiled and sealed deployment.

    Returns:
        Path to the compiled deployment directory.
    """
    from embodichain.lab.task_program.integrations.invocation_policy import (
        BoundInvocationPolicies,
        InvocationPolicyBinding,
    )
    from embodichain.lab.task_program.integrations._program_identity import (
        task_program_fingerprint,
    )
    from embodichain.lab.task_program.language.authoring import normalize_task_program

    base = execution_context.resolve(strict=True)
    output = output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    if output.is_relative_to(base) or base.is_relative_to(output):
        raise ValueError("Execution context and compiled output must be disjoint.")
    if (
        not report["accepted"]
        or report["source_joint_sha256"] != segment_bindings["reference_sha256"]
    ):
        raise ValueError("Fit does not belong to the qualified control reference.")
    ticks = np.asarray(arrays["timestamps"])
    if ticks.ndim != 1 or len(ticks) < 2 or not np.isfinite(ticks).all():
        raise ValueError("Fitted timeline has no usable control clock.")
    dt = float(segment_bindings["control_dt_s"])
    if dt <= 0 or not np.allclose(np.diff(ticks), dt, atol=1e-8, rtol=0):
        raise ValueError("Fitted timeline changed the execution cadence.")
    timeline = _execution_timeline(report["timeline"], len(ticks))
    fitted = [span for span in timeline if span["kind"] == "atomic_skill"]
    retained = [span for span in timeline if span["kind"] == "demo_skill"]
    if len(fitted) != 1 or not retained:
        raise ValueError(
            "This compiler requires one fitted Pickup and qualified retained motion."
        )
    (pickup,) = fitted
    if pickup["skill_id"] != "pick_up" or not pickup["accepted"]:
        raise ValueError("The fitted core must be an accepted Pickup.")
    fit = report["fits"][pickup["fit_index"]]
    if fit["selected_participants"] != {"primary": segment_bindings["participants"]}:
        raise ValueError(
            "The fitted motion and grasp endpoints differ from the execution binding."
        )
    goal = fit["native_goal"]
    if goal["object_id"] != segment_bindings["object"]:
        raise ValueError("The fitted grasp is bound to a different scene object.")
    relation = np.linalg.solve(goal["object_pose"], goal["grasp_xpos"])
    measured = fit.get("scene", {}).get("object_to_tcp")
    if measured is not None:
        measured = np.asarray(measured, dtype=float)
        if measured.shape != (4, 4) or not np.allclose(
            measured, relation, atol=1e-6, rtol=0
        ):
            raise ValueError(
                "The measured grasp disagrees with the fitted native goal."
            )
        # The solver serializes its world goal in float32. Preserve the
        # original relative transform rather than reconstructing it twice.
        relation = measured
    if (
        not np.isfinite(relation).all()
        or not np.allclose(relation[3], (0, 0, 0, 1), atol=1e-6)
        or not np.allclose(relation[:3, :3].T @ relation[:3, :3], np.eye(3), atol=1e-5)
        or np.linalg.det(relation[:3, :3]) < 0
    ):
        raise ValueError("Fitted grasp must be a proper object-to-TCP transform.")
    if fit.get("replay_interpolation_factor", 1) != 1:
        raise ValueError(
            "A duration scaling policy must be explicitly compiled before use."
        )
    motion = dict(fit["native_motion_policy"])
    if not np.isclose(motion.pop("control_dt"), dt, atol=1e-8, rtol=0):
        raise ValueError("Pickup fit does not use the authoritative control cadence.")
    options = {
        key: value
        for key, value in fit["native_options"].items()
        if value is not None and value != []
    }
    if options.pop("grasp_variant", "closest") != "closest":
        raise ValueError(
            "The standard Pick requires the default closest grasp selector."
        )
    if "grasp_frame_to_eef" in options:
        options["grasp_frame_to_eef"] = np.ravel(options["grasp_frame_to_eef"]).tolist()
    options.update(segment_bindings["option_overrides"])
    options["fixed_object_to_eef"] = relation.reshape(-1).tolist()
    motion.update(segment_bindings["motion_overrides"])

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".task-program-", dir=output.parent) as temporary:
        root = Path(temporary) / "bundle"
        shutil.copytree(base, root)
        deploy = _read_yaml(root / "deployment.yaml")
        integration_path = root / deploy["task_program"]["integration"]
        integration = _read_yaml(integration_path)
        program_path = root / deploy["task_program"]["program"]
        registry = integration.get("runtime_services", {}).get("demo_skills", {})
        declaration = normalize_task_program(
            _read_yaml(program_path),
            demo_skills={alias: entry["call_id"] for alias, entry in registry.items()},
        ).declaration
        body = declaration["program"]
        if body.get("kind") != "sequence":
            raise ValueError(
                "A fitted program must contain a sequence of named segments."
            )
        segments = {}
        for index, segment in enumerate(body["items"]):
            if segment.get("kind") != "segment":
                raise ValueError("Fitted programs require top-level named segments.")
            name = segment["name"]
            if name in segments:
                raise ValueError(f"Duplicate fitted program segment: {name}.")
            steps = segment["steps"]
            segments[name] = (
                index,
                steps.get("call", {}) if steps.get("kind") == "invoke" else {},
            )
        selected = segments.get(segment_bindings["segment"])
        if selected is None or selected[1].get("kind") != "pick":
            raise ValueError("Expected one authored standard Pick for the fit.")
        if selected[1]["object"] != goal["object_id"]:
            raise ValueError("The authored Pick binds a different entity.")
        demos = _bound_demos(
            root, segments, integration, retained, pickup, segment_bindings, arrays, dt
        )
        alias = pickup["annotation"]["id"]
        if not alias or "/" in alias or ":" in alias:
            raise ValueError("Pickup annotation must have a portable identifier.")
        preset_id = f"fit-{alias}"
        presets = integration["profile"].setdefault("invocation_presets", {})
        preset = {
            "skill_options": {"pick": {"kind": "pick_up", **options}},
            "motion": motion,
        }
        if preset_id in presets and presets[preset_id] != preset:
            raise ValueError("Fit preset identifier already has a different policy.")
        presets[preset_id] = preset
        integration_path.write_text(yaml.safe_dump(integration, sort_keys=False))
        program, manifest, integration_hash = _compile(root)
        compiled_segments = {
            row.source_path: row for row in program.iter_segments() if not row.implicit
        }

        def compiled_call(name):
            if name not in segments:
                raise ValueError(f"Unknown fitted invocation segment: {name}.")
            index, _ = segments[name]
            owner = compiled_segments[("program", "items", index)]
            (entry,) = owner.calls
            identity = (
                f"{program.program_id}/{owner.segment_id}:{entry.segment_call_index}"
            )
            return entry, identity

        entry, invocation = compiled_call(segment_bindings["segment"])
        if (
            entry.call.semantic_id != "pick"
            or entry.call.object.entity_id != goal["object_id"]
        ):
            raise ValueError(
                "The compiled Pickup occurrence differs from the fitted object binding."
            )
        for demo in demos:
            _, demo["invocation_id"] = compiled_call(demo["program_segment"])
        owners = {
            (demo["start"], demo["stop"]): demo["invocation_id"] for demo in demos
        }
        owners[(pickup["start"], pickup["stop"])] = invocation
        source_events = []
        for event in segment_bindings.get("source_events", ()):
            tick = event["control_tick"]
            matching = [
                owner for (start, stop), owner in owners.items() if start <= tick < stop
            ]
            if len(matching) != 1:
                raise ValueError(
                    f"Source event {event['event_id']} has no unique control interval."
                )
            source_events.append({**event, "invocation_id": matching[0]})
        fingerprint = task_program_fingerprint(program)
        bindings = [InvocationPolicyBinding(fingerprint, invocation, preset_id)]
        for authored_name, name in segment_bindings.get(
            "companion_presets", {}
        ).items():
            _, identity = compiled_call(authored_name)
            bindings.append(InvocationPolicyBinding(fingerprint, identity, name))
        BoundInvocationPolicies(program, bindings).validate_manifest(program, manifest)
        _write(
            root / "source_map.json",
            {
                "timeline": timeline,
                "pickup_invocation": invocation,
                "retained_demos": demos,
                "source_events": source_events,
                **(
                    {"continuation": segment_bindings["continuation"]}
                    if "continuation" in segment_bindings
                    else {}
                ),
            },
        )
        _write(
            root / "compile_report.json",
            {
                "accepted": True,
                "source_joint_sha256": report["source_joint_sha256"],
                "annotation_sha256": report["annotation_sha256"],
                "observed_grasp_sha256": report.get("observed_grasp_sha256"),
                "fit_archive_sha256": segment_bindings["fit_archive_sha256"],
                "object_to_tcp": relation.tolist(),
                "control_dt_s": dt,
                "approved_option_overrides": segment_bindings["option_overrides"],
                "approved_motion_overrides": segment_bindings["motion_overrides"],
                "physical_validation": "pending_native_execution",
            },
        )
        _write(
            root / "invocation_bindings.json",
            {
                "schema": _BINDING_SCHEMA,
                "integration_fingerprint": integration_hash,
                "bindings": [asdict(binding) for binding in bindings],
            },
        )
        _write(
            root / "manifest.json",
            {
                "schema": _BINDING_SCHEMA,
                "files": {
                    item.relative_to(root).as_posix(): file_digest(item)
                    for item in sorted(root.rglob("*"))
                    if item.is_file() and item != root / "manifest.json"
                },
            },
        )
        load_invocation_bindings(root / "deployment.yaml")
        root.rename(output)
    return output


def load_invocation_bindings(deployment: Path) -> tuple[InvocationPolicyBinding, ...]:
    """Verify ownership of all files and validate policies against a fresh compile.

    Args:
        deployment: Path to the Task Program deployment.

    Returns:
        Validated bindings in declaration order, or an empty tuple when the
        deployment has no fitted invocation bindings.
    """
    from embodichain.lab.task_program.integrations.invocation_policy import (
        BoundInvocationPolicies,
        InvocationPolicyBinding,
    )

    root = deployment.resolve(strict=True).parent
    path = root / "invocation_bindings.json"
    if not path.is_file():
        return ()
    if deployment.resolve(strict=True) != root / "deployment.yaml":
        raise ValueError(
            "Fitted invocation bindings require their sealed deployment.yaml."
        )
    seal = json.loads((root / "manifest.json").read_text())
    declaration = json.loads(path.read_text())
    if seal["schema"] != _BINDING_SCHEMA or declaration["schema"] != _BINDING_SCHEMA:
        raise ValueError("Unsupported fitted Task Program binding.")
    for required in (
        "deployment.yaml",
        "invocation_bindings.json",
        "source_map.json",
        "compile_report.json",
    ):
        if required not in seal["files"]:
            raise ValueError(
                f"Missing sealed fitted Task Program artifact: {required}."
            )
    for name, digest in seal["files"].items():
        candidate = (root / name).resolve(strict=True)
        if not candidate.is_relative_to(root) or file_digest(candidate) != digest:
            raise ValueError(f"Fitted Task Program artifact has changed: {name}.")
    program, manifest, integration_hash = _compile(root)
    if declaration["integration_fingerprint"] != integration_hash:
        raise ValueError("Fitted Task Program integration has changed.")
    bindings = tuple(
        InvocationPolicyBinding(**item) for item in declaration["bindings"]
    )
    BoundInvocationPolicies(program, bindings).validate_manifest(program, manifest)
    return bindings
