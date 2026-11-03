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

"""Evidence-driven acceptance and a portable, offline showcase index."""

from __future__ import annotations
from collections.abc import Mapping
import json
from pathlib import Path
import shutil
from ._cases import dump, HERE

__all__: list[str] = []


def read(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def portable_export_check(root: Path, directory: Path, marker: dict) -> dict:
    """Move the real export, hide its source job, and verify independent evidence."""
    from embodichain.data_pipeline.datasets.inspection import (
        validate_dataset,
        create_split_manifest,
    )
    from embodichain.lab.expansion import ArtifactEpisodeSink, load_expansion_episode
    from embodichain.lab.expansion.visual import inspect_observation_variants
    from embodichain.lab.expansion.artifacts import resolve_artifact_path
    from embodichain.lab.expansion.persistence import _source_references
    from embodichain.data_pipeline.recording.media import MediaFrameReader, _digest
    import io
    import numpy as np
    import pyarrow.dataset as ds
    from PIL import Image

    dataset = directory / marker["dataset_path"]
    destination = root / "diagnostics/portability" / directory.name
    hidden = root / "diagnostics" / ("hidden-" + directory.name)
    if destination.exists() or hidden.exists():
        raise RuntimeError("unfinished relocation audit requires recovery before rerun")
    destination.parent.mkdir(parents=True, exist_ok=True)
    intent = root / "diagnostics/portability-move.json"
    dump(
        intent,
        {
            "job": str(directory.relative_to(root)),
            "dataset": str(dataset.relative_to(root)),
            "destination": str(destination.relative_to(root)),
            "hidden": str(hidden.relative_to(root)),
        },
    )
    dataset.rename(destination)
    try:
        directory.rename(hidden)
        try:
            validation = validate_dataset(
                destination, check_media=True, media_frame_counts=True
            ).to_dict()
            if not validation["ok"]:
                raise ValueError(validation)
            bundle = destination / "embodichain_artifacts"
            parents = ArtifactEpisodeSink(bundle).inspect()
            children = inspect_observation_variants(bundle)
            arrow_dataset = ds.dataset(str(destination / "data"), format="parquet")
            rows = [
                json.loads(v)
                for v in (destination / "meta/embodichain_episodes.jsonl")
                .read_text()
                .splitlines()
            ]
            checked = []
            for row in rows:
                episode = load_expansion_episode(
                    resolve_artifact_path(destination, row["source_artifact"])
                )
                for ref in _source_references(row):
                    if (
                        _digest(resolve_artifact_path(destination, ref["path"]))
                        != ref["sha256"]
                    ):
                        raise ValueError("relocated source hash differs")
                columns = ["action", "observation.state"]
                media_reader = None
                rgb = None
                image_key = None
                if row.get("media_manifest"):
                    media = resolve_artifact_path(destination, row["media_manifest"])
                    manifest = json.loads(media.read_text())
                    assert manifest["calibration"] == row["camera_calibration"]
                    rgb = next(
                        k
                        for k in manifest["features"]
                        if k.endswith(("/rgb", "/color"))
                    )
                    image_key = next(
                        k
                        for k in arrow_dataset.schema.names
                        if k.startswith("observation.images.")
                    )
                    columns.append(image_key)
                    media_reader = MediaFrameReader(media, fields=(rgb,))
                offset = 0
                sample_indices = {
                    0,
                    len(episode.actions) // 2,
                    len(episode.actions) - 1,
                }
                try:
                    scanner = arrow_dataset.scanner(
                        columns=columns,
                        filter=ds.field("episode_index")
                        == row["lerobot_episode_index"],
                        batch_size=64,
                        batch_readahead=1,
                        fragment_readahead=1,
                        use_threads=False,
                    )
                    for batch in scanner.to_batches():
                        count = batch.num_rows
                        if count == 0:
                            continue
                        np.testing.assert_array_equal(
                            np.asarray(
                                batch.column("action").to_pylist(), dtype=np.float32
                            ),
                            episode.actions[offset : offset + count].numpy(),
                        )
                        np.testing.assert_array_equal(
                            np.asarray(
                                batch.column("observation.state").to_pylist(),
                                dtype=np.float32,
                            ),
                            episode.observations["joint_positions"][
                                offset : offset + count
                            ].numpy(),
                        )
                        if media_reader:
                            for index in sorted(sample_indices):
                                if offset <= index < offset + count:
                                    value = batch.column(image_key)[
                                        index - offset
                                    ].as_py()
                                    image = (
                                        Image.open(io.BytesIO(value["bytes"]))
                                        if value.get("bytes") is not None
                                        else Image.open(
                                            resolve_artifact_path(
                                                destination, value["path"]
                                            )
                                        )
                                    )
                                    np.testing.assert_array_equal(
                                        np.asarray(image),
                                        media_reader.read(index)[1][rgb][..., :3],
                                    )
                        offset += count
                    assert offset == len(
                        episode.actions
                    ), "exported action count differs"
                finally:
                    if media_reader:
                        media_reader.close()
                checked.append(row["episode_uuid"])
            split = create_split_manifest(
                destination,
                root / "diagnostics/portability" / (directory.name + ".split.json"),
                seed=7,
                group_by="lineage",
            )
            assignments = {
                i: group["split"]
                for group in split["groups"]
                for i in group["episode_indices"]
            }
            ancestors = {}
            for row in rows:
                for ancestor in row.get("source_episode_uuids", []) or [
                    row.get("source_episode_uuid", row["episode_uuid"])
                ]:
                    ancestors.setdefault(ancestor, set()).add(
                        assignments[row["lerobot_episode_index"]]
                    )
            assert all(
                len(s) == 1 for s in ancestors.values()
            ), "source descendants cross split boundaries"
            return {
                "ok": True,
                "dataset": validation,
                "physical_parents": len(parents),
                "visual_children": len(children),
                "exact_action_state_checks": len(checked),
                "rgb_samples_per_episode": 3,
                "calibration_and_source_refs": True,
                "lineage_grouping": True,
                "source_groups": len(ancestors),
            }
        finally:
            hidden.rename(directory)
    finally:
        if destination.exists():
            destination.rename(dataset)
        if directory.exists() and dataset.exists():
            intent.unlink(missing_ok=True)


def visual_difference_checks(receipts, variants) -> list[dict]:
    from embodichain.lab.expansion import load_expansion_episode
    from embodichain.data_pipeline.recording.media import MediaFrameReader
    import numpy as np

    parents = {r.episode_id: load_expansion_episode(r.storage_id) for r in receipts}
    checks = []
    for variant in variants:
        parent = parents[variant["parent_episode_id"]]
        original = parent.metadata["media_manifest"]
        before = read(Path(original))
        after = read(Path(variant["media_manifest"]))
        field = next(
            k
            for k in before["features"]
            if k.startswith("sensor/") and k.endswith(("/rgb", "/color"))
        )
        differences = []
        with (
            MediaFrameReader(original, fields=(field,)) as left,
            MediaFrameReader(variant["media_manifest"], fields=(field,)) as right,
        ):
            for index in (0, len(parent.actions) // 2, len(parent.actions)):
                lt, a = left.read(index)
                rt, b = right.read(index)
                assert abs(lt - rt) < 1e-9
                differences.append(
                    float(
                        np.abs(
                            a[field].astype(np.float32) - b[field].astype(np.float32)
                        ).mean()
                    )
                )
        assert (
            max(differences) > 0
        ), "visual profile produced no measured RGB difference"
        camera_changed = before["calibration"] != after["calibration"]
        if variant.get("profile", {}).get("cameras"):
            assert camera_changed, "camera profile did not change measured calibration"
        checks.append(
            {
                "variant_id": variant["variant_id"],
                "profile": variant["profile_id"],
                "mean_absolute_rgb_differences": differences,
                "camera_calibration_changed": camera_changed,
                "physics_steps": variant["physics_steps"],
            }
        )
    return checks


def audit(root: Path) -> dict:
    from embodichain.lab.expansion import (
        ArtifactEpisodeSink,
        load_expansion_episode,
        inspect_expansion_job,
    )
    from embodichain.lab.expansion.visual import inspect_observation_variants
    from embodichain.data_pipeline.datasets.inspection import validate_dataset
    import numpy as np

    cases = read(root / "cases.json")["cases"]
    checks = []
    batch_recipes = {}
    baseline_parameters = {}
    portable = []
    for case in cases:
        directory = root / "jobs" / case["id"]
        if not (directory / "job.json").exists():
            result = read(root / "results" / (Path(case["job"]).stem + ".json"), {})
            timeout = read(
                root / "results" / (Path(case["job"]).stem + ".timeout.json")
            )
            attempted = bool(result.get("error") or timeout)
            checks.append(
                {
                    "id": case["id"],
                    "status": "failed" if attempted else "not_run",
                    "required": case["required"],
                    "reason": result.get("error")
                    or ("worker timeout" if timeout else "No measured job"),
                }
            )
            continue
        try:
            summary = inspect_expansion_job(directory)
            receipts = ArtifactEpisodeSink(directory).inspect()
            variants = inspect_observation_variants(directory)
            visual_checks = (
                visual_difference_checks(receipts, variants) if variants else []
            )
            recipes = []
            parameters = []
            expansion = []
            scene_ids = set()
            scene_poses = []
            selected_operators = set()
            selected_branches = set()
            grasp_poses = []
            source_recipes = set()
            phase_evidence = []
            requested_physics = []
            for receipt in receipts:
                episode = load_expansion_episode(receipt.storage_id)
                assert (
                    len(episode.timestamps) == len(episode.actions) + 1
                ), "T/T+1 contract"
                assert len(episode.observations["joint_positions"]) == len(
                    episode.timestamps
                )
                assert np.all(
                    np.diff(episode.timestamps.numpy()) > 0
                ), "nonmonotonic clock"
                assert episode.validation.accepted, "receipt for rejected evidence"
                parameters.append(dict(episode.metadata.get("physics_parameters", {})))
                records = episode.metadata.get("expansion", ())
                expansion.append(records)
                for record in records:
                    selected = record.get("trajectory_selected")
                    for variant in record.get("trajectory_variants", ()):
                        if variant.get("variant_ordinal") == selected:
                            selected_operators.add(variant["spatial_operator"])
                if "affordance" in case["tags"]:
                    physical_rows = {r["env_id"] for r in records}
                    assert (
                        len(physical_rows) == 1
                    ), "grasp evidence needs one physical row identity"
                    physical_row = next(iter(physical_rows))
                    for segment in episode.metadata["segments"]:
                        runtime = segment.get("metadata", {}).get(
                            "runtime", segment.get("runtime", {})
                        )
                        for call in runtime.get("calls", ()):
                            for attempt in call.get("plan_attempts", ()):
                                sample = (
                                    attempt.get("planner_diagnostics", {})
                                    .get("metadata", {})
                                    .get("affordance_sample", {})
                                    .get("grasp")
                                )
                                if sample and physical_row in sample["env_ids"]:
                                    row = sample["env_ids"].index(physical_row)
                                    if sample["success"][row]:
                                        selected_branches.add(
                                            sample["candidate_ids"][row]
                                        )
                                        grasp_poses.append(
                                            np.asarray(sample["selected_poses"][row])
                                        )
                if case["source"] == "carton":
                    lineage = episode.metadata.get("real2sim_lineage", {})
                    assert lineage.get(
                        "calls"
                    ), "prototype episode lacks actual source lineage"
                    selected_operators.update(
                        call["operator"] for call in lineage["calls"]
                    )
                    source_recipes.add(episode.metadata["recipe"])
                    phases = episode.metadata.get("phase_duration_validation", ())
                    assert phases and all(
                        p["passed"] for p in phases
                    ), "source phase validation missing or failed"
                    assert all(
                        p["final_samples"] == p["source_samples"]
                        for p in phases
                        if not p["retimed"]
                    ), "protected phase length changed"
                    phase_evidence.extend(dict(p) for p in phases)
                sampled = episode.metadata.get(
                    "expansion_recipe", episode.metadata.get("recipe", {})
                )
                if isinstance(sampled, Mapping):
                    requested_physics.append(sampled.get("physics", {}))
                if isinstance(sampled, Mapping) and sampled.get("scene_candidate_id"):
                    scene_ids.add(sampled["scene_candidate_id"])
                    scene_poses.append(
                        [
                            value
                            for field in episode.metadata["scene_state_fields"]
                            if field["field"] == "pose"
                            for value in episode.observations[field["key"]][0].tolist()
                        ]
                    )
                recipes.append(
                    (
                        episode.metadata["recipe_spec"]["context"]["ordinal"],
                        episode.metadata["recipe_id"],
                    )
                )
            expected = {
                "carton-joint_7": ("joint_7", "joint_residual"),
                "carton-joint_11": ("joint_11", "joint_residual"),
                "carton-task_space": ("task_space", "task_space_residual"),
                "carton-time_115": ("time_115", "retime"),
                "carton-time_130": ("time_130", "retime"),
            }.get(case["id"])
            if expected:
                assert (
                    expected[0] in source_recipes and expected[1] in selected_operators
                ), "requested prototype operator has no admitted evidence"
                if expected[1] == "retime":
                    assert any(
                        p["retimed"]
                        and p["final_samples"] != p["source_samples"]
                        and p["final_samples"] == p["expected_samples"]
                        for p in phase_evidence
                    ), "no measured retimed free phase"
            if "combined" in case["tags"]:
                assert selected_operators - {
                    "none",
                    "nominal",
                }, "combined job has no admitted motion augmentation"
                body = "cube" if case["source"] == "ur5" else "carton"
                base = baseline_parameters[(case["source"], case["backend"])][
                    body + ".mass"
                ]
                for requested, measured in zip(requested_physics, parameters):
                    assert body in requested.get(
                        "mass_scale", {}
                    ), "combined job has no declared physical augmentation"
                    np.testing.assert_allclose(
                        measured[body + ".mass"] / base,
                        requested["mass_scale"][body],
                        rtol=1e-5,
                        atol=1e-7,
                    )
                assert (
                    len(
                        {
                            json.dumps(dict(v), sort_keys=True, default=dict)
                            for v in requested_physics
                        }
                    )
                    > 1
                ), "combined job did not sample distinct physical conditions"
            for operator in ("joint_residual", "via_points"):
                if operator in case["tags"] and operator not in selected_operators:
                    raise ValueError(
                        f"No admitted {operator} execution; nominal commits cannot qualify an operator"
                    )
            if "affordance" in case["tags"] and (
                len(selected_branches) < 2
                or not any(
                    np.max(np.abs(p - grasp_poses[0])) > 1e-4 for p in grasp_poses[1:]
                )
            ):
                raise ValueError("No two admitted distinct grasp candidates")
            if "scene" in case["tags"]:
                if len(scene_ids) < 2 or not any(
                    np.max(np.abs(np.asarray(v) - np.asarray(scene_poses[0]))) > 1e-4
                    for v in scene_poses[1:]
                ):
                    raise ValueError("No measured joint scene/installation diversity")
            if "baseline" in case["tags"] and parameters:
                baseline_parameters[(case["source"], case["backend"])] = parameters[0]
            if (
                "physics" in case["tags"]
                and "combined" not in case["tags"]
                and parameters
            ):
                baseline = baseline_parameters.get((case["source"], case["backend"]))
                if baseline is None:
                    raise ValueError("No measured baseline for parameter comparison")
                suffix = {
                    "mass": ".mass",
                    "friction": ".friction",
                    "com": ".com_pose",
                    "stiffness": ".joint_stiffness",
                    "damping": ".joint_damping",
                }[case["tags"][-1]]
                changed = any(
                    not np.array_equal(
                        np.asarray(values[key]), np.asarray(baseline[key])
                    )
                    for values in parameters
                    for key in values
                    if key.endswith(suffix)
                )
                assert (
                    changed
                ), "requested physical dimension did not change in readback"
                if "mass" in case["tags"]:
                    for measured in parameters:
                        ratio = measured["cube.mass"] / baseline["cube.mass"]
                        np.testing.assert_allclose(
                            measured["cube.inertia"],
                            np.asarray(baseline["cube.inertia"]) * ratio,
                            rtol=1e-5,
                            atol=1e-9,
                        )
            if "batch" in case["tags"]:
                catalogue = read(directory / "recipe-catalog.json")
                assert (
                    catalogue and catalogue["sampled_only"]
                ), "missing recipe catalogue"
                batch_recipes[case["id"]] = {
                    r["context"]["ordinal"]: r["recipe_id"]
                    for r in catalogue["recipes"]
                }
                cohorts = [
                    read(path)
                    for path in sorted((directory / "cohorts").glob("*.json"))
                ]
                executed = [c for c in cohorts if c["state"] == "executed"]
                assert any(
                    len(c["started_rows"]) == case["batch_size"] for c in executed
                ), "no full native cohort"
                if case["batch_size"] > 1:
                    assert any(
                        len(c["started_rows"]) == 1 for c in executed
                    ), "no partial final cohort"
                assert all(
                    len(set(c["started_rows"])) == len(c["started_rows"])
                    for c in executed
                )
            dataset = None
            marker = read(directory / "lerobot-export.json")
            if marker:
                dataset = validate_dataset(
                    directory / marker["dataset_path"],
                    check_media=True,
                    media_frame_counts=True,
                ).to_dict()
                assert dataset["ok"], "dataset validation failed"
                portable_result = portable_export_check(root, directory, marker)
                dump(
                    root / "diagnostics/portability" / (case["id"] + ".json"),
                    portable_result,
                )
                portable.append(
                    {
                        "id": "portable-" + case["id"],
                        "status": "passed",
                        "required": True,
                        "evidence": portable_result,
                    }
                )
            checks.append(
                {
                    "id": case["id"],
                    "status": "passed" if summary["complete"] else "not_qualified",
                    "required": case["required"],
                    "summary": summary,
                    "verified_parents": len(receipts),
                    "verified_children": len(variants),
                    "selected_operators": sorted(selected_operators),
                    "source_recipes": sorted(source_recipes),
                    "phase_evidence": phase_evidence,
                    "visual_checks": visual_checks,
                    "selected_grasp_branches": sorted(selected_branches),
                    "joint_scene_candidates": len(scene_ids),
                    "dataset": dataset,
                }
            )
        except Exception as error:
            checks.append(
                {
                    "id": case["id"],
                    "status": "failed",
                    "required": case["required"],
                    "reason": str(error),
                }
            )
    checks.extend(portable)
    if len(batch_recipes) == 3:
        values = list(batch_recipes.values())
        common = set.intersection(*(set(v) for v in values))
        equal = bool(common) and all(len({v[i] for v in values}) == 1 for i in common)
        checks.append(
            {
                "id": "batch-recipe-identity",
                "required": True,
                "status": "passed" if equal else "failed",
                "shared_ordinals": sorted(common),
                "sampling_only_outside_executed_ordinals": True,
            }
        )
    for name in ("probe-default", "probe-newton", "faults"):
        result = read(root / "results" / (name + ".json"))
        checks.append(
            {
                "id": name,
                "required": True,
                "status": (
                    "not_run"
                    if result is None
                    else "passed" if result.get("ok") else "failed"
                ),
                "evidence": result,
            }
        )
    report = {
        "schema_version": 1,
        "checks": checks,
        "report_complete": all(c["status"] != "not_run" for c in checks),
        "acceptance_passed": all(
            c["status"] == "passed" for c in checks if c["required"]
        ),
    }
    dump(root / "acceptance.json", report)
    return {
        "ok": report["acceptance_passed"],
        "report_complete": report["report_complete"],
        "checks": len(checks),
    }


def failure_details(value) -> list[str]:
    """Retain the actual runtime cause instead of only a generic segment status."""
    messages = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if (
                key in {"message", "failure_reason", "reason", "error"}
                and isinstance(item, str)
                and item
            ):
                messages.append(item)
            messages.extend(failure_details(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            messages.extend(failure_details(item))
    return list(dict.fromkeys(messages))


def preview_joint_types(root: Path, case: dict, episode: dict) -> list[str | None]:
    """Freeze explicit joint types for old previews without creating a simulator."""
    names = episode["metadata"]["joint_names"]
    if episode.get("joint_types") is not None:
        return episode["joint_types"]
    path = root / "replay" / (episode["model"] + ".joints.json")
    saved = read(path)
    if saved and saved["joint_names"] == names:
        return saved["joint_types"]
    job = read(root / "jobs" / case["id"] / "job.json", {})
    robot = job.get("resolved_deployment", {}).get("robot", {})
    source = robot.get("fpath") or robot.get("urdf_cfg", {}).get("fpath")
    if not source or not Path(source).is_file():
        return [None] * len(names)
    import hashlib
    import xml.etree.ElementTree as ET

    raw = Path(source).read_bytes()
    kinds = {
        j.attrib["name"]: j.attrib.get("type")
        for j in ET.fromstring(raw).findall("joint")
    }
    result = [kinds.get(name) for name in names]
    dump(
        path,
        {
            "joint_names": names,
            "joint_types": result,
            "source": "recorded deployment URDF",
            "urdf_sha256": hashlib.sha256(raw).hexdigest(),
        },
    )
    return result


def build_page(root: Path) -> dict:
    inventory = read(root / "cases.json", {"mode": "full", "cases": []})
    acceptance = read(
        root / "acceptance.json",
        {
            "schema_version": 1,
            "checks": [],
            "report_complete": False,
            "acceptance_passed": False,
        },
    )
    cases = []
    episodes = []
    for case in inventory["cases"]:
        result = read(root / "results" / (Path(case["job"]).stem + ".json"), {})
        checked = next((v for v in acceptance["checks"] if v["id"] == case["id"]), {})
        summary = checked.get("summary", result.get("result", {}))
        row = {
            **case,
            "summary": summary,
            "error": result.get("error"),
            "status": next(
                (v["status"] for v in acceptance["checks"] if v["id"] == case["id"]),
                (
                    "collected"
                    if result.get("ok")
                    else "failed" if result.get("error") else "not_run"
                ),
            ),
            "runtime": read(root / "jobs" / case["id"] / "runtime.json", {}),
            "telemetry": result.get("telemetry"),
            "coverage_report": read(root / "jobs" / case["id"] / "coverage.json", {}),
            "attempts": [],
        }
        for path in sorted(
            (root / "jobs" / case["id"] / "attempts").glob("*/attempt.json")
        ):
            attempt = read(path)
            row["attempts"].append(
                {
                    **{
                        k: attempt.get(k)
                        for k in (
                            "sequence",
                            "ordinal",
                            "status",
                            "reason",
                            "steps",
                            "attempt_id",
                            "recipe_id",
                            "commit_id",
                        )
                    },
                    "details": (
                        failure_details(attempt.get("program_segments", ()))
                        if attempt.get("status") not in {"committed", "accepted"}
                        else []
                    ),
                    "evidence": str(path.relative_to(root)),
                }
            )
        for episode in read(root / "replay" / (case["id"] + ".json"), []):
            metadata = episode["metadata"]
            episode = {
                **episode,
                "joint_types": preview_joint_types(root, case, episode),
                "initial_poses": {
                    entity["uid"]: entity["pose"]
                    for entity in metadata.get("initial_state", ())
                },
                "metadata": {
                    k: metadata[k]
                    for k in (
                        "recipe_id",
                        "recipe_spec",
                        "expansion_recipe",
                        "recipe",
                        "physics_parameters",
                        "expert_action_spec",
                        "joint_names",
                        "source_episode_uuid",
                        "source_episode_uuids",
                        "motion_origin",
                        "action_origin",
                        "source_origin",
                        "visual_profile",
                        "visual_profile_revision",
                        "source_binding",
                        "real2sim_lineage",
                        "retime_phase_allowlist",
                        "expansion",
                    )
                    if k in metadata
                },
            }
            episodes.append(episode)
        cases.append(row)
    data = {
        "schema_version": 1,
        "title": "Expansion Engine · 专家轨迹扩增实验室",
        "mode": inventory["mode"],
        "cases": cases,
        "episodes": episodes,
        "acceptance": acceptance,
        "boundaries": [
            {
                "name": "Real2Sim 场景 atlas 变化",
                "status": "unsupported",
                "reason": "需要来源资格覆盖的布局配置；本例保留固定原型布局",
            },
            {
                "name": "单体 Newton damping",
                "status": "unsupported",
                "reason": "当前后端未声明有效；关节驱动 damping 单独验收",
            },
            {
                "name": "动态资产拓扑替换",
                "status": "separate_world",
                "reason": "需要独立部署与 World，不在同一批中变更",
            },
            {
                "name": "原生通用 retiming / nullspace",
                "status": "not_integrated",
                "reason": "当前原生 Engine profile 未接入；映射 retiming 在原型来源案例展示",
            },
        ],
    }
    dump(root / "showcase.json", data)
    template = (HERE / "viewer.html").read_text()
    payload = json.dumps(
        data, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).replace("<", "\\u003c")
    import hashlib

    for script in ("comparison.js", "viewer.js"):
        revision = hashlib.sha256((HERE / script).read_bytes()).hexdigest()[:12]
        template = template.replace(f'src="{script}"', f'src="{script}?v={revision}"')
        shutil.copyfile(HERE / script, root / script)
    (root / "index.html").write_text(template.replace("__SHOWCASE_DATA__", payload))
    return acceptance
