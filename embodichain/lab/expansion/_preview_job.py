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

"""Freeze endpoint/interior experiments and collect them through the native CLI."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from uuid import uuid4
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from .configured import load_expansion_job, preflight_expansion_job
from .persistence import ArtifactEpisodeSink, load_expansion_episode

__all__: list[str] = []


def boundary_samples(
    dimensions: int, interior: int, seed: int
) -> tuple[np.ndarray, list[str]]:
    if (
        type(dimensions) is not int
        or dimensions < 0
        or type(interior) is not int
        or not 1 <= interior <= 61
    ):
        raise ValueError("interior samples must be in [1,61]")
    if not dimensions:
        return np.empty((1, 0)), ["fixed"]
    rng = np.random.default_rng(seed)
    points = [np.full(dimensions, 0.5), np.zeros(dimensions), np.ones(dimensions)]
    # Independent Latin-hypercube strata in every varying coordinate.
    inside = np.stack(
        [
            (rng.permutation(interior) + rng.uniform(0.05, 0.95, interior)) / interior
            for _ in range(dimensions)
        ],
        axis=1,
    )
    points.extend(inside)
    return np.asarray(points), ["center", "lower", "upper"] + [
        f"interior-{i+1}" for i in range(interior)
    ]


def _atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def plan_preview(
    job: Path, output: Path | None, *, interior: int = 5, max_episodes: int = 32
) -> dict[str, Any]:
    from embodichain.lab.sim.motion.expansion.profile import load_expansion_profile
    from embodichain.lab.sim.motion.expansion.combined import (
        enumerate_candidate_recipes,
    )

    if type(max_episodes) is not int or not 1 <= max_episodes <= 64:
        raise ValueError("max_episodes must be in [1,64]")
    cfg = load_expansion_job(job)
    report = preflight_expansion_job(cfg)
    if cfg.source != "task_program" or report.get("motion_source") != "native":
        raise ValueError(
            "preview --job currently requires native Task Program planning"
        )
    if cfg.visual_profile or cfg.source_profile:
        raise ValueError(
            "Distribution preview does not sample visual or prototype profiles"
        )
    axes = []

    def axis(name: str, lower: float, upper: float, kind: str, **details: Any) -> None:
        if lower != upper:
            axes.append(
                dict(
                    name=name,
                    lower=float(lower),
                    upper=float(upper),
                    kind=kind,
                    **details,
                )
            )

    for kind in (
        "mass_scale",
        "friction",
        "joint_stiffness_scale",
        "joint_damping_scale",
    ):
        for uid, bounds in sorted(getattr(cfg, kind).items()):
            axis(f"{kind}/{uid}", *bounds, kind, uid=uid)
    for uid, ranges in sorted(cfg.center_of_mass.items()):
        for i, bounds in enumerate(ranges):
            axis(
                f"center_of_mass/{uid}/{i}", *bounds, "center_of_mass", uid=uid, index=i
            )
    profile = None
    motion = None
    recipe_count = 1
    if cfg.motion_profile:
        profile = load_expansion_profile(cfg.motion_profile)
        motion = yaml.safe_load(Path(cfg.motion_profile).read_text())
        recipe_count = len(enumerate_candidate_recipes(profile))
        if profile.scene_randomization.reference_family_count != 1:
            raise ValueError(
                "Use a scene atlas for spatial bounds, not implicit motion families"
            )
        spatial = profile.trajectory.spatial
        if spatial.enabled and profile.trajectory.variants_per_family > 1:
            if profile.trajectory.variants_per_family < len(spatial.operators) + 1:
                raise ValueError(
                    "Preview needs a nominal variant plus at least one variant per configured operator"
                )
            if set(spatial.operators) - {"joint_residual", "via_points"}:
                raise ValueError(
                    "Explicit boundary draws support joint_residual and via_points"
                )
            robot = report["resolved_deployment"]["robot"]
            groups = robot["control_parts"]
            parts = robot.get("solver_cfg", {}).keys()
            joints = sorted({name for part in parts for name in groups.get(part, [])})
            if not joints or any(not isinstance(j, str) for j in joints):
                raise ValueError(
                    "Named solver control joints are required to bound motion draws"
                )
            for name in joints:
                if name not in spatial.joint_samples:
                    axis(f"joint_residual_fraction/{name}", -1, 1, "joint", uid=name)
            if "via_points" in spatial.operators and not spatial.knot_samples:
                for i in range(spatial.via_count):
                    axis(f"via_knot_fraction/{i}", -1, 1, "knot", index=i)
    prepared = None
    deployment = None
    if cfg.scene_atlas:
        from embodichain.lab.sim.scene.feasibility import load_feasibility_atlas
        from embodichain.lab.sim.scene.feasibility.search import (
            PreparedFeasibilitySearch,
        )
        from embodichain.lab.sim.scene.feasibility.cfg import SceneRobotFeasibilityCfg
        from embodichain.lab.gym.utils.composition import resolve_gym_components

        atlas = load_feasibility_atlas(cfg.scene_atlas)
        if len(atlas.request.modes) != 1:
            raise ValueError("Preview requires one task mode per scene atlas")
        prepared = PreparedFeasibilitySearch(
            atlas.request, SceneRobotFeasibilityCfg(seed=cfg.seed)
        )
        for i, var in enumerate(atlas.request.variables):
            for j, (lo, hi) in enumerate(zip(var.lower, var.upper)):
                axis(
                    f'scene/{var.name}/{("x","y","yaw")[j]}',
                    lo,
                    hi,
                    "scene",
                    index=i * 3 + j,
                )
        raw = yaml.safe_load(Path(cfg.deployment).read_text())
        deployment = resolve_gym_components(
            raw, base_dir=Path(cfg.deployment).parent
        ).config
        for key in ("program", "integration", "execution_policy"):
            value = deployment.get("task_program", {}).get(key)
            if isinstance(value, str):
                deployment["task_program"][key] = str(
                    (Path(cfg.deployment).parent / value).resolve()
                )
            elif isinstance(value, dict) and isinstance(value.get("component"), str):
                value["component"] = str(
                    (Path(cfg.deployment).parent / value["component"]).resolve()
                )
        if "embodiment" in raw:
            deployment.pop("robot", None)
            deployment.pop("sensor", None)
            embodiment = deepcopy(raw["embodiment"])
            embodiment["component"] = str(
                (Path(cfg.deployment).parent / embodiment["component"]).resolve()
            )
            deployment["embodiment"] = embodiment
    points, roles = boundary_samples(len(axes), interior, cfg.seed)
    counts = []
    for unit in points:
        samples = dict(profile.trajectory.spatial.joint_samples) if profile else {}
        for a, u in zip(axes, unit):
            if a["kind"] == "joint":
                samples[a["uid"]] = a["lower"] + float(u) * (a["upper"] - a["lower"])
        counts.append(
            profile.affordance.branches_per_family
            if profile and samples and all(v == 0 for v in samples.values())
            else recipe_count
        )
    total = sum(counts)
    if total > max_episodes or total > 64 or total > cfg.max_attempts:
        raise ValueError(
            f"Preview needs {len(points)} parameter points and up to {recipe_count} motion/affordance recipes per point = {total} attempts; increase --max-episodes (up to 64) and the job max_attempts budget, reduce --interior-samples or use a smaller recipe. Nothing executed."
        )
    root = (
        Path(output).expanduser().resolve()
        if output
        else Path(cfg.output_dir).parent
        / (
            Path(cfg.output_dir).name
            + "-preview-"
            + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            + "-"
            + uuid4().hex[:6]
        )
    )
    if root.exists():
        raise FileExistsError(
            f"Preview output must be new: {root}; use --input to replay"
        )
    # Validate the renderer's supported scene before any costly collection.
    from ._preview_data import _scene

    supported = deepcopy(report)
    supported["config"] = dict(report.get("config", {}), scene_atlas=None)
    _scene(supported)
    root.mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "logs").mkdir()
    plan = {
        "schema_version": 1,
        "source_job": str(Path(job).resolve()),
        "source_sha256": hashlib.sha256(Path(job).read_bytes()).hexdigest(),
        "source_inputs": report["input_files"],
        "source_atlas": cfg.scene_atlas,
        "seed": cfg.seed,
        "sampling": "lower + upper + center + stratified interior; not a Cartesian grid",
        "axes": axes,
        "planned_attempts": total,
        "recipes_per_point": recipe_count,
        "collection_budget_s": cfg.max_wall_time_s,
        "output": str(root),
        "records": [],
        "complete": False,
        "coverage_complete": False,
    }
    for index, (unit, role) in enumerate(zip(points, roles)):
        values = {
            a["name"]: a["lower"] + float(u) * (a["upper"] - a["lower"])
            for a, u in zip(axes, unit)
        }
        case = cfg.to_dict()
        case.update(
            output_dir=str(root / f"jobs/sample-{index:03d}"),
            target_episodes=recipe_count,
            max_attempts=recipe_count,
            retry_interrupted=False,
        )
        m = deepcopy(motion)
        scene_unit = np.zeros(prepared.dimensions) if prepared else None
        if m:
            spatial = m["trajectory"]["spatial"]
            spatial["joint_samples"] = dict(profile.trajectory.spatial.joint_samples)
            knots = list(
                profile.trajectory.spatial.knot_samples
                or [0.0] * profile.trajectory.spatial.via_count
            )
        for a, u in zip(axes, unit):
            value = values[a["name"]]
            kind = a["kind"]
            if kind == "joint":
                spatial["joint_samples"][a["uid"]] = value
            elif kind == "knot":
                knots[a["index"]] = value
            elif kind == "scene":
                scene_unit[a["index"]] = u
            elif kind == "center_of_mass":
                case[kind][a["uid"]] = [list(v) for v in case[kind][a["uid"]]]
                case[kind][a["uid"]][a["index"]] = [value, value]
            else:
                case[kind][a["uid"]] = [value, value]
        if m:
            if "via_points" in profile.trajectory.spatial.operators:
                spatial["knot_samples"] = knots
            # Exact zero residuals are the nominal trajectory, not extra successes.
            # Keep native deduplication intact by requesting only that unique variant.
            if spatial.get("joint_samples") and all(
                v == 0 for v in spatial["joint_samples"].values()
            ):
                m["trajectory"]["variants_per_family"] = 1
                case["target_episodes"] = case["max_attempts"] = (
                    profile.affordance.branches_per_family
                )
            motion_path = root / "configs" / f"motion-{index:03d}.yaml"
            motion_path.write_text(yaml.safe_dump(m, sort_keys=False))
            case["motion_profile"] = str(motion_path)
        requested_positions = {
            obj["uid"]: obj.get("init_pos", [0, 0, 0])
            for obj in report["resolved_deployment"].get("rigid_object", [])
        }
        if prepared:
            candidate = prepared.candidate(
                scene_unit, mode_id=prepared.request.modes[0].mode_id, ordinal=index
            )
            derived = deepcopy(deployment)
            changes = {
                p.entity_uid: np.asarray(p.arena_pose)
                for p in candidate.scene.pose_changes
            }
            for obj in derived.get(
                "rigid_object", derived.get("simulation", {}).get("rigid_object", [])
            ):
                if obj["uid"] in changes:
                    pose = changes.pop(obj["uid"])
                    obj["init_pos"] = pose[:3, 3].tolist()
                    obj["init_rot"] = (
                        Rotation.from_matrix(pose[:3, :3])
                        .as_euler("xyz", degrees=True)
                        .tolist()
                    )
                    requested_positions[obj["uid"]] = obj["init_pos"]
            if changes:
                raise ValueError(
                    f"Unsupported scene variable entities: {list(changes)}"
                )
            robot_uid = report["resolved_deployment"]["robot"]["uid"]
            for uid, pose in candidate.deployment.poses.items():
                if uid != robot_uid:
                    raise ValueError("Scene variable selects a different robot")
                pose = np.asarray(pose)
                target = (
                    derived["embodiment"].setdefault("overrides", {})
                    if "embodiment" in derived
                    else derived["robot"]
                )
                target["init_pos"] = pose[:3, 3].tolist()
                target["init_rot"] = (
                    Rotation.from_matrix(pose[:3, :3])
                    .as_euler("xyz", degrees=True)
                    .tolist()
                )
            deployment_path = root / "configs" / f"task-{index:03d}.yaml"
            deployment_path.write_text(yaml.safe_dump(derived, sort_keys=False))
            case["deployment"] = str(deployment_path)
            case["scene_atlas"] = None
        config_path = root / "configs" / f"job-{index:03d}.yaml"
        config_path.write_text(yaml.safe_dump(case, sort_keys=False))
        # Same public preflight and acceptance configuration as ordinary collection.
        pinned = preflight_expansion_job(load_expansion_job(config_path))
        plan["records"].append(
            dict(
                semantic_id=pinned["semantic_id"],
                scene_candidate_id=candidate.candidate_id if prepared else None,
                index=index,
                role=role,
                unit=unit.tolist(),
                values=values,
                requested_positions=requested_positions,
                job=str(config_path.relative_to(root)),
                result=str(Path(case["output_dir"]).relative_to(root)),
                status="unrun",
                confirmed=0,
                attempts=0,
                planned=case["target_episodes"],
            )
        )
    plan["planned_attempts"] = sum(r["planned"] for r in plan["records"])
    _atomic(root / "preview-plan.json", plan)
    return plan


def _exact_recipe(metadata: Any) -> bool:
    records = metadata.get("expansion", ())
    if any(
        r.get("fallback_count", 0)
        or (
            r.get("trajectory_requested") is not None
            and r.get("trajectory_requested") != r.get("trajectory_selected")
        )
        for r in records
    ):
        return False
    if not any(r.get("affordance_requested") is not None for r in records):
        return True
    rows = {r["env_id"] for r in records if r.get("env_id") is not None}
    env_id = metadata.get("source_env_id", next(iter(rows)) if len(rows) == 1 else None)
    found = False
    for segment in metadata.get("program_segments", ()):
        for call in segment.get("metadata", {}).get("runtime", {}).get("calls", ()):
            attempts = call.get("plan_attempts", ())
            if not attempts:
                continue
            sample = (
                attempts[-1]
                .get("planner_diagnostics", {})
                .get("metadata", {})
                .get("affordance_sample", {})
                .get("grasp")
            )
            if sample is None:
                continue
            ids = list(sample["env_ids"])
            index = (
                ids.index(env_id)
                if env_id in ids
                else 0 if env_id is None and len(ids) == 1 else None
            )
            if (
                index is None
                or not sample["success"][index]
                or sample.get("reused", [True] * len(ids))[index]
            ):
                return False
            found = True
    return found


def collect_preview(plan: dict[str, Any]) -> dict[str, Any]:
    root = Path(plan["output"])
    started = time.monotonic()
    for record in plan["records"]:
        remaining = plan["collection_budget_s"] - (time.monotonic() - started)
        if remaining <= 0:
            record["status"] = "budget_exhausted"
            continue
        job = root / record["job"]
        cfg = yaml.safe_load(job.read_text())
        cfg["max_wall_time_s"] = min(cfg["max_wall_time_s"], remaining)
        job.write_text(yaml.safe_dump(cfg, sort_keys=False))
        current = preflight_expansion_job(load_expansion_job(job))
        if current["semantic_id"] != record["semantic_id"]:
            record["status"] = "source_changed"
            _atomic(root / "preview-plan.json", plan)
            raise ValueError("Preview inputs changed after planning; create a new plan")
        remaining = plan["collection_budget_s"] - (time.monotonic() - started)
        if remaining <= 0:
            record["status"] = "budget_exhausted"
            _atomic(root / "preview-plan.json", plan)
            continue
        record["status"] = "running"
        _atomic(root / "preview-plan.json", plan)
        print(
            f"Preview sample {record['index']+1}/{len(plan['records'])}: {record['role']}",
            flush=True,
        )
        env = dict(os.environ)
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("SDL_VIDEODRIVER", "offscreen")
        with (root / "logs" / f"sample-{record['index']:03d}.log").open("w") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "embodichain",
                    "expansion",
                    "run",
                    "--job",
                    str(job),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
            )
            timed_out = False
            try:
                process.wait(timeout=remaining)
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                record["status"] = (
                    "interrupted"
                    if isinstance(error, KeyboardInterrupt)
                    else "budget_exhausted"
                )
                _atomic(root / "preview-plan.json", plan)
                if isinstance(error, KeyboardInterrupt):
                    raise
                timed_out = True
        result = root / record["result"]
        receipts = ArtifactEpisodeSink(result).inspect() if result.is_dir() else ()
        attempts = [
            json.loads(p.read_text())
            for p in (result / "attempts").glob("*/attempt.json")
        ]
        record.update(
            confirmed=sum(r.confirmed for r in receipts),
            attempts=len(attempts),
            returncode=process.returncode,
            outcomes=[
                {
                    "ordinal": a.get("ordinal"),
                    "status": a["status"],
                    "reason": a.get("reason") or a.get("error"),
                }
                for a in attempts
            ],
        )
        exact = 0
        for receipt in receipts:
            episode = load_expansion_episode(receipt.storage_id)
            if _exact_recipe(episode.metadata):
                exact += 1
        record["exact_recipes"] = exact
        record["status"] = (
            "budget_exhausted"
            if timed_out
            else (
                "confirmed"
                if record["confirmed"] == record["planned"]
                and exact == record["planned"]
                else (
                    "partial"
                    if record["confirmed"]
                    else "rejected" if process.returncode in (0, 2) else "error"
                )
            )
        )
        _atomic(root / "preview-plan.json", plan)
    plan["complete"] = all(
        r["attempts"] >= r["planned"]
        and r["status"] in {"confirmed", "partial", "rejected"}
        for r in plan["records"]
    )
    plan["coverage_complete"] = all(r["status"] == "confirmed" for r in plan["records"])
    plan["confirmed"] = sum(r["confirmed"] for r in plan["records"])
    plan["elapsed_s"] = time.monotonic() - started
    _atomic(root / "preview-plan.json", plan)
    return plan
