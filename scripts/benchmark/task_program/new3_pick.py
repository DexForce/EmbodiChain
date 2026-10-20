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

"""Frozen new3 Pick census through the existing native bundle runner.

Run: python -m scripts.benchmark.task_program.new3_pick prepare|run|report
This local campaign needs the archived resource controller. It is not an E1/E3
task benchmark, and it never edits production code or source scene assets.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

__all__: list[str] = []

WORKSPACE = Path(__file__).resolve().parents[3]
ROOT = WORKSPACE / "outputs/gen_sim/new3_pick60_r4_20261018"
SCENES = WORKSPACE / "gym_project/task100/new3"
MODEL = Path(
    "/home/dex/.cache/embodichain_data/assembled/Panda_robotiq_arg2f_140_Panda_robotiq_arg2f_140/Panda_robotiq_arg2f_140_Panda_robotiq_arg2f_140.urdf"
)
# Explicit selections precede native results; no failed item is substituted.
SELECTION = {
    1171: ("paper_cup_001", "plastic_bottle_001", "can_001"),
    1172: ("apple_001", "orange_001", "fruit_plate_001"),
    1173: ("coffee_pot_001", "cup_001", "block_001"),
    1174: ("cup_001", "bottle_001", "service_tray_001"),
    1175: ("jar_001", "bottle_001", "can_001"),
    1176: ("puzzle_cube_001", "ball_001", "wooden_block_001"),
    1177: ("oven_mitt_001", "bowl_001", "water_bottle_001"),
    1178: ("rubik_cube_001", "water_bottle_001", "cup_001"),
    1179: ("kettle_001", "bowl_001", "cup_001"),
    1180: ("spoon_001", "tray_001", "bottle_001"),
    1181: ("cup_001", "wooden_block_001", "ball_001"),
    1182: ("measuring_cup_001", "bowl_001", "puzzle_cube_001"),
    1183: ("spoon_001", "paper_cup_001", "rubber_ball_001"),
    1184: ("cube_001", "bottle_001", "service_tray_001"),
    1185: ("oven_mitt_001", "bowl_001", "block_001"),
    1186: ("pan_001", "apple_001", "orange_001"),
    1187: ("kettle_001", "mug_001", "disc_001"),
    1188: ("dish_rack_001", "cup_001", "block_001"),
    1189: ("ball_001", "bottle_001", "service_tray_001"),
    1190: ("tray_001", "wooden_block_001", "bottle_001"),
}
# These are explicit experiment declarations, not runtime category inference.
UPRIGHT = {
    (1171, "paper_cup_001"),
    (1173, "coffee_pot_001"),
    (1173, "cup_001"),
    (1174, "cup_001"),
    (1178, "cup_001"),
    (1179, "kettle_001"),
    (1179, "cup_001"),
    (1181, "cup_001"),
    (1182, "measuring_cup_001"),
    (1183, "paper_cup_001"),
    (1187, "kettle_001"),
    (1187, "mug_001"),
    (1188, "cup_001"),
}


def read(path: Path) -> Any:
    return json.loads(path.read_text()) if path.is_file() else {}


def write(path: Path, value: Any, *, new: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if new else "w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reference() -> Any:
    path = WORKSPACE / "outputs/gen_sim/e3_core_rebuild_20261017/refactor_campaign.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("new3_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def controller() -> Any:
    api = reference().controller()
    api.WORKER = WORKSPACE / "outputs/gen_sim/e3_broad_6sources_20261017/worker.py"
    return api


def graph_for(identifier: str, uid: str, resource: str, upright: bool) -> dict:
    kind = "E3" if upright else "E1"
    return {
        "schema_version": "semantic_task_graph/v1",
        "task_id": identifier,
        "instruction": f"Diagnostic Pick only: grasp and lift {uid}; no full task credit.",
        "planner_route": "offline",
        "integration_fingerprint": "0" * 64,
        "targets": {},
        "nodes": [
            {
                "id": "pick",
                "call": {
                    "kind": "pick",
                    "object": uid,
                    "resources": {"primary": resource},
                },
                "depends_on": [],
                "task_instance_id": "diagnostic",
                "task_type": kind,
                "role": "primary",
            }
        ],
        "task_groups": [
            {
                "id": "diagnostic",
                "task_type": kind,
                "node_ids": ["pick"],
                "depends_on": [],
                "success": {"kind": "diagnostic_pick_only"},
            }
        ],
        "success": {"kind": "diagnostic_pick_only"},
    }


def prepare() -> None:
    from embodichain.gen_sim.task_engine._task_program.physical_inputs import (
        load_frozen_physical_input,
        verify_physical_input_manifest,
    )
    from embodichain.gen_sim.task_engine.orchestration.source_scene import prepare_scene
    from embodichain.gen_sim.task_engine.task_program_bundle import (
        generate_task_program_bundle,
    )

    ROOT.mkdir(parents=True, exist_ok=False)
    ref = reference()
    head, digest = ref.identity(WORKSPACE)
    pins, rows, inventory = {}, [], []
    for name in ref.source_paths():
        source = WORKSPACE / name
        if source.is_file():
            pins[str(source)] = sha(source)
            target = ROOT / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    for task, uids in SELECTION.items():
        config = SCENES / f"task{task}/scene_export/scene_config.json"
        source = read(config)
        objects = {o["uid"]: o for o in source["rigid_object"]}
        assert len(uids) == len(set(uids)) == 3 and set(uids) <= objects.keys()
        scene = prepare_scene(config)
        for file in config.parent.rglob("*"):
            if file.is_file() and file.suffix.lower() in {
                ".json",
                ".glb",
                ".gltf",
                ".usdc",
                ".usd",
                ".usda",
            }:
                pins[str(file)] = sha(file)
        for slot, uid in enumerate(uids, 1):
            item = objects[uid]
            assert item.get("body_type", "dynamic") == "dynamic"
            identifier = f"t{task}_o{slot}_{uid}_s0"
            upright = (task, uid) in UPRIGHT
            resource = "left" if item["init_pos"][0] >= 0 else "right"
            graph = graph_for(identifier, uid, resource, upright)
            # Producer-owned effort calibration depends on this exact Pick.
            origin = ROOT / "physical_origins" / identifier
            generate_task_program_bundle(
                graph,
                scene,
                origin,
                robot_profile="dual_franka",
                fit_grasp_assets=False,
                verify_pick_phases=True,
                pick_contact_revision=4,
                max_episodes=1,
                max_episode_steps=10000,
                upright_container_picks={"pick": 30.0} if upright else None,
            )
            frozen = load_frozen_physical_input(scene, origin)
            pins.update(frozen.manifest()["input_files"])
            bundle = ROOT / "cases" / identifier / "prepared"
            generate_task_program_bundle(
                graph,
                frozen,
                bundle,
                robot_profile=frozen.robot_profile,
                fit_grasp_assets=False,
                verify_pick_phases=True,
                pick_contact_revision=4,
                upright_container_picks={"pick": 30.0} if upright else None,
            )
            assert verify_physical_input_manifest(bundle) == frozen.manifest()
            asset = (config.parent / item["shape"]["fpath"]).resolve()
            row = {
                "id": identifier,
                "case_id": identifier,
                "E": "E3" if upright else "E1",
                "task": task,
                "slot": slot,
                "source_uid": uid,
                "name": item["name"],
                "category": item["category"],
                "seed": 0,
                "bundle": str(bundle),
                "source_config": str(config),
                "source_asset_sha256": sha(asset),
                "source_position": item["init_pos"],
                "source_scale": item["body_scale"],
                "resource": resource,
                "gripper_max_effort": frozen.embodiment["simulation"][
                    "joint_drive_props"
                ]["max_effort"],
                "upright_quality": upright,
                "physical_digest": frozen.physical_digest,
                "original_task_credit": False,
            }
            rows.append(row)
            for file in bundle.rglob("*"):
                if file.is_file():
                    pins[str(file)] = sha(file)
        inventory.append(
            {"task": task, "selected": list(uids), "rigid_count": len(objects)}
        )
        print("PREPARED", task, flush=True)
    api = controller()
    for path in (
        MODEL,
        Path(__file__),
        Path(api.__file__),
        Path(api.WORKER),
        WORKSPACE / "scripts/benchmark/task_program/gen_sim_e1.py",
        WORKSPACE
        / "outputs/gen_sim/goal_norm_v2_20261015_040916/qualification_worker_control.py",
    ):
        pins[str(path)] = sha(path)
    for module in list(sys.modules.values()):
        file = getattr(module, "__file__", "")
        if file and "/outputs/gen_sim/" in str(file) and str(file).endswith(".py"):
            pins[str(Path(file).resolve())] = sha(Path(file))
    assert ref.identity(WORKSPACE) == (head, digest)
    plan = {
        "schema_version": "gensim.new3-pick60/v1",
        "source_head": head,
        "source_digest": digest,
        "input_manifest": pins,
        "runs": rows,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "native_wall_guard_seconds": 180,
        "original_task_credit": False,
        "schedule": "20 current scenes x 3 distinct rigid UIDs; seed0 once; independent episodes; no substitutions or automatic retries",
        "policy": "existing ordinary Pick for ordinary objects; explicit upright-container E3 Pick diagnostic for declared cup/pot UIDs; not original task semantics",
        "physical_policy": "preserve current source layout, dimensions, mass, friction, gravity and all obstacles; per-object producer-owned robot effort calibration; no physical adaptation or attachment",
        "excluded_development_campaign": "../new3_pick60_r3_20261018/protocol_issue.json",
        "acceptance": {
            "minimum_final_lift_m": 0.08,
            "minimum_positive_contacts": 3,
            "terminal_contact_and_lift_window_s": 0.20,
            "terminal_window_lift_m": 0.08,
            "upright_tilt_limit_degrees": 30,
        },
        "hold_scope": "existing Pick lift/terminal acceptance; last 0.20s retained height/contact, not a separate multi-second stationary hold",
    }
    write(ROOT / "inventory.json", inventory, new=True)
    write(ROOT / "campaign.json", plan, new=True)
    (ROOT / "before.patch").write_bytes(
        subprocess.check_output(
            ["git", "diff", "HEAD", "--binary", "--full-index"], cwd=WORKSPACE
        )
    )
    report()


def physical_verdict(
    *,
    completed: bool,
    acquisition: bool,
    resource_ok: bool,
    final_lift: float | None,
    terminal_lift: float | None,
    contact_count: int,
    terminal_contacts: bool,
    complete_trace: bool,
) -> bool:
    return bool(
        completed
        and acquisition
        and resource_ok
        and complete_trace
        and final_lift is not None
        and final_lift >= 0.08
        and terminal_lift is not None
        and terminal_lift >= 0.08
        and contact_count >= 3
        and terminal_contacts
    )


def native_wall_failure(
    runtime: dict, log: str, *, exit_code: int | None
) -> dict | None:
    """Keep a triggered native deadline invalid even if Python swallowed it."""
    failures = [
        f
        for s in runtime.get("segments", [])
        for f in s.get("metadata", {}).get("runtime", {}).get("failures", [])
    ]
    host = next(
        (f for f in failures if "native_wall_guard" in f.get("message", "")), None
    )
    logged = (
        (exit_code in (124, 137) and "timeout:" in log)
        or "native_wall_guard_180_seconds" in log
        or any(
            marker in log
            for marker in (
                "timeout: sending signal INT",
                "timeout: sending signal KILL",
            )
        )
    )
    if host is None and not logged and exit_code != 124:
        return None
    phase = None if host is None else host.get("phase")
    category = {
        "preparation": "planning_wall_budget",
        "execution": "execution_wall_budget",
    }.get(phase, "native_wall_budget")
    return {
        "kind": "native_wall_budget_exceeded",
        "category": category,
        "message": (
            host["message"]
            if host
            else "Native deadline triggered; inspect native.log and controller receipt."
        ),
        "phase": phase,
        "source": "runtime_failure_or_native_log_or_external_timeout_receipt",
    }


def evaluation_digest() -> str:
    """Bind cached verdicts to this collector and its acceptance helpers."""
    return sha(Path(__file__))


def collect(row: dict, plan: dict) -> dict:
    import cv2
    import torch

    output = ROOT / "runs" / row["id"]
    receipt = read(output / "controller_receipt.json")
    if not receipt:
        return {
            **row,
            "status": (
                "running" if (output / "admission.json").exists() else "not_started"
            ),
            "accepted": False,
            "videos": [],
        }
    cached = output / "pick_result.json"
    evaluator = evaluation_digest()
    if cached.exists():
        result = read(cached)
        assert all(sha(Path(p)) == digest for p, digest in result["raw_hashes"].items())
        if result.get("evaluation_digest") == evaluator:
            return result
        # Preserve the historical verdict; new rules derive a separate result
        # from the same immutable raw evidence rather than rewriting history.
        cached = output / "evaluations" / evaluator / "pick_result.json"
        if cached.exists():
            result = read(cached)
            assert result["evaluation_digest"] == evaluator
            assert all(
                sha(Path(p)) == digest for p, digest in result["raw_hashes"].items()
            )
            return result
    execution = read(output / "execution_report.json")
    runtime = execution.get("runtime_result") or {}
    native_log = output / "native.log"
    wall_failure = native_wall_failure(
        runtime,
        native_log.read_text(errors="replace") if native_log.exists() else "",
        exit_code=receipt.get("exit_code"),
    )
    calls = [
        c
        for s in runtime.get("segments", [])
        for c in s.get("metadata", {}).get("runtime", {}).get("calls", [])
    ]
    call = calls[0] if calls else {}
    events = call.get("events", [])
    resource = read(output / "native.log.resources.receipt.json")
    grip = read(output / "grip_contact_evidence.json")
    trace = [
        q
        for q in grip.get("trace", [])
        if q["object_id"] == row["source_uid"] and q.get("expected") is True
    ]
    good = [
        q
        for q in trace
        if any(
            r.get("valid") is True
            and r.get("value") is True
            and r.get("finite") is True
            and sum(r.get("impulses", [])) > 0
            for r in q.get("rows", [])
        )
    ]
    final_lift = terminal_lift = tilt = None
    dt = None
    trajectories = list((output / "trajectory").glob("*.pt"))
    if len(trajectories) == 1:
        data = torch.load(trajectories[0], weights_only=False, map_location="cpu")
        dt = float(data["meta"]["step_dt"])
        poses = data["states"]["rigid_objects", row["source_uid"], "pose"][0]
        z = poses[:, 2] - poses[0, 2]
        count = int(round(0.20 / dt)) + 1
        final_lift = float(z[-1])
        terminal_lift = float(z[-count:].min()) if len(z) >= count else None
        if good:
            import contextlib
            import io
            import pytorch_kinematics as pk
            from embodichain.utils.math import xyz_quat_to_4x4_matrix

            indices = sorted(
                {max(0, min(len(z) - 1, round(q["timestamp"] / dt) - 1)) for q in good}
            )
            with (
                contextlib.redirect_stderr(io.StringIO()),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                chain = pk.build_chain_from_urdf(MODEL.read_text())
            qpos = data["states"]["robot", "qpos"][0, indices]
            fk = chain.forward_kinematics(
                {n: qpos[:, i] for i, n in enumerate(data["meta"]["joint_names"])}
            )
            pads = [
                name
                for name in fk
                if name.startswith(row["resource"] + "_")
                and name.endswith("finger_pad")
            ]
            assert len(pads) == 2
            root = xyz_quat_to_4x4_matrix(
                data["states"]["robot", "root_pose"][0, indices]
            )
            a, b = [(root @ fk[name].get_matrix())[:, :3, 3] for name in pads]
            axis = b - a
            axis /= torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
            tilt = float(torch.rad2deg(torch.asin(axis[:, 2].abs().clamp(max=1))).max())
    end = max((e.get("timestamp", 0) for e in events), default=0)
    terminal = [q for q in trace if q["timestamp"] >= end - 0.20 - 1e-6]
    terminal_good = [q for q in good if q["timestamp"] >= end - 0.20 - 1e-6]
    terminal_contacts = bool(
        dt
        and terminal
        and len(terminal) == len(terminal_good)
        and terminal[0]["timestamp"] <= end - 0.20 + (dt or 0) + 1e-6
        and terminal[-1]["timestamp"] >= end - (dt or 0) - 1e-6
    )
    acquisition = any(
        e["kind"] == "phase_effect_gate_satisfied"
        and "destination_acquired" in e.get("message", "")
        for e in events
    )
    resource_ok = (
        receipt.get("status") == "returned"
        and receipt.get("exit_code") == 0
        and resource.get("status") == "finished"
        and not (output / "drift_receipt.json").exists()
        and wall_failure is None
    )
    physical = physical_verdict(
        completed=call.get("status") == "completed",
        acquisition=acquisition,
        resource_ok=resource_ok,
        final_lift=final_lift,
        terminal_lift=terminal_lift,
        contact_count=len(good),
        terminal_contacts=terminal_contacts,
        complete_trace=grip.get("trace_complete") is True,
    )
    emitted = [
        a.get("planner_diagnostics", {})
        .get("metadata", {})
        .get("gen_sim_pick_contact_geometry", {})
        .get("grasp_quality")
        for a in call.get("plan_attempts", [])
    ]
    quality = not row["upright_quality"] or bool(
        tilt is not None
        and tilt <= 30.0001
        and any(q and q.get("valid") is True for q in emitted)
    )
    failures = [
        e
        for s in runtime.get("segments", [])
        for e in s.get("metadata", {}).get("runtime", {}).get("events", [])
        if e.get("kind")
        in {
            "action_planning_failed",
            "phase_effect_gate_failed",
            "held_object_lost",
            "terminal_acceptance_failed",
            "tracking_feedback_failed",
            "effect_verification_timeout",
            "action_timeout",
        }
    ]
    failure = failures[0] if failures else {}
    if physical and quality:
        category = "passed"
    elif wall_failure is not None:
        category = wall_failure["category"]
    elif (
        receipt.get("status") != "returned"
        or resource.get("status") != "finished"
        or not call
    ):
        category = "resource_or_runtime"
    elif failure.get("kind") == "action_planning_failed":
        category = "planning"
    elif not acquisition:
        category = "acquisition"
    elif not physical:
        category = "lift_or_retention"
    else:
        category = "grasp_quality"
    videos = []
    cv2.setNumThreads(1)
    for i, video in enumerate(sorted(output.rglob("*.mp4"))):
        cap = cv2.VideoCapture(str(video))
        frames, last = 0, None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames, last = frames + 1, frame
        cap.release()
        poster = ROOT / "posters" / f"{row['id']}_{i}.jpg"
        poster.parent.mkdir(exist_ok=True)
        if last is not None:
            cv2.imwrite(str(poster), last)
        videos.append(
            {
                "path": str(video),
                "frames": frames,
                "poster": str(poster) if last is not None else None,
            }
        )
    raw = [
        p
        for p in output.rglob("*")
        if p.is_file()
        and p.suffix in {".json", ".jsonl", ".log", ".pt", ".mp4"}
        and p.name != "pick_result.json"
        and "evaluations" not in p.relative_to(output).parts
    ]
    result = {
        **row,
        "evaluation_digest": evaluator,
        "status": "finished",
        "accepted": physical and quality,
        "physical_pass": physical,
        "quality_pass": quality,
        "failure_category": category,
        "first_failure": failure or wall_failure or receipt.get("exception"),
        "wall_budget_failure": wall_failure,
        "acquisition": acquisition,
        "contacts": len(good),
        "terminal_contacts": terminal_contacts,
        "final_lift_m": final_lift,
        "terminal_min_lift_m": terminal_lift,
        "actual_pad_frame_tilt_degrees": tilt,
        "emitted_quality": emitted,
        "wall_seconds": receipt.get("wall_seconds"),
        "resource_status": resource.get("status"),
        "videos": videos,
        "raw_hashes": {str(p): sha(p) for p in raw},
    }
    write(cached, result, new=True)
    return result


def report() -> None:
    import torch

    torch.set_num_threads(1)
    plan = read(ROOT / "campaign.json")
    rows = [collect(row, plan) for row in plan["runs"]]
    completed = [r for r in rows if r["status"] == "finished"]
    summary = {
        "scope": "new3 independent Pick only, not original E1/E3 tasks",
        "source_head": plan["source_head"],
        "source_digest": plan["source_digest"],
        "declared": len(rows),
        "finished": len(completed),
        "passed": sum(r["accepted"] for r in rows),
        "unique_source_file_hashes": len({r["source_asset_sha256"] for r in rows}),
        "by_failure": dict(Counter(r["failure_category"] for r in completed)),
        "rows": rows,
    }
    write(ROOT / "pick_summary.json", summary)
    labels = {
        "passed": "通过",
        "planning": "规划失败",
        "acquisition": "未形成有效夹持",
        "lift_or_retention": "抬升/保持未通过",
        "grasp_quality": "抓型不合格",
        "resource_or_runtime": "资源/运行异常",
        "planning_wall_budget": "规划墙钟超时",
        "execution_wall_budget": "执行墙钟超时",
        "native_wall_budget": "原生进程墙钟超时",
    }
    table, cards = [], []
    for row in rows:
        category = row.get("failure_category", row["status"])
        label = labels.get(
            category,
            {"not_started": "待测试", "running": "运行中"}.get(category, category),
        )
        css = (
            "pass"
            if row["accepted"]
            else "pending" if row["status"] != "finished" else "fail"
        )
        lift = (
            "待测"
            if row.get("final_lift_m") is None
            else f"{row['final_lift_m']*100:.1f}cm"
        )
        policy = "直立杯壶30°" if row["upright_quality"] else "普通Pick"
        table.append(
            f'<tr><td>{row["task"]}</td><td><a href="#{row["id"]}">{html.escape(row["source_uid"])}</a></td><td>{policy}</td><td>{row["resource"]}</td><td class="{css}">{label}</td><td>{lift}</td><td>{row.get("contacts", "待测")}</td></tr>'
        )
        media = []
        for v in row["videos"]:
            path = Path(v["path"]).relative_to(ROOT).as_posix()
            poster = (
                Path(v["poster"]).relative_to(ROOT).as_posix() if v["poster"] else ""
            )
            media.append(
                f'<video controls preload="none" poster="{poster}" src="{path}"></video>'
            )
        reason = row.get("first_failure") or ""
        if isinstance(reason, dict):
            reason = reason.get("kind", "") + ": " + reason.get("message", "")
        if not media:
            media.append(f'<div class="empty">{label}；无执行视频</div>')
        if row["status"] == "finished":
            evidence = f'<a href="runs/{row["id"]}/pick_result.json">实测与哈希</a> · <a href="runs/{row["id"]}/execution_report.json">原始回执</a>'
        else:
            evidence = ""
        cards.append(
            f'<article id="{row["id"]}"><h3>{row["task"]} · {html.escape(row["source_uid"])}</h3><p>{html.escape(row["name"])} · {policy} · {row["resource"]}</p><p class="{css}">{label} · 末帧抬升{lift}</p>{"".join(media)}<p>{html.escape(str(reason))}</p><p>{evidence}</p></article>'
        )
    updated = datetime.now().astimezone().isoformat(timespec="seconds")
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>new3 · 60次独立抓取</title><style>*{{box-sizing:border-box}}body{{margin:0;background:#fafbfc;color:#20282d;font:14px/1.65 system-ui,sans-serif;letter-spacing:0}}main{{max-width:1220px;margin:auto;padding:24px}}h1{{font-size:25px}}h2{{font-size:20px}}h3{{font-size:16px;overflow-wrap:anywhere}}a{{color:#246675}}nav{{display:flex;gap:18px;flex-wrap:wrap}}section{{border-top:1px solid #dce2e5;margin-top:24px;padding-top:18px}}.pass{{color:#176c46}}.fail{{color:#ae3441}}.pending{{color:#586770}}.scroll{{overflow:auto}}table{{width:100%;min-width:780px;border-collapse:collapse}}td,th{{text-align:left;padding:8px 6px;border-bottom:1px solid #dce2e5}}.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}}article{{border:1px solid #dce2e5;background:white;border-radius:4px;padding:12px;min-width:0}}article p{{overflow-wrap:anywhere}}video,.empty{{display:block;width:100%;aspect-ratio:16/9;background:#20272a;object-fit:contain}}.empty{{display:flex;align-items:center;justify-content:center;color:#edf0f2;text-align:center;padding:20px}}@media(max-width:900px){{.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}@media(max-width:600px){{main{{padding:16px 12px}}.grid{{grid-template-columns:1fr}}h1{{font-size:22px}}}}</style><main><h1>new3 · 60次独立抓取</h1><p><strong>已结束 {summary["finished"]}/60 · 通过 {summary["passed"]}/{summary["finished"] or 0}</strong> · 20个场景 · {summary["unique_source_file_hashes"]}份去重源GLB · {updated}</p><nav><a href="campaign.json">冻结协议</a><a href="inventory.json">事前选物</a><a href="pick_summary.json">逐例实测</a><a href="../e3_closure_quality_20261018/index.html">此前30°对照</a></nav><p>保留new3当前布局、尺寸、质量、摩擦和全部障碍；不挪物、不缩放、不吸附。每个物体独立恢复场景，seed0一次，无自动补考。原生双指接触、8cm抬升与末0.20秒接触/高度保持分别检查；不是额外多秒静止保持。明确声明的直立杯壶增加30°抓型检查；普通Pick不改变默认。</p><p>只验收Pick，不计原始E1/E3任务成绩；托盘/锅的载物稳定性不在本轮验收内。去重计数按源文件字节，不等于独立几何类别数。失败不以更换物体或缺失视频消失。</p><section><h2>逐例进展</h2><div class="scroll"><table><tr><th>场景</th><th>对象UID</th><th>策略</th><th>手臂</th><th>结论</th><th>实测抬升</th><th>双侧接触</th></tr>{"".join(table)}</table></div></section><section><h2>视频与原始证据</h2><div class="grid">{"".join(cards)}</div></section></main></html>"""
    (ROOT / "index.html").write_text(page)
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False
        ),
        flush=True,
    )


def run(limit: int | None = None) -> None:
    path = ROOT / "campaign.json"
    plan = read(path)
    api = controller()
    count = 0
    for row in plan["runs"]:
        if (ROOT / "runs" / row["id"] / "admission.json").exists():
            continue
        if limit is not None and count >= limit:
            break
        print("NATIVE_START", row["id"], flush=True)
        state = api.execute(plan, path, [row], workspace=WORKSPACE, max_runs=1)
        report()
        count += 1
        if state.startswith("paused_"):
            raise RuntimeError(state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "report"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and (args.phase != "run" or args.limit < 1):
        parser.error("--limit requires run and a positive integer")
    if args.phase == "prepare":
        prepare()
    elif args.phase == "run":
        run(args.limit)
    else:
        report()


if __name__ == "__main__":
    main()
