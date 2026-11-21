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
"""Reproduce the five local, frozen DROID demonstrations from the Feishu document."""

from __future__ import annotations

import argparse
import ast
import copy
import datetime as dt
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import sys
import time
import uuid
from urllib.parse import quote
import xml.etree.ElementTree as ET

ENTRY = Path(__file__).resolve()
HERE = (
    Path(os.environ.get("DROID_REPLAY_RESOURCES", ENTRY.parent)).expanduser().resolve()
)
LOCK = HERE / "tasks.json"
LEGACY_ROOT = "/media/dex/extra/data/droid/real2sim_80"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rewrite_root(source: str, root: Path) -> str:
    """Change only the one ROOT/R path literal, preserving the remaining source."""
    tree = ast.parse(source)
    matches = []
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(x, ast.Name) and x.id in ("ROOT", "R") for x in node.targets
            )
            and isinstance(node.value, ast.Call)
            and node.value.args
            and isinstance(node.value.args[0], ast.Constant)
            and node.value.args[0].value == LEGACY_ROOT
        ):
            matches.append(node.value.args[0])
    if len(matches) != 1:
        raise ValueError("Expected exactly one legacy root assignment")
    node = matches[0]
    lines = source.splitlines(keepends=True)
    start = sum(len(x) for x in lines[: node.lineno - 1]) + node.col_offset
    stop = sum(len(x) for x in lines[: node.end_lineno - 1]) + node.end_col_offset
    derived = source[:start] + repr(str(root)) + source[stop:]
    normalized = ast.parse(derived)
    for item in normalized.body:
        if (
            isinstance(item, ast.Assign)
            and any(
                isinstance(x, ast.Name) and x.id in ("ROOT", "R") for x in item.targets
            )
            and isinstance(item.value, ast.Call)
            and item.value.args
            and isinstance(item.value.args[0], ast.Constant)
            and item.value.args[0].value == str(root)
        ):
            item.value.args[0].value = LEGACY_ROOT
    if ast.dump(tree) != ast.dump(normalized):
        raise ValueError("Derived runner changes more than the input root")
    return derived


def references(path: Path) -> list[str]:
    if path.suffix.lower() == ".urdf":
        return [
            n.attrib["filename"]
            for n in ET.parse(path).getroot().iter()
            if n.tag in ("mesh", "texture") and "filename" in n.attrib
        ]
    if path.suffix.lower() == ".usda":
        import re

        return re.findall(r"@([^@]+)@", path.read_text())
    if path.suffix.lower() == ".glb":
        data = path.read_bytes()
        if data[:4] != b"glTF":
            raise ValueError(f"Invalid GLB: {path}")
        length, kind = struct.unpack_from("<II", data, 12)
        if kind != 0x4E4F534A:
            raise ValueError(f"Missing GLB JSON: {path}")
        document = json.loads(data[20 : 20 + length])
        return [
            n["uri"]
            for group in ("buffers", "images")
            for n in document.get(group, [])
            if "uri" in n and not n["uri"].startswith("data:")
        ]
    return []


def child_env(config: dict, task: dict) -> dict:
    env = os.environ.copy()
    repo = Path(config["repository"])
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo / "extensions"), str(repo), env.get("PYTHONPATH", "")]
    )
    env["DROID_EPISODE_ID"] = task["episode_id"]
    env["DROID_ATTEMPT"] = task["attempt"]
    return env


def check_files(files: dict[str, str]) -> None:
    for name, expected in files.items():
        path = Path(name)
        if not path.is_file():
            raise FileNotFoundError(f"Missing locked resource: {path}")
        if digest(path) != expected:
            raise ValueError(f"Locked resource changed: {path}")


def preflight(config: dict, tasks: list[dict], output: Path) -> dict:
    root = Path(config["source_root"]).resolve()
    output = output.resolve()
    if output == root or output.is_relative_to(root / "1.0.1"):
        raise ValueError("Output must not be a historical episode or source root")
    ancestor = output
    while not ancestor.exists():
        ancestor = ancestor.parent
    if not os.access(ancestor, os.W_OK):
        raise PermissionError(f"Output parent not writable: {ancestor}")
    for name in ("ffmpeg", "ffprobe", "nvidia-smi"):
        if not shutil.which(name):
            raise RuntimeError(f"Missing executable: {name}")
    subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    head = subprocess.check_output(
        ["git", "-C", config["repository"], "rev-parse", "HEAD"], text=True
    ).strip()
    if head != config["repository_commit"]:
        raise ValueError(
            f'Expected repository {config["repository_commit"]}, found {head}'
        )
    check_files(config["robot_resources"])
    check_files(config["repository_files"])
    for task in tasks:
        check_files(task["source_files"])
        check_files({str(HERE / task["validator"]): task["validator_sha256"]})
        for path in (Path(task["frozen_inputs"]) / "assets").rglob("*"):
            if path.is_file():
                for ref in references(path):
                    if "://" in ref:
                        raise ValueError(f"Nonlocal asset reference: {path}: {ref}")
                    target = Path(ref) if Path(ref).is_absolute() else path.parent / ref
                    if not target.is_file():
                        raise FileNotFoundError(f"Unresolved reference: {path}: {ref}")
    code = (
        "import json,sys,importlib.metadata as m,dexsim,torch,numpy,PIL,cv2; "
        "from embodichain.lab.sim import SimulationManager; "
        "from embodichain.lab.sim.cfg import DefaultPhysicsCfg,RenderCfg; "
        'print(json.dumps({"python":sys.version.split()[0],"dexsim":dexsim.__version__, '
        '"torch":torch.__version__,"numpy":numpy.__version__,"scipy":m.version("scipy"),'
        '"trimesh":m.version("trimesh")}))'
    )
    result = subprocess.run(
        [config["python"], "-c", code],
        env=child_env(config, tasks[0]),
        capture_output=True,
        text=True,
        timeout=90,
        check=True,
    )
    versions = json.loads(result.stdout.strip().splitlines()[-1])
    for key, expected in config["versions"].items():
        if versions[key] != expected:
            raise ValueError(f"Expected {key}={expected}, found {versions[key]}")
    return {
        "passed": True,
        "versions": versions,
        "repository_commit": head,
        "episodes": [t["episode_id"] for t in tasks],
        "output": str(output),
    }


def run_child(command: list[str], env: dict, log: Path, timeout: int = 900) -> None:
    """Own and reap the worker's process group, including its FFmpeg children."""
    with log.open("w") as stream:
        process = subprocess.Popen(
            command,
            env=env,
            cwd=HERE,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
            if code:
                raise RuntimeError(f"Child exited {code}; see {log}")
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            if process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            # An exited worker may leave a child in the same owned process group.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def stage_task(task: dict, dest: Path) -> tuple[Path, Path, dict]:
    root = dest / "w"
    episode = root / "p"
    reconstruction = episode / "reconstruction"
    reconstruction.mkdir(parents=True)
    frozen = Path(task["frozen_inputs"])
    for source in frozen.iterdir():
        if source.is_file():
            shutil.copy2(source, reconstruction / source.name)
    for source in (frozen / "reconstruction_evidence").glob("*"):
        if source.is_file():
            shutil.copy2(source, reconstruction / source.name)
    for name in ("assets",):
        (episode / name).symlink_to(frozen / name, target_is_directory=True)
    (episode / "real").symlink_to(
        Path(task["episode_folder"]) / "real", target_is_directory=True
    )
    shutil.copy2(
        frozen / "recorded_trajectory.npz", episode / "recorded_trajectory.npz"
    )
    write_json(
        root / "manifest.json",
        {"episodes": [{"id": task["episode_id"], "folder": str(episode)}]},
    )
    original = (frozen / "replay_source.py").read_text()
    derived = rewrite_root(original, root)
    runner = dest / "derived_replay.py"
    runner.write_text(derived)
    run = episode / "simulation" / task["attempt"]
    generated = (
        run / "inputs/assembled" / ("articulated_" + task["episode_id"] + "_robot")
    )
    if (
        len(
            str(
                generated / ("articulated_" + task["episode_id"] + "_robot.urdf")
            ).encode()
        )
        >= 240
    ):
        raise ValueError(
            "Output path is too long for the legacy native runtime; use a shorter --output"
        )
    evidence = {
        "task_number": task["number"],
        "episode_id": task["episode_id"],
        "baseline_attempt": task["attempt"],
        "frozen_inputs": str(frozen),
        "baseline_validation": str(Path(task["episode_folder"]) / "validation.json"),
        "source_files": task["source_files"],
        "original_runner_sha256": digest(frozen / "replay_source.py"),
        "derived_runner_sha256": digest(runner),
        "root_only_AST_equivalence": True,
        "staged_root": str(root),
        "fresh_run": str(run),
        "runner_args": task["runner_args"],
        "control_semantics": "Legacy observation-reference proxy and declared preload, not measured finger angles or exact original actuator commands.",
    }
    write_json(dest / "inputs_manifest.json", evidence)
    (dest / "source_assets").symlink_to(frozen / "assets", target_is_directory=True)
    return runner, run, evidence


def compare_physics(new: dict, old: dict) -> dict:
    import numpy as np

    def numeric(value: object) -> object:
        array = np.asarray(value, dtype=float)
        if not np.isfinite(array).all():
            raise ValueError("Nonfinite physics state")
        return array

    maxima = {
        "robot_joint_rad": 0.0,
        "object_position_m": 0.0,
        "object_rotation_rad": 0.0,
        "articulation_joint_native_units": 0.0,
    }
    for group in ("states", "terminal_open_gripper_stability"):
        if len(new[group]) != len(old[group]):
            raise ValueError(f"Physics sample count differs: {group}")
        for actual, baseline in zip(new[group], old[group]):
            if "robot_qpos" in baseline:
                measured = numeric(actual["robot_qpos"])
                expected = numeric(baseline["robot_qpos"])
                if measured.shape != expected.shape:
                    raise ValueError("Robot joint count differs")
                maxima["robot_joint_rad"] = max(
                    maxima["robot_joint_rad"],
                    float(np.max(np.abs(measured - expected))),
                )
            for key in ("objects", "articulation_roots"):
                if set(actual.get(key, {})) != set(baseline.get(key, {})):
                    raise ValueError(f"Entity set differs: {key}")
                for uid, p in baseline.get(key, {}).items():
                    p, q = numeric(p), numeric(actual[key][uid])
                    if p.shape != (7,) or q.shape != (7,):
                        raise ValueError(f"Invalid rigid pose: {uid}")
                    maxima["object_position_m"] = max(
                        maxima["object_position_m"],
                        float(np.linalg.norm(np.array(p[:3]) - q[:3])),
                    )
                    first, second = np.asarray(p[3:]), np.asarray(q[3:])
                    if min(np.linalg.norm(first), np.linalg.norm(second)) <= 0:
                        raise ValueError(f"Invalid quaternion: {uid}")
                    dot = abs(float(first @ second)) / float(
                        np.linalg.norm(first) * np.linalg.norm(second)
                    )
                    maxima["object_rotation_rad"] = max(
                        maxima["object_rotation_rad"],
                        float(2 * np.arccos(np.clip(dot, 0, 1))),
                    )
            if set(actual.get("articulations", {})) != set(
                baseline.get("articulations", {})
            ):
                raise ValueError("Articulation set differs")
            for uid, values in baseline.get("articulations", {}).items():
                expected = numeric(values)
                measured = numeric(actual["articulations"][uid])
                if measured.shape != expected.shape:
                    raise ValueError(f"Articulation joint count differs: {uid}")
                if expected.size:
                    maxima["articulation_joint_native_units"] = max(
                        maxima["articulation_joint_native_units"],
                        float(np.max(np.abs(measured - expected))),
                    )
    tolerances = {
        "robot_joint_rad": 1e-5,
        "object_position_m": 1e-4,
        "object_rotation_rad": 1e-3,
        "articulation_joint_native_units": 1e-4,
    }
    return {
        "passed": all(np.isfinite(v) and v <= tolerances[k] for k, v in maxima.items()),
        "max_difference": maxima,
        "tolerances": tolerances,
        "scope": "Every recorded robot/object/articulation pose and post-video stability sample; no assertion about unrecorded fields or identical encoded bytes.",
    }


def validator_program(task: dict, root: Path) -> str:
    source = (HERE / task["validator"]).read_text()
    if task["episode_id"] in ("002", "003", "016"):
        tree = ast.parse(source)
        for node in tree.body:
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "write_text"
                and "validation.json" in ast.unparse(node.value.func.value)
            ):
                source = "\n".join(source.splitlines()[: node.end_lineno]) + "\n"
                # A semicolon line can include packaging after the validation write.
                keep = ast.parse(source)
                keep.body = keep.body[
                    : keep.body.index(
                        next(
                            n
                            for n in keep.body
                            if isinstance(n, ast.Expr)
                            and isinstance(n.value, ast.Call)
                            and isinstance(n.value.func, ast.Attribute)
                            and n.value.func.attr == "write_text"
                            and "validation.json" in ast.unparse(n.value.func.value)
                        )
                    )
                    + 1
                ]
                source = ast.unparse(keep) + "\n"
                break
        else:
            raise ValueError("Missing validator output boundary")
    elif task["episode_id"] == "019":
        source = source[: source.index("from publish_result import publish_result")]
        source += "\nvalidation['task_type']='block_stacking'\n(RUN/'validation.json').write_text(json.dumps(validation,indent=2))\n"
    return rewrite_root(source, root)


def validate_worker(task_path: Path, run: Path, root: Path) -> None:
    import numpy as np

    task = read_json(task_path)
    # The immutable source centroids are retained; cached historical projections
    # are recomputed from THIS rollout before the original Episode003 checks.
    if task["episode_id"] == "003":
        ref_path = root / "p/reconstruction/brick_motion_reference.json"
        rows = json.loads(ref_path.read_text())
        scene = read_json(run / "inputs/scene_measured.json")
        result = read_json(run / "replay_result.json")
        for row in rows:
            cam = scene["cameras"][row["camera"]]
            k, e = np.array(cam["intrinsic"]), np.array(cam["world_to_camera"])
            pose = np.array(result["states"][row["frame"]]["objects"]["blue_brick"][:3])
            projected = k @ (e[:3, :3] @ pose + e[:3, 3])
            uv = projected[:2] / projected[2]
            row["sim_projected_center_px"] = uv.tolist()
            row["centroid_error_px"] = float(
                np.linalg.norm(uv - row["source_centroid_px"])
            )
        write_json(ref_path, rows)
    source = validator_program(task, root)
    derived = run.parent.parent.parent.parent / "derived_validation.py"
    derived.write_text(source)
    old_argv = sys.argv
    sys.argv = (
        [str(derived), "--attempt", task["attempt"]]
        if task["episode_id"] == "010"
        else [str(derived)]
    )
    try:
        exec(
            compile(source, str(derived), "exec"),
            {"__name__": "__main__", "__file__": str(derived)},
        )
    finally:
        sys.argv = old_argv
    fresh = read_json(run / "validation.json")
    new = read_json(run / "replay_result.json")
    baseline = read_json(Path(task["frozen_inputs"]).parent / "replay_result.json")
    if new["frames"] != task["frames"]:
        raise ValueError("Not a complete trajectory")
    parity = compare_physics(new, baseline)
    media = {}
    for name in [*(f"{c}_sim.mp4" for c in task["cameras"]), "real_vs_sim.mp4"]:
        path = run / name
        stream = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-show_entries",
                    "stream=width,height,nb_read_frames,avg_frame_rate",
                    "-of",
                    "json",
                    str(path),
                ]
            )
        )["streams"][0]
        decoded = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-threads",
                "1",
                "-i",
                str(path),
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
        )
        if (
            int(stream["nb_read_frames"]) != task["frames"]
            or decoded.returncode
            or decoded.stderr.strip()
        ):
            raise ValueError(f"Incomplete or invalid media: {path}: {decoded.stderr}")
        media[name] = dict(stream, sha256=digest(path), full_decode_passed=True)
    fresh["attempt"] = task["attempt"]
    fresh["original_actuator_action_replay_verified"] = False
    fresh["control_semantics_clarification"] = (
        "Replay retains the old observation-reference mapping and explicit preload. The legacy joint_angle_rad argument names a model reference, not directly measured finger joint angles. No firmware/force identity claim."
    )
    if "calibration_adjustments" in fresh:
        fresh["calibration_adjustments"]["note"] = fresh[
            "control_semantics_clarification"
        ]
    fresh["replay_consistency"] = parity
    fresh["media_verification"] = media
    fresh["historical_validation"] = str(
        Path(task["episode_folder"]) / "validation.json"
    )
    fresh["validation_recomputed_on"] = str(run)
    fresh["validator_source_sha256"] = task["validator_sha256"]
    fresh["derived_validator_sha256"] = digest(derived)
    fresh["reproduced"] = bool(fresh["physical_task_pass"] and parity["passed"])
    write_json(run / "validation.json", fresh)
    print(
        json.dumps(
            {"reproduced": fresh["reproduced"], "parity": parity}, ensure_ascii=False
        ),
        flush=True,
    )


def link(source: Path, destination: Path) -> None:
    destination.symlink_to(source.resolve())


def expose_results(task: dict, dest: Path, run: Path) -> None:
    for name in ("replay_result.json", "validation.json"):
        if (run / name).exists():
            link(run / name, dest / name)
    for name in ("sim", "comparison", "preview"):
        (dest / name).mkdir(exist_ok=True)
    for cam in task["cameras"]:
        if (run / f"{cam}_sim.mp4").exists():
            link(run / f"{cam}_sim.mp4", dest / "sim" / f"{cam}_sim.mp4")
        for source in run.glob(f"{cam}_*.png"):
            link(source, dest / "preview" / source.name)
        source = Path(task["episode_folder"]) / "reconstruction" / f"{cam}_rgb0.png"
        if source.exists():
            link(source, dest / "preview" / f"{cam}_real_first.png")
    if (run / "real_vs_sim.mp4").exists():
        link(run / "real_vs_sim.mp4", dest / "comparison/real_vs_sim.mp4")


def report(batch: Path, rows: list[dict]) -> None:
    cards = []
    for row in rows:
        task, directory = row["task"], Path(row["directory"])
        rel = directory.relative_to(batch).as_posix()
        links = []
        for title, file in [
            ("原仿对比", "comparison/real_vs_sim.mp4"),
            ("验证", "validation.json"),
            ("运行日志", "run.log"),
            ("Blender场景", "source_assets/foreground_scene.blend"),
        ]:
            if (directory / file).exists():
                links.append(f'<a href="{quote(rel + "/" + file)}">{title}</a>')
        images = "".join(
            f'<img width="320" src="{quote(rel + "/preview/" + c + "_real_first.png")}">'
            for c in task["cameras"]
        )
        cards.append(
            f'<article><h2>{task["number"]}. {html.escape(task["title"])}</h2><p>Episode {task["episode_id"]} · {html.escape(row["status"])}</p>{images}<p>{" · ".join(links)}</p><p>{html.escape(row.get("error", ""))}</p></article>'
        )
    page = (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>文档前5任务复现</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:30px auto;background:#f4f6f9}article{background:white;padding:20px;margin:20px 0}a{color:#145da1}img{max-width:48%}</style><h1>文档前5任务：本机物理复现</h1><p>原视频30FPS展示，物理使用源时间戳；控制和外观为历史已披露近似。新结果独立保存，不修改原40条验收。</p>'
        + "".join(cards)
        + "</html>"
    )
    (batch / "index.html").write_text(page)
    write_json(
        batch / "summary.json",
        {
            "tasks": rows,
            "all_reproduced": len(rows) > 0
            and all(x["status"] == "reproduced" for x in rows),
        },
    )


def archive_tool(batch: Path) -> Path:
    """Preserve the exact launcher, locked list and validators used by a batch."""
    target = batch / "tool"
    target.mkdir()
    shutil.copy2(ENTRY, target / "replay_document_tasks.py")
    shutil.copy2(ENTRY.with_name("README.md"), target / "README.md")
    for name in ("tasks.json", "test_replay_document_tasks.py"):
        if (HERE / name).is_file():
            shutil.copy2(HERE / name, target / name)
    shutil.copytree(HERE / "validators", target / "validators")
    write_json(
        batch / "tool_sha256.json",
        {
            str(p.relative_to(target)): digest(p)
            for p in target.rglob("*")
            if p.is_file()
        },
    )
    return target / "replay_document_tasks.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--list",
        action="store_true",
        help="List document order, source episodes and assets",
    )
    modes.add_argument(
        "--check",
        action="store_true",
        help="Check all locked resources and the local environment",
    )
    modes.add_argument("--task", choices=["1", "2", "3", "4", "5", "all"])
    parser.add_argument("--output", type=Path, default=HERE / "runs")
    parser.add_argument(
        "--_validate",
        nargs=3,
        metavar=("TASK_JSON", "RUN", "ROOT"),
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args._validate:
        validate_worker(*(Path(x) for x in args._validate))
        return 0
    config = read_json(LOCK)
    tasks = config["tasks"]
    if args.list:
        for task in tasks:
            print(
                f'{task["number"]}. {task["title"]} | Episode {task["episode_id"]} | {task["attempt"]} | {task["frames"]} frames\n   assets: {task["frozen_inputs"]}/assets\n   Blender: {task["frozen_inputs"]}/assets/foreground_scene.blend'
            )
        return 0
    selected = tasks if args.task in (None, "all") else [tasks[int(args.task) - 1]]
    checks = preflight(config, selected, args.output)
    print(json.dumps(checks, ensure_ascii=False), flush=True)
    if args.check:
        return 0
    import fcntl

    with (HERE / ".replay.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        batch = args.output.resolve() / (
            dt.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4]
        )
        batch.mkdir(parents=True, exist_ok=False)
        validation_entry = archive_tool(batch)
        write_json(batch / "preflight.json", checks)
        write_json(batch / "source_document.json", config["source_document"])
        source_manifest_hash = digest(Path(config["source_root"]) / "manifest.json")
        rows = []
        for task in selected:
            dest = batch / f'{task["number"]:02d}'
            dest.mkdir()
            row = {
                "task": {k: v for k, v in task.items() if k != "source_files"},
                "directory": str(dest),
                "status": "running",
            }
            rows.append(row)
            report(batch, rows)
            run = None
            print(
                f'START task {task["number"]}: {task["title"]}\nLOG {dest / "run.log"}',
                flush=True,
            )
            try:
                runner, run, evidence = stage_task(task, dest)
                task_file = dest / "task.json"
                write_json(task_file, task)
                command = [
                    config["python"],
                    str(runner),
                    "--attempt",
                    task["attempt"],
                    *task["runner_args"],
                ]
                write_json(
                    dest / "command.json",
                    {
                        "argv": command,
                        "cwd": str(HERE),
                        "environment": {
                            "PYTHONPATH": child_env(config, task)["PYTHONPATH"],
                            "DROID_EPISODE_ID": task["episode_id"],
                        },
                    },
                )
                run_child(command, child_env(config, task), dest / "run.log")
                run_child(
                    [
                        config["python"],
                        str(validation_entry),
                        "--_validate",
                        str(task_file),
                        str(run),
                        str(dest / "w"),
                    ],
                    child_env(config, task)
                    | {"DROID_REPLAY_RESOURCES": str(validation_entry.parent)},
                    dest / "validation.log",
                )
                result = read_json(run / "validation.json")
                row["status"] = (
                    "reproduced" if result["reproduced"] else "not_consistent"
                )
                row["physical_task_pass"] = result["physical_task_pass"]
                row["max_difference"] = result["replay_consistency"]["max_difference"]
            except KeyboardInterrupt:
                row["status"] = "interrupted"
                report(batch, rows)
                raise
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                if run is not None:
                    expose_results(task, dest, run)
                check_files(task["source_files"])
                if (
                    digest(Path(config["source_root"]) / "manifest.json")
                    != source_manifest_hash
                ):
                    raise RuntimeError(
                        "Historical global manifest changed during this run"
                    )
                report(batch, rows)
            print(f'END task {task["number"]}: {row["status"]}', flush=True)
        print(f'REPORT {batch / "index.html"}', flush=True)
        latest = HERE / (".latest-" + uuid.uuid4().hex)
        latest.symlink_to(batch, target_is_directory=True)
        os.replace(latest, HERE / "latest")
        return 0 if all(r["status"] == "reproduced" for r in rows) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as error:
        print(f"ERROR {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2)
