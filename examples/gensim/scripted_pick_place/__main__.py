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

"""Rebuild and execute a scripted GenSim pick/place example with wall-clock timing."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

__all__: list[str] = []


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _prepare(output: Path) -> None:
    import trimesh
    from PIL import Image

    Image.new("RGB", (100, 80), "white").save(output / "input.png")
    export = output / "scene/scene_export"
    export.mkdir(parents=True)
    entries = []
    for uid, extents, position, body_type in (
        ("table", (0.6, 0.04, 1.0), (0.6, 0.0, 0.73), "static"),
        ("cube_native", (0.04, 0.04, 0.04), (0.45, 0.02, 0.771), "dynamic"),
    ):
        trimesh.creation.box(extents).export(export / f"{uid}.glb")
        entries.append(
            {
                "uid": uid,
                "category": "table" if uid == "table" else "cube",
                "name": uid,
                "description": uid,
                "shape": {"shape_type": "Mesh", "fpath": f"{uid}.glb"},
                "init_pos": list(position),
                "init_rot": [0, 0, 0],
                "body_scale": [1, 1, 1],
                "body_type": body_type,
                "attrs": {"mass_props": {"mass": 0.05}},
                "max_convex_hull_num": 1,
            }
        )
    _write_json(
        export / "scene_config.json",
        {
            "format": "embodichain.scene-export/v1",
            "scene_id": "test-scene",
            "background": entries[:1],
            "rigid_object": entries[1:],
            "articulation": [],
        },
    )


def _worker(output: Path) -> int:
    from collections import deque
    from importlib.metadata import version

    import torch
    import yaml

    from embodichain.gen_sim._run_config import _load_run_config
    from embodichain.gen_sim._workflow import _run
    from embodichain.gen_sim.services import (
        ChatResponse,
        GenSimServices,
        GenSimServicesCfg,
    )

    class ScriptedChat:
        def __init__(self, responses: list[dict[str, Any]]) -> None:
            self.responses = deque(responses)

        def complete(self, request: Any) -> ChatResponse:
            if not self.responses:
                raise RuntimeError(
                    "The example's scripted model responses were exhausted."
                )
            return ChatResponse(text=json.dumps(self.responses.popleft()))

    torch.set_num_threads(4)
    started = time.perf_counter()
    _prepare(output)
    resources = Path(__file__).resolve().parent
    runtime = yaml.safe_load((resources / "runtime.yaml").read_text())
    (output / "runtime.yaml").write_text(yaml.safe_dump(runtime, sort_keys=False))
    config = {
        "schema_version": "embodichain.gensim-run/v1",
        "image": "input.png",
        "instruction": "Put the cube on the table",
        "output": "run",
        "runtime": "runtime.yaml",
        "execution_policy": "extensions/embodichain_tasks/configs/components/execution_policies/motion_gen_verified.yaml",
        "scene": "scene",
        "task_id": "gensim_task",
        "headless": True,
    }
    (output / "run.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    responses = json.loads((resources / "responses.json").read_text())
    _write_json(output / "responses.json", responses)
    preparation_seconds = time.perf_counter() - started
    workflow_started = time.perf_counter()
    status, exit_code = "error", 1
    try:
        cfg = _load_run_config(argparse.Namespace(config=output / "run.yaml"))
        with GenSimServices(
            GenSimServicesCfg(),
            injected={role: ScriptedChat(values) for role, values in responses.items()},
        ) as services:
            result = _run(cfg, services=services)
        status = result["status"]
        exit_code = 0 if status == "succeeded" else 1
    except KeyboardInterrupt:
        status, exit_code = "interrupted", 130
    except Exception:
        traceback.print_exc()
    finally:
        _write_json(
            output / "timing.json",
            {
                "status": status,
                "workflow_exit_code": exit_code,
                "input_preparation_seconds": preparation_seconds,
                "workflow_wall_seconds": time.perf_counter() - workflow_started,
                "timing_scope": "configuration loading through shared workflow return, including runtime cleanup",
                "model_services": "scripted; no live model inference or image reconstruction",
                "rendering": "no-render",
                "seed": 0,
                "torch_threads": 4,
                "asset_cache": "existing download, URDF assembly and grasp caches are retained",
                "python": sys.version.split()[0],
                "embodichain": version("embodichain"),
                "dexsim": version("dexsim-engine"),
            },
        )
    return exit_code


def _comparison(current: Path, previous: Path) -> dict[str, Any]:
    import yaml

    def payload(root: Path, relative: str) -> Any:
        return json.loads((root / "run" / relative).read_text())

    def verdict(root: Path) -> dict[str, Any]:
        report = payload(root, "execution_report.json")
        return {
            "status": report["status"],
            "steps": report["length"],
            "segments": [
                {
                    key: segment[key]
                    for key in ("name", "start_step", "end_step", "outcome_kind")
                }
                for segment in report["segments"]
            ],
        }

    checks = {}
    for relative in (
        "task/semantic_task_graph.json",
        "scene/scene_export/scene_config.json",
        "binding_report.json",
        "deployment/program.json",
    ):
        # Reused scene inputs live outside run/ in this example.
        if relative.startswith("scene/"):
            a = json.loads((current / relative).read_text())
            b = json.loads((previous / relative).read_text())
        else:
            a, b = payload(current, relative), payload(previous, relative)
        if relative.endswith("program.json"):
            a.pop("program_id", None)
            b.pop("program_id", None)
        if relative == "binding_report.json":
            a = {key: a[key] for key in ("objects", "resources")}
            b = {key: b[key] for key in ("objects", "resources")}
        checks[relative] = a == b
    for relative in ("input.png", "responses.json", "runtime.yaml"):
        checks[relative] = (current / relative).read_bytes() == (
            previous / relative
        ).read_bytes()
    for relative in ("deployment/embodiment.yaml", "deployment/execution_policy.yaml"):
        checks[relative] = yaml.safe_load(
            (current / "run" / relative).read_text()
        ) == yaml.safe_load((previous / "run" / relative).read_text())
    now, before = verdict(current), verdict(previous)
    return {
        "same_semantic_artifacts": all(checks.values()),
        "artifact_checks": checks,
        "same_segment_outcomes_and_steps": now == before,
        "current": now,
        "previous": before,
        "scope": "semantic artifacts, segment verdicts and step spans; no per-step numeric trajectory comparison",
    }


def main() -> int:
    """Run one isolated attempt and preserve failure status and intermediate outputs.

    Returns:
        The native workflow's exit status, or an interrupted worker's status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="New example output directory."
    )
    parser.add_argument(
        "--compare", type=Path, help="Previous example output directory."
    )
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if args._worker:
        return _worker(output)
    if args.compare is not None:
        args.compare = args.compare.expanduser().resolve()
        if not (args.compare / "run/execution_report.json").is_file():
            parser.error("--compare must name a completed example output directory")
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        parser.error(str(error))

    import embodichain

    env = dict(os.environ)
    sdk_root = str(Path(embodichain.__file__).resolve().parents[1])
    inherited = env.get("PYTHONPATH")
    env["PYTHONPATH"] = sdk_root + (os.pathsep + inherited if inherited else "")
    started = time.perf_counter()
    print(
        f"Running a fresh workflow; native output is saved to {output / 'worker.log'}",
        flush=True,
    )
    try:
        with (output / "worker.log").open("w") as stream:
            process = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--_worker",
                    "--output",
                    str(output),
                ],
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        exit_code = process.returncode
    except KeyboardInterrupt:
        exit_code = 130
    elapsed = time.perf_counter() - started
    _write_json(
        output / "process_timing.json",
        {
            "worker_process_wall_seconds": elapsed,
            "worker_exit_code": exit_code,
            "scope": "worker startup/imports, procedural inputs, complete shared workflow, cleanup and Python shutdown",
            "excludes": "parent launcher startup and optional post-run comparison",
        },
    )
    if args.compare is not None and (output / "run/execution_report.json").is_file():
        _write_json(output / "comparison.json", _comparison(output, args.compare))
    print(f"Full worker process wall time: {elapsed:.3f} s; exit code: {exit_code}")
    print(f"Task graph: {output / 'run/task/semantic_task_graph.html'}")
    print(f"Execution report: {output / 'run/execution_report.json'}")
    print(f"Timing: {output / 'process_timing.json'}")
    return exit_code if exit_code >= 0 else 128 - exit_code


if __name__ == "__main__":
    raise SystemExit(main())
