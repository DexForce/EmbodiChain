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

"""Generate two meshes and an assembly pose through a Codex action/observation loop."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import hashlib
from pathlib import Path
import sys
import tempfile
import time

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.tools.assemble._prompt import build_prompt
from scripts.tools.assemble._planning import resolve_design
from scripts.tools.assemble._protocol import (
    action_schema,
    codex_action,
    load_config,
    run_process,
)
from scripts.tools.assemble._json_io import pose_matrix, read_json, write_json

__all__ = ["build_parser", "run_harness", "main"]


def _check_action(action: dict) -> None:
    if not isinstance(action, dict) or set(action) != set(action_schema()["required"]):
        raise ValueError("Action has missing or unknown fields")
    name = action["action"]
    if name not in ("plan", "generate", "evaluate", "finish", "fail") or not isinstance(
        action["reason"], str
    ):
        raise ValueError("Invalid action or reason")
    owned = {
        "plan": "design",
        "generate": "source",
        "evaluate": "T_base_assemble",
        "finish": "candidate_id",
        "fail": None,
    }[name]
    for field in ("design", "source", "T_base_assemble", "candidate_id"):
        if field != owned and action[field] is not None:
            raise ValueError(f"{field} must be null for {name}")
    if name == "generate" and (
        not isinstance(action["source"], str) or not 1 <= len(action["source"]) <= 60000
    ):
        raise ValueError("generate.source must contain 1..60000 characters of Python")
    if name == "evaluate":
        pose_matrix(action["T_base_assemble"])
    if name == "finish" and type(action["candidate_id"]) is not int:
        raise ValueError("finish.candidate_id must be an integer")


def _generate(source: str, directory: Path, config: dict) -> dict:
    directory.mkdir()
    path = directory / "build.py"
    path.write_text(source, encoding="utf-8")
    compile(source, str(path), "exec")
    run_process(
        [
            sys.executable,
            str(Path(__file__).with_name("_build_worker.py")),
            "--source",
            str(path),
            "--max-faces",
            str(config["max_faces"]),
        ],
        directory,
        directory / "build",
        config["timeout_seconds"],
    )
    geometry = read_json(directory / "geometry.json")
    for role, asset in geometry["assets"].items():
        processing = asset.get("mesh_processing", {})
        if processing.get("decimated"):
            print(
                f"[assemble] {role} mesh simplified: "
                f"{processing['input_triangles']} -> {processing['exported_triangles']} "
                f"triangles (limit {config['max_faces']})",
                flush=True,
            )
    return geometry


def run_harness(
    config: dict,
    cycle: int = 1,
    decide: Callable = codex_action,
    result_file: Path | None = None,
) -> dict:
    """Execute a fresh generation job, saving JSON results even on failure.

    Args:
        config: Normalized config from load_config.
        cycle: Sequence number supplied by the interactive viewer.
        decide: Structured-action provider; injectable for offline tests.
        result_file: Optional job-specific handoff updated alongside each checkpoint.

    Returns:
        Complete result JSON with success flag, asset paths, and T_base_assemble.
    """
    from scripts.tools.assemble._validation import PlacementValidator

    started = time.perf_counter()
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f"cycle_{cycle:03d}_", dir=output))
    result = {
        "schema": "codex-assemble/v1",
        "success": False,
        "status": "running",
        "cycle": cycle,
        "run_directory": str(directory),
        "config": config,
        "design_plan": None,
        "resolved_validation": None,
        "assets": None,
        "T_base_assemble": None,
        "validation": None,
        "trace": [],
        "candidates": [],
        "timing_seconds": {
            "planning": 0.0,
            "objects": 0.0,
            "relative_pose": 0.0,
            "total": 0.0,
        },
        "reason": "Generation has started",
    }
    trace, candidates = result["trace"], result["candidates"]

    def save() -> None:
        result["timing_seconds"]["total"] = time.perf_counter() - started
        write_json(directory / "result.json", result)
        write_json(output / "latest.json", result)
        if result_file is not None:
            write_json(result_file, result)

    save()
    geometry = validator = design = resolved_validation = None
    generation = 0
    try:
        for turn in range(1, config["codex"]["max_turns"] + 1):
            print(f"[assemble] Cycle {cycle}, Codex turn {turn}", flush=True)
            turn_started = time.perf_counter()
            # If the model call fails without an action, charge the waiting stage.
            stage = (
                "planning"
                if design is None
                else ("objects" if validator is None else "relative_pose")
            )
            entry = None
            try:
                action = decide(
                    build_prompt(
                        config,
                        cycle,
                        trace,
                        geometry,
                        candidates,
                        design,
                        resolved_validation,
                    ),
                    directory,
                    turn,
                    config["codex"],
                )
                if isinstance(action, dict) and isinstance(action.get("action"), str):
                    stage = {
                        "plan": "planning",
                        "generate": "objects",
                        "evaluate": "relative_pose",
                        "finish": "relative_pose",
                    }.get(action.get("action"), stage)
                entry = {"turn": turn, "action": action}
                trace.append(entry)
                try:
                    _check_action(action)
                    name = action["action"]
                    print(f"[assemble] {name}: {action['reason']}", flush=True)
                    if name == "plan":
                        if design is not None:
                            raise ValueError(
                                "The accepted design and axis constraints are frozen for this cycle"
                            )
                        design, resolved_validation = resolve_design(
                            action["design"], config["validation"]
                        )
                        result.update(
                            design_plan=design, resolved_validation=resolved_validation
                        )
                        write_json(
                            directory / "design_plan.json",
                            {
                                "design": design,
                                "resolved_validation": resolved_validation,
                            },
                        )
                        entry["observation"] = {
                            "design_accepted": True,
                            "resolved_validation": resolved_validation,
                        }
                    elif name == "generate":
                        if design is None:
                            raise ValueError(
                                "Expand the descriptions with a valid plan before generating meshes"
                            )
                        # Invalidate the old assets BEFORE building, including when the new build fails.
                        geometry = validator = None
                        candidates.clear()
                        result["assets"] = None
                        generation += 1
                        geometry = _generate(
                            action["source"],
                            directory / f"generation_{generation:02d}",
                            config["geometry"],
                        )
                        validator = PlacementValidator(
                            geometry, resolved_validation, output / "cache"
                        )
                        result["assets"] = geometry["assets"]
                        entry["observation"] = geometry | {
                            "visacd_hull_counts": validator.hull_counts
                        }
                    elif name == "evaluate":
                        if validator is None:
                            raise ValueError(
                                "Generate valid objects before evaluating a pose"
                            )
                        candidate = validator.evaluate(action["T_base_assemble"])
                        candidate["candidate_id"] = turn
                        candidate["generation"] = generation
                        candidates.append(candidate)
                        entry["observation"] = candidate
                    elif name == "finish":
                        if validator is None:
                            raise ValueError("No generated assets are available")
                        selected = next(
                            (
                                x
                                for x in candidates
                                if x["candidate_id"] == action["candidate_id"]
                            ),
                            None,
                        )
                        if selected is None or not selected["validation"]["accepted"]:
                            raise ValueError(
                                "finish requires an accepted candidate from the current generation"
                            )
                        fresh = PlacementValidator(
                            geometry, resolved_validation, output / "cache"
                        )
                        check = fresh.validate(selected["T_base_assemble"])
                        if not check["accepted"]:
                            raise ValueError(
                                f"Independent final validation failed: {check}"
                            )
                        for asset in result["assets"].values():
                            asset["sha256"] = hashlib.sha256(
                                Path(asset["path"]).read_bytes()
                            ).hexdigest()
                        result.update(
                            success=True,
                            status="complete",
                            T_base_assemble=selected["T_base_assemble"],
                            validation=check,
                            selected_candidate_id=selected["candidate_id"],
                            geometry_json=str(
                                Path(result["assets"]["base"]["path"]).with_name(
                                    "geometry.json"
                                )
                            ),
                            reason=action["reason"],
                        )
                        entry["observation"] = {
                            "accepted": True,
                            "independently_revalidated": True,
                        }
                        return result
                    else:
                        result.update(status="failed", reason=action["reason"])
                        entry["observation"] = {"stopped": True}
                        return result
                except Exception as error:
                    entry["observation"] = {"error": f"{type(error).__name__}: {error}"}
                    print(f"[assemble] Feedback: {error}", flush=True)
            finally:
                elapsed = time.perf_counter() - turn_started
                result["timing_seconds"][stage] += elapsed
                if entry is not None:
                    entry["timing"] = {"stage": stage, "seconds": elapsed}
                label = {
                    "planning": "Planning",
                    "objects": "Object generation",
                    "relative_pose": "Relative pose generation",
                }[stage]
                print(
                    f"[assemble] {label}: {elapsed:.2f}s this turn; "
                    f"{result['timing_seconds'][stage]:.2f}s cumulative",
                    flush=True,
                )
            save()
        result.update(
            status="failed",
            reason="Maximum Codex turns reached without an accepted finish",
        )
    except (Exception, KeyboardInterrupt) as error:
        result.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            reason=f"{type(error).__name__}: {error}",
        )
        if isinstance(error, KeyboardInterrupt):
            raise
    finally:
        save()
        timing = result["timing_seconds"]
        print(
            f"[assemble] Cycle {cycle} timing ({result['status']}): "
            f"objects={timing['objects']:.2f}s, "
            f"relative pose={timing['relative_pose']:.2f}s, "
            f"planning={timing['planning']:.2f}s, total={timing['total']:.2f}s",
            flush=True,
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone JSON generation CLI without importing Blender or sim."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--cycle",
        type=int,
        default=1,
        help="Sequence number for a viewer generation job.",
    )
    parser.add_argument(
        "--result-file",
        type=Path,
        help="Also write the final result to this job-specific handoff file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Generate one assembly and return a process exit status.

    Args:
        argv: Optional CLI argument list.

    Returns:
        Zero for a verified result; one for an unsuccessful generation.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cycle < 1:
        parser.error("--cycle must be positive")
    if args.result_file and args.result_file.resolve() == args.config.resolve():
        parser.error("--result-file cannot overwrite the input configuration")
    result = run_harness(
        load_config(args.config),
        args.cycle,
        result_file=args.result_file.resolve() if args.result_file else None,
    )
    print(
        f"[assemble] {result['status']}: {result['run_directory']}/result.json",
        flush=True,
    )
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
