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

"""Strict JSON configuration, action schema, and Codex subprocess transport."""

from __future__ import annotations

import math
import os
from pathlib import Path
import shutil
import signal
import subprocess

from scripts.tools.assemble._planning import design_schema
from scripts.tools.assemble._json_io import read_json, write_json


def load_config(path: Path) -> dict:
    """Resolve a description-driven job and validate all user-owned settings.

    Args:
        path: JSON config; output paths are relative to this file.

    Returns:
        Normalized JSON-compatible configuration.
    """
    path = path.resolve()
    value = read_json(path)
    defaults = {
        "output_dir": "../outputs",
        "codex": {"model": None, "max_turns": 10, "timeout_seconds": 300},
        "geometry": {"timeout_seconds": 180, "max_faces": 100000},
        "validation": {
            "collision": "visacd",
            "concavity": 0.015,
            "max_parts": 64,
            "contact_gap": 0.0001,
            "contact_tolerance": 0.0003,
            "probe": 0.001,
            "settle_distance": 0.05,
            "settle_step": 0.001,
            "assemble_axis": None,
            "target_axis": None,
            "axis_tolerance_degrees": 5.0,
        },
    }
    required = {"base_description", "assemble_description", "action_description"}
    if not required <= value.keys() or value.keys() - required - defaults.keys():
        raise ValueError(
            "Config requires base_description, assemble_description, action_description; unknown fields are not allowed"
        )
    for key in required:
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    for section, default in defaults.items():
        if isinstance(default, dict):
            supplied = value.get(section, {})
            if not isinstance(supplied, dict) or supplied.keys() - default.keys():
                raise ValueError(f"Unknown or invalid {section} settings")
            value[section] = default | supplied
        else:
            value.setdefault(section, default)
    if not isinstance(value["output_dir"], str) or not value["output_dir"].strip():
        raise ValueError("output_dir must be a nonempty path")
    value["output_dir"] = str((path.parent / value["output_dir"]).resolve())
    if path.is_relative_to(Path(value["output_dir"])):
        raise ValueError(
            "Keep configuration files outside the generated output directory"
        )
    for section, key, low, high, integer in [
        ("codex", "max_turns", 4, 30, True),
        ("codex", "timeout_seconds", 1, 1800, False),
        ("geometry", "timeout_seconds", 1, 1800, False),
        ("geometry", "max_faces", 12, 1000000, True),
        ("validation", "max_parts", 1, 256, True),
        ("validation", "concavity", 0.0001, 0.5, False),
        ("validation", "contact_gap", 0.000001, 0.01, False),
        ("validation", "contact_tolerance", 0.000001, 0.02, False),
        ("validation", "probe", 0.00001, 0.05, False),
        ("validation", "settle_distance", 0.0, 0.5, False),
        ("validation", "settle_step", 0.0001, 0.01, False),
        ("validation", "axis_tolerance_degrees", 0, 90, False),
    ]:
        number = value[section][key]
        if (
            type(number) not in (int, float)
            or not math.isfinite(number)
            or not low <= number <= high
            or (integer and type(number) is not int)
        ):
            raise ValueError(
                f"{section}.{key} must be {'an integer' if integer else 'a number'} in [{low}, {high}]"
            )
    v = value["validation"]
    if v["collision"] not in ("visacd", "exact"):
        raise ValueError("validation.collision must be visacd or exact")
    if not v["contact_gap"] <= v["contact_tolerance"] < v["probe"]:
        raise ValueError("Require contact_gap <= contact_tolerance < probe")
    for key in ("assemble_axis", "target_axis"):
        axis = v[key]
        if axis is not None:
            if (
                not isinstance(axis, list)
                or len(axis) != 3
                or any(
                    type(x) not in (int, float) or not math.isfinite(x) for x in axis
                )
            ):
                raise ValueError(f"validation.{key} must be a finite 3-vector or null")
            norm = math.hypot(*axis)
            if not math.isfinite(norm) or norm < 1e-12:
                raise ValueError(f"validation.{key} must have a finite nonzero norm")
            v[key] = [x / norm for x in axis]
    if (v["assemble_axis"] is None) != (v["target_axis"] is None):
        raise ValueError("assemble_axis and target_axis must be specified together")
    model = value["codex"]["model"]
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError("codex.model must be a model name or null")
    value["config_path"] = str(path)
    return value


def action_schema() -> dict:
    """Return the strict structured-output schema shared by every model turn."""
    row = {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}
    matrix = {"type": "array", "items": row, "minItems": 4, "maxItems": 4}
    properties = {
        "action": {
            "type": "string",
            "enum": ["plan", "generate", "evaluate", "finish", "fail"],
        },
        "reason": {"type": "string"},
        "design": {"anyOf": [design_schema(), {"type": "null"}]},
        "source": {"type": ["string", "null"]},
        "T_base_assemble": {"anyOf": [matrix, {"type": "null"}]},
        "candidate_id": {"type": ["integer", "null"]},
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def run_process(
    command: list[str],
    cwd: Path,
    prefix: Path,
    timeout: float,
    stdin: str | None = None,
) -> None:
    """Run one subprocess with retained logs and bounded process-group lifetime.

    Args:
        command: Argument list, never interpreted by a shell.
        cwd: Working directory.
        prefix: Log filename prefix.
        timeout: Maximum elapsed seconds.
        stdin: Optional complete prompt.
    """
    with (
        Path(str(prefix) + ".stdout.log").open("w") as out,
        Path(str(prefix) + ".stderr.log").open("w") as err,
    ):
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            text=True,
            start_new_session=True,
        )
        try:
            process.communicate(stdin, timeout=timeout)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    if process.returncode:
        tail = Path(str(prefix) + ".stderr.log").read_text(errors="replace")[-6000:]
        raise RuntimeError(
            f"{command[0]} exited {process.returncode}: {tail}; logs: {prefix}.*.log"
        )


def codex_action(prompt: str, run_dir: Path, turn: int, config: dict) -> dict:
    """Request one GPT action through Codex CLI using its existing login.

    Args:
        prompt: Job, tools, and full observation history.
        run_dir: Isolated audit directory.
        turn: Sequential decision number.
        config: Model and timeout settings.

    Returns:
        The structured final response, not the JSONL event stream.
    """
    executable = shutil.which("codex")
    if executable is None:
        raise RuntimeError("Codex CLI is not on PATH; install it and run codex login")
    schema = run_dir / "action.schema.json"
    write_json(schema, action_schema())
    prefix = run_dir / f"turn_{turn:02d}"
    response = prefix.with_suffix(".json")
    prefix.with_suffix(".prompt.txt").write_text(prompt, encoding="utf-8")
    command = [
        executable,
        "exec",
        "--ignore-user-config",
        "--ephemeral",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--disable",
        "shell_tool",
        "--disable",
        "apps",
        "--disable",
        "multi_agent",
        "-c",
        "project_doc_max_bytes=0",
        "-c",
        'web_search="disabled"',
        "--json",
        "--color",
        "never",
        "--output-schema",
        str(schema),
        "--output-last-message",
        str(response),
        "--cd",
        str(run_dir),
    ]
    if config["model"]:
        command.extend(["--model", config["model"]])
    command.append("-")
    write_json(prefix.with_suffix(".command.json"), {"argv": command})
    run_process(command, run_dir, prefix, config["timeout_seconds"], prompt)
    return read_json(response)
