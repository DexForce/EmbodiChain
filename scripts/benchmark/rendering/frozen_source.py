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

"""Standard-library source pinning for untimed follow-up proof workers.

This module can be imported by its bare filename from an absolute worker script.
It deliberately imports no ``scripts`` package before original source is pinned.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

__all__ = [
    "artifact",
    "freeze_original_run",
    "load_original_run",
    "original_environment",
    "pin_source_imports",
    "read_json",
    "renderer_control_signature",
    "runtime_versions",
    "sha256",
    "stable_hash",
    "verify_imported_sources",
    "verify_source_worktree",
    "write_json",
]

_IDENTITY = (
    "experiment_id",
    "run_id",
    "case_id",
    "backend",
    "repeat",
    "config_sha256",
    "scene_sha256",
    "view_sha256",
    "profile_sha256",
    "hardware_id",
)
_QUALITY_FIELDS = {
    "quality_status",
    "quality_evaluation_id",
    "quality_evidence",
    "quality",
    "record_id",
}


def sha256(path: Path) -> str:
    """Hash a file with bounded memory."""
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def stable_hash(value: object) -> str:
    """Use the unchanged core JSON identity encoding without package imports."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def read_json(path: Path) -> Any:
    """Read finite JSON data without importing a runtime."""

    def invalid(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant: {value}")

    def finite_float(value: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"Non-finite JSON number: {value}")
        return result

    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=invalid,
        parse_float=finite_float,
    )


def write_json(path: Path, value: object) -> Path:
    """Write a producer artifact atomically; source files are never rewritten."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _inside(root: Path, value: str | Path) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Frozen evidence escapes its root")
    return path


def artifact(
    root: Path, path: Path, kind: str, *, role: str | None = None
) -> dict[str, Any]:
    """Describe one immutable evidence file relative to its proof root."""
    result = {
        "path": _inside(root, path).relative_to(root.resolve()).as_posix(),
        "sha256": sha256(path),
        "kind": kind,
    }
    if role is not None:
        result["role"] = role
    return result


def _run(command: list[str], *, cwd: Path) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=30,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0", "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if result.returncode:
        raise ValueError(
            "Read-only source verification failed: " + result.stderr.strip()[:1000]
        )
    return result.stdout.strip()


def load_original_run(experiment_root: Path, run_id: str) -> dict[str, Any]:
    """Read one completed original run and verify its raw/result/config bindings.

    Args:
        experiment_root: Frozen formal experiment root.
        run_id: Exact run identifier in its original raw ledger.

    Returns:
        Original row/config/capture and resolved source-artifact paths. No writes.
    """
    root = Path(experiment_root).resolve()
    raw_path = root / "raw.jsonl"
    matches = [
        json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()
    ]
    matches = [row for row in matches if row.get("run_id") == run_id]
    if len(matches) != 1:
        raise ValueError("Original raw ledger must contain exactly one requested run")
    row = matches[0]
    if row.get("status") != "completed" or row.get("backend") not in {
        "embodichain",
        "isaaclab",
    }:
        raise ValueError(
            "Follow-up proofs require a completed original native/Isaac run"
        )
    if any(row.get(key) is None for key in _IDENTITY):
        raise ValueError("Original run identity, hashes and GPU UUID must be present")
    if not row["hardware_id"] or not row.get("resources", {}).get("gpu", {}).get(
        "driver"
    ):
        raise ValueError("Original GPU UUID and driver evidence must be nonempty")
    run = _inside(root, row.get("run_dir", run_id))
    paths = {
        "raw": raw_path,
        "result": run / "result.json",
        "worker_config": run / "worker_config.json",
        "capture": run / "capture.json",
        "freshness": run / "freshness_probe.json",
        "definition": root / "definition.json",
    }
    result = read_json(paths["result"])
    clean = lambda value: {
        key: item for key, item in value.items() if key not in _QUALITY_FIELDS
    }
    if clean(result) != clean(row):
        raise ValueError("Original result differs from its immutable raw record")
    config = read_json(paths["worker_config"])
    if (
        config.get("case") != row.get("config")
        or stable_hash(config["case"]) != row["config_sha256"]
    ):
        raise ValueError("Original case/config checksum mismatch")
    if stable_hash(config["profile"]) != row["profile_sha256"]:
        raise ValueError("Original profile checksum mismatch")
    for key in ("scene_sha256", "view_sha256", "profile_sha256"):
        if config.get(key) != row[key]:
            raise ValueError("Original worker binding mismatch: " + key)
    definition = read_json(paths["definition"])
    if definition.get("definition_version") != "2.0":
        raise ValueError("Original definition must be formal v2")
    paths["scene"] = _inside(root, config["scene_path"])
    scene = read_json(paths["scene"])
    if stable_hash(scene) != row["scene_sha256"]:
        raise ValueError("Original scene checksum mismatch")
    for obj in scene["objects"]:
        geometry = obj["geometry"]
        if geometry["kind"] == "mesh":
            path = _inside(paths["scene"].parent, geometry["path"])
            if sha256(path) != geometry["sha256"]:
                raise ValueError("Original mesh checksum mismatch")
            paths["geometry:" + geometry["path"]] = path
    if config.get("replay_path"):
        paths["replay"] = _inside(root, config["replay_path"])
    for name in ("freshness_probe.npz", "quality_capture.json", "frames.json"):
        if (run / name).is_file():
            paths["original_" + name] = run / name
    capture = read_json(paths["capture"])
    freshness = read_json(paths["freshness"])
    if freshness.get("passed") is not True:
        raise ValueError("Original freshness prefix did not pass")
    return {
        "root": root,
        "run_root": run,
        "row": row,
        "config": config,
        "capture": capture,
        "scene": scene,
        "replay": read_json(paths["replay"]) if "replay" in paths else None,
        "paths": paths,
        "input_hashes": {name: sha256(path) for name, path in paths.items()},
    }


def verify_source_worktree(
    original: dict[str, Any], source_worktree: Path, source_commit: str
) -> dict[str, Any]:
    """Verify a clean original checkout and every recorded benchmark source hash."""
    root = Path(source_worktree).resolve()
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", source_commit):
        raise ValueError("An explicit hexadecimal source commit is required")
    commit = _run(
        ["git", "rev-parse", "--verify", source_commit + "^{commit}"], cwd=root
    )
    if _run(["git", "rev-parse", "HEAD"], cwd=root) != commit:
        raise ValueError("Source worktree HEAD differs from the requested commit")
    if _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root):
        raise ValueError("Original source worktree has tracked modifications")
    software = original["row"]["software"]
    if software.get("benchmark_commit") != commit:
        raise ValueError("Requested source commit differs from the timed original run")
    files = software.get("benchmark_source_sha256", {})
    required = {
        "scripts/benchmark/rendering/formal_worker.py",
        "scripts/benchmark/rendering/capture.py",
        "scripts/benchmark/core/measurement.py",
        "scripts/benchmark/rendering/backends/scene_"
        + original["row"]["backend"]
        + ".py",
    }
    if not isinstance(files, dict) or not required <= files.keys():
        raise ValueError(
            "Original source provenance lacks required capture/core hashes"
        )
    for relative, expected in files.items():
        if sha256(_inside(root, relative)) != expected:
            raise ValueError("Frozen source mismatch: " + relative)
    isaaclab_root = (
        original["config"].get("isaaclab_root")
        if original["row"]["backend"] == "isaaclab"
        else None
    )
    if isaaclab_root:
        external = Path(isaaclab_root).resolve()
        if _run(["git", "rev-parse", "HEAD"], cwd=external) != software.get(
            "isaaclab_commit"
        ):
            raise ValueError("External Isaac source commit changed")
        if _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=external):
            raise ValueError("External Isaac source has tracked modifications")
    return {
        "worktree": str(root),
        "commit": commit,
        "source_files": files,
        "isaaclab_root": isaaclab_root,
        "isaaclab_commit": software.get("isaaclab_commit"),
        "tracked_checkout_clean": True,
    }


def freeze_original_run(
    original: dict[str, Any], source: dict[str, Any], output: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Copy portable original evidence and return binding plus rebased worker config."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    mapping = {}
    for role, path in original["paths"].items():
        if sha256(path) != original["input_hashes"][role]:
            raise ValueError("Original input changed while freezing: " + role)
        relative = (
            ("original/scene/" + role.removeprefix("geometry:"))
            if role.startswith("geometry:")
            else (
                "original/scene/scene.json"
                if role == "scene"
                else (
                    "original/auxiliary/" + role.removeprefix("original_")
                    if role.startswith("original_")
                    else "original/"
                    + {
                        "worker_config": "worker_config.json",
                        "raw": "raw.jsonl",
                        "freshness": "freshness_probe.json",
                        "replay": "replay.json",
                    }.get(role, role + ".json")
                )
            )
        )
        target = _inside(output, relative)
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if sha256(target) != original["input_hashes"][role]:
            raise ValueError("Copied original evidence checksum mismatch")
        records.append(artifact(output, target, "frozen_original_input", role=role))
        mapping[role] = target
    for relative, expected in source["source_files"].items():
        target = _inside(output, "original/source/" + relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_inside(Path(source["worktree"]), relative), target)
        if sha256(target) != expected:
            raise ValueError("Source changed during snapshot: " + relative)
        records.append(
            artifact(
                output, target, "frozen_benchmark_source", role="source:" + relative
            )
        )
    row = original["row"]
    binding = {
        **{key: row[key] for key in _IDENTITY},
        "source_commit": source["commit"],
        "source_files": source["source_files"],
        "runtime_versions": deepcopy(row["software"]),
        "gpu_driver": row.get("resources", {}).get("gpu", {}).get("driver"),
        "artifacts": records,
    }
    config = deepcopy(original["config"])
    config["scene_path"] = str(mapping["scene"])
    config["replay_path"] = str(mapping["replay"]) if "replay" in mapping else None
    return binding, config


def original_environment(
    source_worktree: Path, interpreter: Path, isaaclab_root: Path | None = None
) -> dict[str, str]:
    """Call the original launcher environment helper without importing it here.

    Interpreter paths deliberately use :meth:`Path.absolute`, not symlink
    resolution: resolving an Isaac venv executable loses its virtualenv.
    """
    root, python = Path(source_worktree).resolve(), Path(interpreter).absolute()
    code = "import sys,json; from pathlib import Path; sys.path.insert(0,sys.argv[1]); from scripts.benchmark.rendering.formal_runner import _runtime_environment; print(json.dumps(_runtime_environment(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]) if sys.argv[3] else None)))"
    result = subprocess.run(
        [
            str(python),
            "-S",
            "-c",
            code,
            str(root),
            str(python),
            str(isaaclab_root or ""),
        ],
        cwd=root,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "GIT_OPTIONAL_LOCKS": "0"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode:
        raise ValueError("Original environment helper failed: " + result.stderr[:1000])
    environment = json.loads(result.stdout)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def pin_source_imports(source: dict[str, Any]) -> None:
    """Pin source paths in a fresh absolute-script worker before runtime imports."""
    if any(
        name == "scripts"
        or name.startswith("scripts.")
        or name == "embodichain"
        or name.startswith("embodichain.")
        or name.startswith("isaaclab")
        for name in sys.modules
    ):
        raise RuntimeError(
            "Source pinning must precede any benchmark/simulator imports"
        )
    root = Path(source["worktree"]).resolve()
    paths = [str(root)]
    if source.get("isaaclab_root"):
        paths.extend(
            str(path)
            for path in sorted((Path(source["isaaclab_root"]) / "source").iterdir())
            if path.is_dir()
        )
    sys.path[:0] = paths
    sys.dont_write_bytecode = True


def verify_imported_sources(source: dict[str, Any]) -> dict[str, Any]:
    """Check actual imported benchmark module origins against the original ledger."""
    root = Path(source["worktree"]).resolve()
    observed = {}
    for name, module in list(sys.modules.items()):
        if (
            name == "embodichain"
            or name.startswith("embodichain.")
            or name.startswith("isaaclab")
        ):
            filename = getattr(module, "__file__", None)
            if filename:
                path = Path(filename).resolve()
                owner = (
                    Path(source["isaaclab_root"]).resolve()
                    if name.startswith("isaaclab") and source.get("isaaclab_root")
                    else root
                )
                if not path.is_relative_to(owner):
                    raise RuntimeError(
                        "Imported simulator code from a different checkout: " + name
                    )
                observed[name] = {
                    "path": path.relative_to(owner).as_posix(),
                    "sha256": sha256(path),
                    "source_root": (
                        "isaaclab"
                        if name.startswith("isaaclab")
                        else "original_worktree"
                    ),
                }
            continue
        if not name.startswith(
            ("scripts.benchmark.core.", "scripts.benchmark.rendering.")
        ):
            continue
        filename = getattr(module, "__file__", None)
        if not filename or not filename.endswith(".py"):
            continue
        path = Path(filename).resolve()
        if not path.is_relative_to(root):
            raise RuntimeError(
                "Imported a modified/nonoriginal benchmark module: " + name
            )
        relative = path.relative_to(root).as_posix()
        digest = sha256(path)
        if source["source_files"].get(relative) != digest:
            raise RuntimeError(
                "Imported benchmark source lacks original matching hash: " + relative
            )
        observed[name] = {"path": relative, "sha256": digest}
    return observed


def runtime_versions(original: dict[str, Any], torch: Any = None) -> dict[str, Any]:
    """Read exact runtime versions and fail if they differ from the original run."""
    expected = original["runtime_versions"]
    packages = {}
    for name in expected.get("packages", {}):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    result = {"packages": packages, "python": sys.version}
    for key, value in result.items():
        if expected.get(key) != value:
            raise RuntimeError("Runtime differs from the timed original: " + key)
    if torch is not None:
        for key, value in (
            ("torch_runtime", getattr(torch, "__version__", None)),
            ("cuda_runtime", getattr(getattr(torch, "version", None), "cuda", None)),
        ):
            if key in expected and expected[key] != value:
                raise RuntimeError("Loaded runtime differs from original: " + key)
            result[key] = value
    return result


def renderer_control_signature(
    renderer: dict[str, Any], backend: str
) -> dict[str, Any]:
    """Keep actual shader/light controls, excluding explicit history counters."""
    keys = (
        "effective_lighting_controls",
        "postprocess_evidence",
        "rgb_delivery_conversion",
        "delivered_rgb_encoding",
    )
    required = ("effective_lighting_controls", "postprocess_evidence")
    if backend == "embodichain":
        keys += (
            "effective_profile",
            "resolved_render_configuration",
            "spp",
            "denoising",
            "tone_mapping",
            "publication_policy",
        )
        required += ("effective_profile", "resolved_render_configuration")
    elif backend == "isaaclab":
        keys += (
            "render_settings",
            "resolved_rtx_settings",
            "observed_renderer_settings",
            "light_evidence",
            "accumulation_setup",
        )
        required += (
            "resolved_rtx_settings",
            "observed_renderer_settings",
            "light_evidence",
        )
    else:
        raise ValueError("Unknown original backend")
    if any(not isinstance(renderer.get(key), dict) for key in required):
        raise ValueError("Actual renderer controls are incomplete")
    result = {key: deepcopy(renderer[key]) for key in keys if key in renderer}
    if "resolved_rtx_settings" in result:
        result["resolved_rtx_settings"].pop("/rtx/externalFrameCounter", None)
    return result
