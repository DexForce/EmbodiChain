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

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
from pathlib import Path
from typing import Any
import re

from embodichain.gen_sim.action_agent_pipeline.generation.config_io import (
    read_json,
    write_json,
)
from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    TargetReplacementSpec,
    _BasketTaskRoles,
    _ResolvedTargetReplacement,
)
from embodichain.gen_sim.action_agent_pipeline.generation.naming import (
    _normalize_runtime_uid,
)

__all__ = [
    "_apply_replacement_names",
    "_normalize_target_replacements",
    "_run_target_replacements",
    "_validate_target_replacement_sources",
]

_TARGET_REPLACEMENT_MANIFEST_FILENAME = ".embodichain_replacement_manifest.json"


def _normalize_target_replacements(
    target_replacements: Sequence[TargetReplacementSpec] | None,
) -> tuple[TargetReplacementSpec, ...]:
    if not target_replacements:
        return ()

    normalized = []
    seen_source_uids = set()
    seen_output_dirs = set()
    for replacement in target_replacements:
        if not isinstance(replacement, TargetReplacementSpec):
            raise TypeError(
                "target_replacements must contain TargetReplacementSpec values."
            )
        source_uid = str(replacement.source_uid).strip()
        prompt = str(replacement.prompt).strip()
        output_dir_name = str(replacement.output_dir_name).strip()
        if not source_uid:
            raise ValueError("target replacement source_uid must be non-empty.")
        if not prompt:
            raise ValueError("target replacement prompt must be non-empty.")
        if not output_dir_name:
            raise ValueError("target replacement output_dir_name must be non-empty.")
        output_dir_path = Path(output_dir_name)
        if (
            output_dir_path.is_absolute()
            or len(output_dir_path.parts) != 1
            or output_dir_name in {".", ".."}
        ):
            raise ValueError(
                "target replacement output_dir_name must be a single relative "
                f"directory name, got: {output_dir_name!r}"
            )
        if source_uid in seen_source_uids:
            raise ValueError(f"Duplicate target replacement source uid: {source_uid}")
        if output_dir_name in seen_output_dirs:
            raise ValueError(
                f"Duplicate target replacement output dir: {output_dir_name}"
            )
        seen_source_uids.add(source_uid)
        seen_output_dirs.add(output_dir_name)
        normalized.append(
            TargetReplacementSpec(
                source_uid=source_uid,
                prompt=prompt,
                output_dir_name=output_dir_name,
            )
        )
    return tuple(normalized)


def _validate_target_replacement_sources(
    roles: _BasketTaskRoles,
    replacement_specs: Sequence[TargetReplacementSpec],
) -> None:
    if not replacement_specs:
        return

    target_source_uids = {
        roles.left_target_source_uid,
        roles.right_target_source_uid,
    }
    unknown = [
        replacement.source_uid
        for replacement in replacement_specs
        if replacement.source_uid not in target_source_uids
    ]
    if unknown:
        raise ValueError(
            "target_replacements must reference the selected basket target "
            f"source uid(s) {sorted(target_source_uids)}, got: {unknown}"
        )


def _run_target_replacements(
    *,
    scene_dir: Path,
    replacement_specs: Sequence[TargetReplacementSpec],
    reuse_target_replacements: bool,
    prompt2geometry_runner: Callable[..., Mapping[str, Any]] | None = None,
) -> tuple[_ResolvedTargetReplacement, ...]:
    if prompt2geometry_runner is None:
        prompt2geometry_runner = _run_prompt2geometry_replacement

    resolved = []
    for replacement in replacement_specs:
        runtime_noun = _replacement_runtime_noun(replacement.prompt)
        output_root = scene_dir / "mesh_assets" / replacement.output_dir_name
        output_name = f"{runtime_noun}.glb"
        mesh_path = None
        reused = False
        if reuse_target_replacements:
            mesh_path = _resolve_reusable_target_replacement_mesh_path(
                output_root=output_root,
                prompt=replacement.prompt,
                output_name=output_name,
            )
            reused = mesh_path is not None
        if mesh_path is None:
            result = prompt2geometry_runner(
                prompt=replacement.prompt,
                output_root=output_root,
                output_name=output_name,
            )
            mesh_path = _resolve_prompt2geometry_mesh_path(result, output_root)
            _write_target_replacement_manifest(
                output_root=output_root,
                prompt=replacement.prompt,
                output_name=output_name,
                mesh_path=mesh_path,
            )
        elif reused:
            _write_target_replacement_manifest(
                output_root=output_root,
                prompt=replacement.prompt,
                output_name=output_name,
                mesh_path=mesh_path,
            )
        resolved.append(
            _ResolvedTargetReplacement(
                source_uid=replacement.source_uid,
                prompt=replacement.prompt,
                output_dir_name=replacement.output_dir_name,
                mesh_path=mesh_path,
                runtime_noun=runtime_noun,
                reused=reused,
            )
        )
    return tuple(resolved)


def _resolve_reusable_target_replacement_mesh_path(
    *,
    output_root: Path,
    prompt: str,
    output_name: str,
) -> Path | None:
    expected_mesh_path = (output_root / output_name).expanduser().resolve()
    if not expected_mesh_path.is_file():
        return None

    manifest_path = _target_replacement_manifest_path(output_root)
    if not manifest_path.is_file():
        return expected_mesh_path

    try:
        manifest = read_json(manifest_path)
    except (OSError, json.JSONDecodeError):
        return None

    if manifest.get("prompt") != prompt or manifest.get("output_name") != output_name:
        return None

    manifest_mesh_path = Path(
        str(manifest.get("mesh_path", expected_mesh_path))
    ).expanduser()
    if not manifest_mesh_path.is_absolute():
        manifest_mesh_path = (output_root / manifest_mesh_path).resolve()
    else:
        manifest_mesh_path = manifest_mesh_path.resolve()
    if manifest_mesh_path.is_file():
        return manifest_mesh_path
    return expected_mesh_path


def _write_target_replacement_manifest(
    *,
    output_root: Path,
    prompt: str,
    output_name: str,
    mesh_path: Path,
) -> None:
    write_json(
        _target_replacement_manifest_path(output_root),
        {
            "prompt": prompt,
            "output_name": output_name,
            "mesh_path": mesh_path.expanduser().resolve().as_posix(),
        },
    )


def _target_replacement_manifest_path(output_root: Path) -> Path:
    return output_root / _TARGET_REPLACEMENT_MANIFEST_FILENAME


def _run_prompt2geometry_replacement(
    *,
    prompt: str,
    output_root: Path,
    output_name: str,
) -> dict[str, Any]:
    from embodichain.gen_sim.action_agent_pipeline.gym_project_api.prompt2geometry import (
        Prompt2GeometryRequest,
        load_prompt2geometry_config,
        run_prompt2geometry,
    )

    cfg = load_prompt2geometry_config()
    return run_prompt2geometry(
        Prompt2GeometryRequest(
            prompt=prompt,
            output_root=output_root,
            output_name=output_name,
            zimage_base_url=cfg.zimage_base_url,
            sam3_base_url=cfg.sam3_base_url,
            sam3d_base_url=cfg.sam3d_base_url,
            llm_api_key=cfg.llm_api_key,
            llm_model=cfg.llm_model,
            llm_base_url=cfg.llm_base_url,
            llm_timeout_s=cfg.llm_timeout_s,
        )
    )


def _resolve_prompt2geometry_mesh_path(
    result: Mapping[str, Any],
    output_root: Path,
) -> Path:
    raw_path = result.get("scaled_mesh_path") or result.get("mesh_path")
    if not raw_path:
        raise ValueError("prompt2geometry result did not include a GLB mesh path.")

    mesh_path = Path(str(raw_path)).expanduser()
    if not mesh_path.is_absolute():
        mesh_path = (output_root / mesh_path).resolve()
    else:
        mesh_path = mesh_path.resolve()

    if not mesh_path.is_file():
        raise FileNotFoundError(f"Generated replacement GLB not found: {mesh_path}")
    return mesh_path


def _replacement_runtime_noun(prompt: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", prompt.lower())
    while tokens and tokens[0] in {"a", "an", "the"}:
        tokens.pop(0)
    stem = "_".join(tokens)
    if not stem:
        stem = "replacement_object"
    return _normalize_runtime_uid(stem)


def _apply_replacement_names(
    roles: _BasketTaskRoles,
    resolved_replacements: Sequence[_ResolvedTargetReplacement],
) -> _BasketTaskRoles:
    replacement_by_uid = {
        replacement.source_uid: replacement for replacement in resolved_replacements
    }
    left_replacement = replacement_by_uid.get(roles.left_target_source_uid)
    right_replacement = replacement_by_uid.get(roles.right_target_source_uid)
    left_target_noun = (
        left_replacement.runtime_noun
        if left_replacement is not None
        else roles.left_target_noun
    )
    right_target_noun = (
        right_replacement.runtime_noun
        if right_replacement is not None
        else roles.right_target_noun
    )
    target_noun = (
        left_target_noun if left_target_noun == right_target_noun else "target_object"
    )
    return _BasketTaskRoles(
        table_source_uid=roles.table_source_uid,
        container_source_uid=roles.container_source_uid,
        left_target_source_uid=roles.left_target_source_uid,
        right_target_source_uid=roles.right_target_source_uid,
        container_runtime_uid=roles.container_runtime_uid,
        left_target_runtime_uid=f"left_{left_target_noun}",
        right_target_runtime_uid=f"right_{right_target_noun}",
        target_noun=target_noun,
        left_target_noun=left_target_noun,
        right_target_noun=right_target_noun,
        container_noun=roles.container_noun,
    )
