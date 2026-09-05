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

"""Build and load scene-level USD snapshots for Scene Engine exports."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import shutil
from contextlib import contextmanager
from typing import Any

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import ArticulationCfg, LightCfg, MeshCfg, RigidObjectCfg
from embodichain.lab.sim.objects import Articulation
from embodichain.lab.visualization import VisualizationCfg

_SCENE_EXPORT_FORMAT = "embodichain.scene-export/v1"
_SCENE_USD_FORMAT = "embodichain.scene-usd/v1"


def load_scene_export_into_sim(
    *,
    sim: SimulationManager,
    output_root: str | Path,
    force_static_rigids: bool = False,
    runtime_assets: dict[str, Path] | None = None,
) -> list[Articulation]:
    """Add the mixed GLB/USDC scene export to an existing simulator.

    GLB entries create rigid bodies and USDC entries create articulations.  An
    articulated object's GLB proxy is deliberately not loaded, so it cannot
    duplicate the runtime articulation or hide its texture/material bindings.
    """
    config_path = _scene_export_config_path(output_root)
    scene_config = _read_scene_export(config_path)
    _add_lights(sim)
    _add_objects(
        sim=sim,
        entries=_config_entries(scene_config, "background"),
        config_dir=config_path.parent,
        label="table",
        force_static=force_static_rigids,
        runtime_assets=runtime_assets,
    )
    _add_objects(
        sim=sim,
        entries=_config_entries(scene_config, "rigid_object"),
        config_dir=config_path.parent,
        label="asset",
        force_static=force_static_rigids,
        runtime_assets=runtime_assets,
    )
    return _add_articulations(
        sim=sim,
        entries=_config_entries(scene_config, "articulation"),
        config_dir=config_path.parent,
        runtime_assets=runtime_assets,
    )


def build_scene_usd(
    *,
    output_root: str | Path,
    device: str = "cpu",
) -> Path:
    """Materialize a Scene Engine export as a self-contained USD package.

    ``scene.usda`` remains the scene-level USD interchange artifact.  Its
    adjacent ``assets/`` directory contains externally textured GLTF/USDC
    runtime payloads, and the manifest binds every payload to its scene pose.
    Native DexSim preview uses those payloads because its current flattened
    USD mesh importer does not preserve GLB internal node transforms.

    Args:
        output_root: Scene Engine output root containing ``scene_export/``.
        device: Simulation device used for the transient export scene.

    Returns:
        Absolute path to ``scene_usd/scene.usda``.
    """
    resolved_output_root = Path(output_root).expanduser().resolve()
    config_path = _scene_export_config_path(resolved_output_root)
    scene_config = _read_scene_export(config_path)
    scene_usd_root = resolved_output_root / "scene_usd"
    scene_usd_root.mkdir(parents=True, exist_ok=True)
    scene_usd_path = scene_usd_root / "scene.usda"
    temporary_scene_usd_path = scene_usd_root / "scene.in_progress.usda"
    manifest_path = scene_usd_root / "scene_usd_manifest.json"
    runtime_assets_root = scene_usd_root / ".export_runtime_assets"
    packaged_assets_root = scene_usd_root / "assets"
    temporary_scene_usd_path.unlink(missing_ok=True)

    sim = SimulationManager(
        SimulationManagerCfg(
            width=1920,
            height=1080,
            headless=True,
            physics_dt=1.0 / 100.0,
            sim_device=device,
            visualization=VisualizationCfg(),
        )
    )
    try:
        if sim.is_use_gpu_physics:
            sim.init_gpu_physics()
        runtime_assets = _prepare_runtime_assets(
            scene_config=scene_config,
            config_dir=config_path.parent,
            runtime_assets_root=runtime_assets_root,
        )
        packaged_assets = _package_runtime_assets(
            runtime_assets=runtime_assets,
            destination_root=packaged_assets_root,
        )
        articulations = load_scene_export_into_sim(
            sim=sim,
            output_root=resolved_output_root,
            runtime_assets=runtime_assets,
        )
        # The native exporter calls MaterialInst.get_base_color_map().  That
        # pybind getter is unsafe for embedded GLB maps (it can raise
        # MemoryError/SystemError).  Export the material scalar values first,
        # then bind the packaged files below using the standard USD graph.
        with _safe_usd_material_export():
            if not sim.export_usd(str(temporary_scene_usd_path)):
                raise RuntimeError(
                    "DexSim could not export scene USD without dropping render "
                    f"materials/textures: {scene_usd_path}"
                )
        _apply_runtime_textures_to_usd(
            scene_usd_path=temporary_scene_usd_path,
            scene_usd_root=scene_usd_root,
            runtime_assets=runtime_assets,
            sim=sim,
        )
        temporary_scene_usd_path.replace(scene_usd_path)
        _write_scene_usd_manifest(
            path=manifest_path,
            scene_config=scene_config,
            scene_usd_path=scene_usd_path,
            output_root=resolved_output_root,
            sim=sim,
            articulations=articulations,
            packaged_assets=packaged_assets,
        )
    except Exception:
        # A stale snapshot is worse than no snapshot: ``--usd`` would load a
        # layout that no longer matches the freshly written scene_export.
        scene_usd_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        shutil.rmtree(packaged_assets_root, ignore_errors=True)
        raise
    finally:
        temporary_scene_usd_path.unlink(missing_ok=True)
        shutil.rmtree(runtime_assets_root, ignore_errors=True)
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()
    return scene_usd_path


def load_scene_usd_into_sim(
    *,
    sim: SimulationManager,
    output_root: str | Path,
) -> list[Articulation]:
    """Load a generated scene USD package and restore its named resources."""
    resolved_output_root = Path(output_root).expanduser().resolve()
    manifest_path = resolved_output_root / "scene_usd" / "scene_usd_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Scene USD manifest not found. Generate or re-export the Scene Engine "
            f"output first: {manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Scene USD manifest is not valid JSON: {manifest_path}"
        ) from exc
    if not isinstance(manifest, dict) or manifest.get("format") != _SCENE_USD_FORMAT:
        raise ValueError(f"Expected format={_SCENE_USD_FORMAT!r} in {manifest_path}.")
    raw_scene_usd = manifest.get("scene_usd")
    entries = manifest.get("objects")
    if (
        not isinstance(raw_scene_usd, str)
        or not isinstance(entries, list)
        or not all(isinstance(entry, dict) for entry in entries)
    ):
        raise ValueError(
            "Scene USD manifest must contain scene_usd and object entries."
        )
    scene_usd_path = (resolved_output_root / raw_scene_usd).resolve()
    if (
        resolved_output_root not in scene_usd_path.parents
        or not scene_usd_path.is_file()
    ):
        raise FileNotFoundError(f"Scene USD not found: {scene_usd_path}")
    if all(isinstance(entry.get("runtime_asset"), str) for entry in entries):
        return _load_packaged_scene_usd(
            sim=sim,
            output_root=resolved_output_root,
            entries=entries,
        )

    # Compatibility for scene_usd snapshots generated before the package
    # contained runtime assets.  The legacy flattened USD meshes are still a
    # useful interchange artifact, but DexSim cannot render GLB submesh
    # transforms faithfully after a USD round trip.  Rebuild from the paired
    # Scene Export assets instead, preserving the exact native GLB appearance.
    _resolve_manifest_source_config(output_root=resolved_output_root, manifest=manifest)
    return load_scene_export_into_sim(
        sim=sim,
        output_root=resolved_output_root,
        force_static_rigids=True,
    )


def _load_packaged_scene_usd(
    *,
    sim: SimulationManager,
    output_root: Path,
    entries: list[dict[str, Any]],
) -> list[Articulation]:
    """Build a native preview from the self-contained scene USD package.

    ``scene.usda`` is the package's standard USD interchange artifact.  For
    native DexSim rendering, the adapter uses its preserved GLB/USDC payloads:
    DexSim's flattened-USD mesh importer loses GLB internal node transforms,
    whereas native payload loading retains the render hierarchy, textures, and
    articulation behavior.
    """
    _add_lights(sim)
    articulations: list[Articulation] = []
    for entry in entries:
        uid = entry.get("uid")
        kind = entry.get("kind")
        if not isinstance(uid, str) or not uid:
            raise ValueError("Scene USD manifest object has no valid uid.")
        asset_path = _resolve_manifest_runtime_asset(
            output_root=output_root,
            entry=entry,
            uid=uid,
            suffix=".gltf" if kind == "rigid" else ".usdc",
        )
        init_pos = tuple(_vector3(entry.get("init_pos"), f"{uid}.init_pos"))
        init_rot = tuple(_vector3(entry.get("init_rot"), f"{uid}.init_rot"))
        body_scale = tuple(
            _vector3(entry.get("body_scale", [1.0, 1.0, 1.0]), f"{uid}.body_scale")
        )
        if kind == "rigid":
            shape = MeshCfg(fpath=str(asset_path))
            shape.load_option.gltfloader = True
            sim.add_rigid_object(
                RigidObjectCfg(
                    uid=uid,
                    shape=shape,
                    body_type="static",
                    init_pos=init_pos,
                    init_rot=init_rot,
                    body_scale=body_scale,
                    max_convex_hull_num=max(
                        1, int(entry.get("max_convex_hull_num", 32))
                    ),
                    acd_method="vhacd",
                )
            )
        elif kind == "articulation":
            if entry.get("fix_base", True) is not True:
                raise ValueError(
                    f"Scene USD articulation {uid!r} must set fix_base=true."
                )
            articulations.append(
                sim.add_articulation(
                    ArticulationCfg(
                        uid=uid,
                        fpath=str(asset_path),
                        init_pos=init_pos,
                        init_rot=init_rot,
                        body_scale=body_scale,
                        fix_base=True,
                        build_pk_chain=False,
                    )
                )
            )
        else:
            raise ValueError(f"Scene USD object {uid!r} has invalid kind {kind!r}.")
    return articulations


def _resolve_manifest_runtime_asset(
    *,
    output_root: Path,
    entry: dict[str, Any],
    uid: str,
    suffix: str,
) -> Path:
    raw_asset = entry.get("runtime_asset")
    if not isinstance(raw_asset, str):
        raise ValueError(f"Scene USD object {uid!r} has no runtime_asset.")
    asset_path = (output_root / raw_asset).resolve()
    if (
        output_root not in asset_path.parents
        or asset_path.suffix.lower() != suffix
        or not asset_path.is_file()
    ):
        raise FileNotFoundError(
            f"Scene USD {uid!r} runtime asset is invalid or missing: {asset_path}"
        )
    return asset_path


def _scene_export_config_path(output_root: str | Path) -> Path:
    config_path = (
        Path(output_root).expanduser().resolve() / "scene_export" / "scene_config.json"
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"Scene config not found: {config_path}")
    return config_path


def _resolve_manifest_source_config(
    *, output_root: Path, manifest: dict[str, Any]
) -> Path:
    raw_path = manifest.get("source_scene_export")
    if not isinstance(raw_path, str):
        raise ValueError("Scene USD manifest has no source_scene_export path.")
    source_path = (output_root / raw_path).resolve()
    if output_root not in source_path.parents or not source_path.is_file():
        raise FileNotFoundError(f"Source scene export not found: {source_path}")
    return source_path


def _read_scene_export(config_path: Path) -> dict[str, Any]:
    try:
        scene_config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Scene config is not valid JSON: {config_path}") from exc
    if not isinstance(scene_config, dict):
        raise ValueError("Scene config must be a JSON object.")
    if scene_config.get("format") != _SCENE_EXPORT_FORMAT:
        raise ValueError(
            "Expected an EmbodiChain scene export "
            f"(format={_SCENE_EXPORT_FORMAT!r})."
        )
    return scene_config


def _config_entries(
    scene_config: dict[str, Any], field_name: str
) -> list[dict[str, Any]]:
    entries = scene_config.get(field_name, [])
    if not isinstance(entries, list) or not all(
        isinstance(entry, dict) for entry in entries
    ):
        raise ValueError(
            f"Scene config field {field_name!r} must be a list of objects."
        )
    return entries


def _add_lights(sim: SimulationManager) -> None:
    for index in range(8):
        angle = 2.0 * math.pi * index / 8
        sim.add_light(
            LightCfg(
                uid=f"light_{index + 1}",
                intensity=80.0,
                radius=600,
                init_pos=[5.0 * math.cos(angle), 5.0 * math.sin(angle), 8.0],
            )
        )


def _add_objects(
    *,
    sim: SimulationManager,
    entries: list[dict[str, Any]],
    config_dir: Path,
    label: str,
    force_static: bool = False,
    runtime_assets: dict[str, Path] | None = None,
) -> None:
    """Add exported GLB meshes while preserving their embedded GLB materials."""
    resolved_config_dir = config_dir.resolve()
    for entry in entries:
        uid = entry.get("uid")
        shape = entry.get("shape")
        if not isinstance(uid, str) or not uid:
            raise ValueError(f"Scene {label} has no valid uid.")
        if not isinstance(shape, dict) or not isinstance(shape.get("fpath"), str):
            raise ValueError(f"Scene {label} {uid!r} has no shape.fpath.")
        if shape.get("shape_type") != "Mesh":
            raise ValueError(f"Scene {label} {uid!r} must use shape_type='Mesh'.")
        mesh_path = _resolve_asset_path(
            config_dir=resolved_config_dir,
            raw_path=shape["fpath"],
            suffix=".glb",
            uid=uid,
            label=label,
        )
        runtime_path = runtime_assets.get(uid) if runtime_assets is not None else None
        if runtime_path is not None:
            mesh_path = runtime_path
        body_scale = _vector3(
            entry.get("body_scale", [1.0, 1.0, 1.0]), f"{uid}.body_scale"
        )
        body_type = "static" if force_static else entry.get("body_type", "static")
        if body_type not in {"dynamic", "kinematic", "static"}:
            raise ValueError(
                f"Scene {label} {uid!r} has invalid body_type {body_type!r}."
            )
        shape_cfg = MeshCfg(fpath=str(mesh_path))
        shape_cfg.load_option.gltfloader = mesh_path.suffix.lower() == ".gltf"
        sim.add_rigid_object(
            RigidObjectCfg(
                uid=uid,
                shape=shape_cfg,
                body_type=body_type,
                init_pos=tuple(_vector3(entry.get("init_pos"), f"{uid}.init_pos")),
                init_rot=tuple(_vector3(entry.get("init_rot"), f"{uid}.init_rot")),
                body_scale=tuple(body_scale),
                max_convex_hull_num=max(1, int(entry.get("max_convex_hull_num", 32))),
                acd_method="vhacd",
            )
        )


def _add_articulations(
    *,
    sim: SimulationManager,
    entries: list[dict[str, Any]],
    config_dir: Path,
    runtime_assets: dict[str, Path] | None = None,
) -> list[Articulation]:
    """Add one USDC articulation per entry, without its GLB proxy."""
    articulations: list[Articulation] = []
    for entry in entries:
        uid = entry.get("uid")
        if not isinstance(uid, str) or not uid:
            raise ValueError("Articulation entry has no valid uid.")
        usdc_path = _resolve_asset_path(
            config_dir=config_dir.resolve(),
            raw_path=entry.get("fpath"),
            suffix=".usdc",
            uid=uid,
            label="articulation",
        )
        runtime_path = runtime_assets.get(uid) if runtime_assets is not None else None
        if runtime_path is not None:
            usdc_path = runtime_path
        if entry.get("fix_base", True) is not True:
            raise ValueError(f"Articulation entry {uid!r} must set fix_base=true.")
        articulations.append(
            sim.add_articulation(
                ArticulationCfg(
                    uid=uid,
                    fpath=str(usdc_path),
                    init_pos=tuple(_vector3(entry.get("init_pos"), f"{uid}.init_pos")),
                    init_rot=tuple(_vector3(entry.get("init_rot"), f"{uid}.init_rot")),
                    body_scale=tuple(
                        _vector3(
                            entry.get("body_scale", [1.0, 1.0, 1.0]),
                            f"{uid}.body_scale",
                        )
                    ),
                    fix_base=True,
                    build_pk_chain=False,
                )
            )
        )
    return articulations


def _prepare_runtime_assets(
    *,
    scene_config: dict[str, Any],
    config_dir: Path,
    runtime_assets_root: Path,
) -> dict[str, Path]:
    """Create external-texture runtime assets without changing scene_export."""
    if runtime_assets_root.exists():
        shutil.rmtree(runtime_assets_root)
    runtime_assets_root.mkdir(parents=True)
    runtime_assets: dict[str, Path] = {}
    for field_name in ("background", "rigid_object"):
        for entry in _config_entries(scene_config, field_name):
            uid = entry.get("uid")
            shape = entry.get("shape")
            if not isinstance(uid, str) or not isinstance(shape, dict):
                raise ValueError(f"Scene {field_name} entry is missing uid or shape.")
            source_glb = _resolve_asset_path(
                config_dir=config_dir,
                raw_path=shape.get("fpath"),
                suffix=".glb",
                uid=uid,
                label=field_name,
            )
            runtime_assets[uid] = _externalize_glb_textures(
                source_glb=source_glb,
                destination_root=runtime_assets_root / uid,
            )
    for entry in _config_entries(scene_config, "articulation"):
        uid = entry.get("uid")
        if not isinstance(uid, str):
            raise ValueError("Scene articulation entry has no valid uid.")
        source_usdc = _resolve_asset_path(
            config_dir=config_dir,
            raw_path=entry.get("fpath"),
            suffix=".usdc",
            uid=uid,
            label="articulation",
        )
        runtime_assets[uid] = _externalize_usdc_textures(
            source_usdc=source_usdc,
            destination_root=runtime_assets_root / uid,
        )
    return runtime_assets


def _package_runtime_assets(
    *,
    runtime_assets: dict[str, Path],
    destination_root: Path,
) -> dict[str, Path]:
    """Persist the externally textured runtime assets beside ``scene.usda``.

    The transient assets are used to create a standard USD texture package.
    Keeping an identical copy in the delivery directory additionally lets the
    native DexSim adapter preserve GLB submesh transforms that its USD mesh
    round-trip currently cannot represent faithfully.
    """
    shutil.rmtree(destination_root, ignore_errors=True)
    packaged_assets: dict[str, Path] = {}
    for uid, source_asset in runtime_assets.items():
        target_root = destination_root / uid
        shutil.copytree(source_asset.parent, target_root)
        packaged_assets[uid] = target_root / source_asset.name
    return packaged_assets


def _externalize_glb_textures(*, source_glb: Path, destination_root: Path) -> Path:
    """Write a glTF whose images are external PNG files instead of GLB blobs."""
    import trimesh

    scene = trimesh.load(source_glb, force="scene")
    exported = scene.export(file_type="gltf")
    if not isinstance(exported, dict) or "model.gltf" not in exported:
        raise RuntimeError(f"Could not convert GLB to glTF: {source_glb}")
    try:
        tree = json.loads(exported["model.gltf"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Converted glTF is not valid JSON: {source_glb}") from exc
    image_specs = tree.get("images", [])
    if not isinstance(image_specs, list) or not all(
        isinstance(image_spec, dict) for image_spec in image_specs
    ):
        raise RuntimeError(f"Converted glTF has invalid image entries: {source_glb}")
    destination_root.mkdir(parents=True, exist_ok=True)
    texture_root = destination_root / "textures"
    if image_specs:
        texture_root.mkdir()
    for index, image_spec in enumerate(image_specs):
        image_data, suffix = _gltf_embedded_image_data(
            image_spec=image_spec,
            tree=tree,
            exported=exported,
            source_glb=source_glb,
        )
        texture_name = f"image_{index}{suffix}"
        (texture_root / texture_name).write_bytes(image_data)
        image_spec.pop("bufferView", None)
        image_spec.pop("mimeType", None)
        image_spec["uri"] = f"textures/{texture_name}"
    for name, payload in exported.items():
        destination = destination_root / name
        if name == "model.gltf":
            destination.write_text(
                json.dumps(tree, separators=(",", ":")), encoding="utf-8"
            )
        elif isinstance(payload, bytes):
            destination.write_bytes(payload)
        else:
            raise RuntimeError(f"Unexpected glTF payload {name!r} for {source_glb}.")
    return destination_root / "model.gltf"


def _gltf_embedded_image_data(
    *,
    image_spec: dict[str, Any],
    tree: dict[str, Any],
    exported: dict[str, Any],
    source_glb: Path,
) -> tuple[bytes, str]:
    """Extract one image authored in a trimesh glTF buffer view.

    Reading the converted glTF's ``images`` array, rather than material
    attributes, preserves every PBR texture channel and the exporter-defined
    image order.
    """
    buffer_view_index = image_spec.get("bufferView")
    mime_type = image_spec.get("mimeType")
    if not isinstance(buffer_view_index, int) or not isinstance(mime_type, str):
        raise RuntimeError(
            "Converted glTF image must use an embedded buffer view and MIME type: "
            f"{source_glb}"
        )
    buffer_views = tree.get("bufferViews")
    buffers = tree.get("buffers")
    if (
        not isinstance(buffer_views, list)
        or not isinstance(buffers, list)
        or buffer_view_index < 0
        or buffer_view_index >= len(buffer_views)
        or not isinstance(buffer_views[buffer_view_index], dict)
    ):
        raise RuntimeError(
            f"Converted glTF image has invalid buffer view: {source_glb}"
        )
    buffer_view = buffer_views[buffer_view_index]
    buffer_index = buffer_view.get("buffer")
    byte_length = buffer_view.get("byteLength")
    byte_offset = buffer_view.get("byteOffset", 0)
    if (
        not isinstance(buffer_index, int)
        or not isinstance(byte_length, int)
        or not isinstance(byte_offset, int)
        or buffer_index < 0
        or buffer_index >= len(buffers)
        or not isinstance(buffers[buffer_index], dict)
    ):
        raise RuntimeError(
            f"Converted glTF image has invalid buffer data: {source_glb}"
        )
    buffer_uri = buffers[buffer_index].get("uri")
    payload = exported.get(buffer_uri) if isinstance(buffer_uri, str) else None
    if not isinstance(payload, bytes):
        raise RuntimeError(f"Converted glTF image buffer is missing: {source_glb}")
    image_data = payload[byte_offset : byte_offset + byte_length]
    if len(image_data) != byte_length:
        raise RuntimeError(f"Converted glTF image exceeds its buffer: {source_glb}")
    return image_data, _image_suffix_from_mime_type(mime_type, source_glb)


def _image_suffix_from_mime_type(mime_type: str, source_glb: Path) -> str:
    """Return the file suffix required by a standard glTF image MIME type."""
    suffix_by_mime_type = {
        "image/jpeg": ".jpg",
        "image/ktx2": ".ktx2",
        "image/png": ".png",
        "image/webp": ".webp",
    }
    suffix = suffix_by_mime_type.get(mime_type.lower())
    if suffix is None:
        raise RuntimeError(
            f"Converted glTF image uses unsupported MIME type {mime_type!r}: "
            f"{source_glb}"
        )
    return suffix


def _externalize_usdc_textures(*, source_usdc: Path, destination_root: Path) -> Path:
    """Copy USDC and rewrite directly referenced texture assets locally."""
    from pxr import Sdf, Usd

    destination_root.mkdir(parents=True, exist_ok=True)
    destination_usdc = destination_root / "model.usdc"
    shutil.copy2(source_usdc, destination_usdc)
    source_stage = Usd.Stage.Open(str(source_usdc))
    destination_stage = Usd.Stage.Open(str(destination_usdc))
    if source_stage is None or destination_stage is None:
        raise RuntimeError(f"Could not open USDC for texture packaging: {source_usdc}")
    texture_root = destination_root / "textures"
    copied_names: set[str] = set()
    for source_prim in source_stage.TraverseAll():
        destination_prim = destination_stage.GetPrimAtPath(source_prim.GetPath())
        for source_attr in source_prim.GetAttributes():
            value = source_attr.Get()
            if not isinstance(value, Sdf.AssetPath) or not value.resolvedPath:
                continue
            source_asset = Path(value.resolvedPath)
            if not source_asset.is_file():
                continue
            texture_root.mkdir(exist_ok=True)
            name = source_asset.name
            if name in copied_names:
                name = f"{len(copied_names)}_{name}"
            copied_names.add(name)
            target_asset = texture_root / name
            shutil.copy2(source_asset, target_asset)
            destination_attr = destination_prim.GetAttribute(source_attr.GetName())
            destination_attr.Set(
                Sdf.AssetPath(target_asset.relative_to(destination_root).as_posix())
            )
    destination_stage.GetRootLayer().Save()
    return destination_usdc


@contextmanager
def _safe_usd_material_export() -> Any:
    """Temporarily avoid DexSim's unsafe embedded-texture material getter."""
    from dexsim.kit.usd import mesh_object

    original_exporter = mesh_object._apply_material_to_usd
    mesh_object._apply_material_to_usd = _export_material_without_texture
    try:
        yield
    finally:
        mesh_object._apply_material_to_usd = original_exporter


def _export_material_without_texture(
    source_material: object, stage: object, source_path: object
) -> object:
    """Export safe scalar material values; texture links are authored later."""
    from pxr import Gf, Sdf, UsdShade

    original_name = source_material.get_name().replace(".filamat", "")
    material_name = original_name
    create_name = not Sdf.Path.IsValidIdentifier(material_name)
    if create_name:
        material_name = material_name or "material"
        material_name = re.sub(r"[^a-zA-Z0-9_]", "_", material_name)
        if material_name[0].isdigit():
            material_name = f"_{material_name}"
    material_path = source_path.AppendChild(material_name)
    material = UsdShade.Material.Define(stage, material_path)
    material.CreateInput("frame:stPrimvarName", Sdf.ValueTypeNames.Token).Set("st")
    if create_name:
        material.GetPrim().CreateAttribute("name", Sdf.ValueTypeNames.String).Set(
            original_name
        )
    shader = UsdShade.Shader.Define(stage, material_path.AppendChild("PBRShader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    _set_safe_material_input(
        source_material=source_material,
        getter_name="get_base_color",
        shader=shader,
        input_name="diffuseColor",
        value_type=Sdf.ValueTypeNames.Color3f,
        converter=lambda value: Gf.Vec3f(*map(float, value[:3])),
    )
    _set_safe_material_input(
        source_material=source_material,
        getter_name="get_emissive",
        shader=shader,
        input_name="emissiveColor",
        value_type=Sdf.ValueTypeNames.Color3f,
        converter=lambda value: Gf.Vec3f(*map(float, value[:3])),
    )
    _set_safe_material_input(
        source_material=source_material,
        getter_name="get_roughness",
        shader=shader,
        input_name="roughness",
        value_type=Sdf.ValueTypeNames.Float,
        converter=float,
    )
    _set_safe_material_input(
        source_material=source_material,
        getter_name="get_ior",
        shader=shader,
        input_name="ior",
        value_type=Sdf.ValueTypeNames.Float,
        converter=float,
    )
    return material


def _set_safe_material_input(
    *,
    source_material: object,
    getter_name: str,
    shader: object,
    input_name: str,
    value_type: object,
    converter: object,
) -> None:
    """Set a scalar material input without making USD export fragile."""
    try:
        value = getattr(source_material, getter_name)()
    except (MemoryError, SystemError, RuntimeError, TypeError, ValueError):
        return
    if value is not None:
        shader.CreateInput(input_name, value_type).Set(converter(value))


def _apply_runtime_textures_to_usd(
    *,
    scene_usd_path: Path,
    scene_usd_root: Path,
    runtime_assets: dict[str, Path],
    sim: SimulationManager,
) -> None:
    """Bind packaged GLTF PBR maps to the exported USD materials."""
    from pxr import Usd

    stage = Usd.Stage.Open(str(scene_usd_path))
    if stage is None:
        raise RuntimeError(f"Could not reopen exported USD: {scene_usd_path}")
    texture_root = scene_usd_root / "textures"
    bound_texture_count = 0
    exported_objects = [
        (uid, sim.get_rigid_object(uid)) for uid in sim.get_rigid_object_uid_list()
    ]
    exported_objects.extend(
        (uid, sim.get_articulation(uid)) for uid in sim.get_articulation_uid_list()
    )
    for uid, scene_object in exported_objects:
        runtime_asset = runtime_assets.get(uid)
        if runtime_asset is None or scene_object is None:
            continue
        runtime_name = scene_object._entities[0].get_name()
        object_prim = _find_named_prim(stage, runtime_name)
        if object_prim is None:
            raise RuntimeError(
                f"Exported USD has no prim for textured object {uid!r} "
                f"({runtime_name!r})."
            )
        texture_root.mkdir(exist_ok=True)
        if runtime_asset.suffix.lower() == ".gltf":
            gltf_texture_paths = _gltf_material_texture_paths(runtime_asset)
            if not gltf_texture_paths:
                continue
            materials_by_index = _usd_materials_by_gltf_index(
                stage=stage,
                object_prim=object_prim,
            )
            for material_index, texture_paths in gltf_texture_paths.items():
                material = materials_by_index.get(material_index)
                if material is None:
                    raise RuntimeError(
                        "Exported USD has no material corresponding to GLTF "
                        f"material index {material_index} for {uid!r}."
                    )
                relative_textures = {
                    channel: _copy_runtime_texture(
                        source_texture=source_texture,
                        texture_root=texture_root,
                        scene_usd_root=scene_usd_root,
                        uid=uid,
                    )
                    for channel, source_texture in texture_paths.items()
                }
                _bind_gltf_material_textures(
                    stage=stage,
                    material=material,
                    relative_textures=relative_textures,
                )
                bound_texture_count += len(relative_textures)
        else:
            # Articulated USDC assets preserve their own material graph during
            # export. Keep the previous base-colour fallback for their external
            # texture files until DexSim exposes a USDC material-index mapping.
            source_textures = sorted((runtime_asset.parent / "textures").glob("*"))
            if not source_textures:
                continue
            relative_texture = _copy_runtime_texture(
                source_texture=source_textures[0],
                texture_root=texture_root,
                scene_usd_root=scene_usd_root,
                uid=uid,
            )
            for material in _usd_bound_materials(stage=stage, object_prim=object_prim):
                _bind_base_color_texture(
                    stage=stage,
                    material=material,
                    relative_texture=relative_texture,
                )
                bound_texture_count += 1
    stage.GetRootLayer().Save()
    if any((path.parent / "textures").is_dir() for path in runtime_assets.values()):
        if bound_texture_count == 0:
            raise RuntimeError(
                "Exported USD did not expose any texture-bindable material."
            )


def _find_named_prim(stage: object, name: str) -> object | None:
    for prim in stage.TraverseAll():
        if prim.GetName() == name:
            return prim
    return None


def _gltf_material_texture_paths(
    runtime_asset: Path,
) -> dict[int, dict[str, Path]]:
    """Resolve every PBR texture image used by each packaged GLTF material."""
    try:
        tree = json.loads(runtime_asset.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Packaged GLTF is not valid JSON: {runtime_asset}") from exc
    if not isinstance(tree, dict):
        raise RuntimeError(f"Packaged GLTF must be a JSON object: {runtime_asset}")
    images = tree.get("images", [])
    textures = tree.get("textures", [])
    materials = tree.get("materials", [])
    if not all(isinstance(value, list) for value in (images, textures, materials)):
        raise RuntimeError(f"Packaged GLTF has invalid PBR arrays: {runtime_asset}")

    material_texture_paths: dict[int, dict[str, Path]] = {}
    for material_index, material in enumerate(materials):
        if not isinstance(material, dict):
            raise RuntimeError(f"Packaged GLTF has invalid material: {runtime_asset}")
        pbr = material.get("pbrMetallicRoughness", {})
        if not isinstance(pbr, dict):
            raise RuntimeError(
                f"Packaged GLTF has invalid PBR material: {runtime_asset}"
            )
        texture_specs = {
            "base_color": pbr.get("baseColorTexture"),
            "metallic_roughness": pbr.get("metallicRoughnessTexture"),
            "normal": material.get("normalTexture"),
            "occlusion": material.get("occlusionTexture"),
            "emissive": material.get("emissiveTexture"),
        }
        paths: dict[str, Path] = {}
        for channel, texture_spec in texture_specs.items():
            texture_index = (
                texture_spec.get("index") if isinstance(texture_spec, dict) else None
            )
            if texture_index is None:
                continue
            if not isinstance(texture_index, int) or not 0 <= texture_index < len(
                textures
            ):
                raise RuntimeError(
                    f"Packaged GLTF has invalid {channel} texture index: {runtime_asset}"
                )
            texture = textures[texture_index]
            if not isinstance(texture, dict):
                raise RuntimeError(
                    f"Packaged GLTF has invalid texture: {runtime_asset}"
                )
            image_index = texture.get("source")
            if not isinstance(image_index, int) or not 0 <= image_index < len(images):
                raise RuntimeError(
                    f"Packaged GLTF has invalid {channel} image index: {runtime_asset}"
                )
            image = images[image_index]
            image_uri = image.get("uri") if isinstance(image, dict) else None
            if not isinstance(image_uri, str):
                raise RuntimeError(
                    f"Packaged GLTF {channel} image must use an external URI: "
                    f"{runtime_asset}"
                )
            texture_path = (runtime_asset.parent / image_uri).resolve()
            if runtime_asset.parent.resolve() not in texture_path.parents:
                raise RuntimeError(
                    f"Packaged GLTF {channel} image escapes its package: {runtime_asset}"
                )
            if not texture_path.is_file():
                raise FileNotFoundError(
                    f"Packaged GLTF {channel} image is missing: {texture_path}"
                )
            paths[channel] = texture_path
        if paths:
            material_texture_paths[material_index] = paths
    return material_texture_paths


def _usd_bound_materials(*, stage: object, object_prim: object) -> list[object]:
    """Return the direct USD materials bound to meshes below one object prim."""
    from pxr import Usd, UsdGeom, UsdShade

    material_paths: set[object] = set()
    for prim in Usd.PrimRange(object_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        material_paths.update(
            UsdShade.MaterialBindingAPI(prim).GetDirectBindingRel().GetTargets()
        )
    materials = [UsdShade.Material.Get(stage, path) for path in material_paths]
    return [material for material in materials if material]


def _usd_materials_by_gltf_index(
    *, stage: object, object_prim: object
) -> dict[int, object]:
    """Match DexSim-exported material names to their GLTF material indices."""
    materials_by_index: dict[int, object] = {}
    for material in _usd_bound_materials(stage=stage, object_prim=object_prim):
        match = re.search(
            r"(?:^|_)gltf_material_index_(?P<index>\d+)$",
            material.GetPrim().GetName(),
        )
        if match is None:
            continue
        material_index = int(match.group("index"))
        if material_index in materials_by_index:
            raise RuntimeError(
                "Exported USD binds multiple materials for GLTF material index "
                f"{material_index}: {object_prim.GetPath()}"
            )
        materials_by_index[material_index] = material
    return materials_by_index


def _copy_runtime_texture(
    *,
    source_texture: Path,
    texture_root: Path,
    scene_usd_root: Path,
    uid: str,
) -> str:
    """Copy one package-local texture and return its scene-USD-relative URI."""
    target_texture = texture_root / f"{uid}_{source_texture.name}"
    if not target_texture.is_file():
        shutil.copy2(source_texture, target_texture)
    return target_texture.relative_to(scene_usd_root).as_posix()


def _bind_gltf_material_textures(
    *, stage: object, material: object, relative_textures: dict[str, str]
) -> None:
    """Author all supported GLTF PBR texture channels on one USD material."""
    from pxr import Sdf, UsdShade

    surface_output = material.GetSurfaceOutput()
    source = surface_output.GetConnectedSource()
    if not source:
        raise RuntimeError(f"Material has no surface shader: {material.GetPath()}")
    shader = UsdShade.Shader(source[0].GetPrim())
    material_path = material.GetPath()
    st_reader = UsdShade.Shader.Define(stage, material_path.AppendChild("stReader"))
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

    for channel, relative_texture in relative_textures.items():
        texture = _usd_texture_shader(
            stage=stage,
            material_path=material_path,
            channel=channel,
            relative_texture=relative_texture,
            st_reader=st_reader,
        )
        if channel == "base_color":
            _connect_usd_texture(
                shader=shader,
                input_name="diffuseColor",
                input_type=Sdf.ValueTypeNames.Color3f,
                texture=texture,
                output_name="rgb",
                output_type=Sdf.ValueTypeNames.Float3,
            )
        elif channel == "metallic_roughness":
            _connect_usd_texture(
                shader=shader,
                input_name="metallic",
                input_type=Sdf.ValueTypeNames.Float,
                texture=texture,
                output_name="b",
                output_type=Sdf.ValueTypeNames.Float,
            )
            _connect_usd_texture(
                shader=shader,
                input_name="roughness",
                input_type=Sdf.ValueTypeNames.Float,
                texture=texture,
                output_name="g",
                output_type=Sdf.ValueTypeNames.Float,
            )
        elif channel == "normal":
            texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
            texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
                (2.0, 2.0, 2.0, 1.0)
            )
            texture.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(
                (-1.0, -1.0, -1.0, 0.0)
            )
            _connect_usd_texture(
                shader=shader,
                input_name="normal",
                input_type=Sdf.ValueTypeNames.Normal3f,
                texture=texture,
                output_name="rgb",
                output_type=Sdf.ValueTypeNames.Float3,
            )
        elif channel == "occlusion":
            texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
            _connect_usd_texture(
                shader=shader,
                input_name="occlusion",
                input_type=Sdf.ValueTypeNames.Float,
                texture=texture,
                output_name="r",
                output_type=Sdf.ValueTypeNames.Float,
            )
        elif channel == "emissive":
            _connect_usd_texture(
                shader=shader,
                input_name="emissiveColor",
                input_type=Sdf.ValueTypeNames.Color3f,
                texture=texture,
                output_name="rgb",
                output_type=Sdf.ValueTypeNames.Float3,
            )


def _usd_texture_shader(
    *,
    stage: object,
    material_path: object,
    channel: str,
    relative_texture: str,
    st_reader: object,
) -> object:
    """Define one USD UV texture node with a shared ``st`` reader."""
    from pxr import Sdf, UsdShade

    texture = UsdShade.Shader.Define(
        stage,
        material_path.AppendChild(f"{channel}Texture"),
    )
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(relative_texture)
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        st_reader.ConnectableAPI(), "result"
    )
    texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    return texture


def _connect_usd_texture(
    *,
    shader: object,
    input_name: str,
    input_type: object,
    texture: object,
    output_name: str,
    output_type: object,
) -> None:
    """Connect one typed output of a USD texture node to a surface input."""
    texture.CreateOutput(output_name, output_type)
    shader.CreateInput(input_name, input_type).ConnectToSource(
        texture.ConnectableAPI(), output_name
    )


def _bind_base_color_texture(
    *, stage: object, material: object, relative_texture: str
) -> None:
    """Attach one standard UsdUVTexture graph to a bound Preview Surface."""
    from pxr import Sdf, UsdShade

    surface_output = material.GetSurfaceOutput()
    source = surface_output.GetConnectedSource()
    if not source:
        raise RuntimeError(f"Material has no surface shader: {material.GetPath()}")
    shader = UsdShade.Shader(source[0].GetPrim())
    material_path = material.GetPath()
    st_reader = UsdShade.Shader.Define(stage, material_path.AppendChild("stReader"))
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    texture = UsdShade.Shader.Define(stage, material_path.AppendChild("diffuseTexture"))
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(relative_texture)
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        st_reader.ConnectableAPI(), "result"
    )
    texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.ConnectableAPI(), "rgb"
    )


def _resolve_asset_path(
    *,
    config_dir: Path,
    raw_path: object,
    suffix: str,
    uid: str,
    label: str,
) -> Path:
    if not isinstance(raw_path, str):
        raise ValueError(f"Scene {label} {uid!r} has no asset path.")
    relative_path = Path(raw_path)
    if relative_path.is_absolute() or relative_path.suffix.lower() != suffix:
        raise ValueError(f"Scene {label} {uid!r} must use a relative {suffix} asset.")
    asset_path = (config_dir / relative_path).resolve()
    if config_dir not in asset_path.parents:
        raise ValueError(f"Scene {label} {uid!r} asset must stay within {config_dir}.")
    if not asset_path.is_file():
        raise FileNotFoundError(f"Scene {label} {uid!r} asset not found: {asset_path}")
    return asset_path


def _vector3(value: object, field_name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"Scene config field {field_name!r} must be a length-3 list.")
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Scene config field {field_name!r} must be numeric.") from exc


def _write_scene_usd_manifest(
    *,
    path: Path,
    scene_config: dict[str, Any],
    scene_usd_path: Path,
    output_root: Path,
    sim: SimulationManager,
    articulations: list[Articulation],
    packaged_assets: dict[str, Path],
) -> None:
    articulation_by_uid = {
        articulation.cfg.uid: articulation for articulation in articulations
    }
    objects: list[dict[str, object]] = []
    for field_name in ("background", "rigid_object"):
        for entry in _config_entries(scene_config, field_name):
            uid = str(entry["uid"])
            rigid = sim.get_rigid_object(uid)
            if rigid is None:
                raise RuntimeError(f"Scene USD export did not register rigid {uid!r}.")
            runtime_asset = packaged_assets.get(uid)
            if runtime_asset is None:
                raise RuntimeError(f"Scene USD export did not package rigid {uid!r}.")
            objects.append(
                {
                    "uid": uid,
                    "kind": "rigid",
                    "body_type": entry["body_type"],
                    "runtime_name": rigid._entities[0].get_name(),
                    "source_asset": entry["shape"]["fpath"],
                    "runtime_asset": runtime_asset.relative_to(output_root).as_posix(),
                    "init_pos": _vector3(entry["init_pos"], f"{uid}.init_pos"),
                    "init_rot": _vector3(entry["init_rot"], f"{uid}.init_rot"),
                    "body_scale": _vector3(
                        entry.get("body_scale", [1.0, 1.0, 1.0]),
                        f"{uid}.body_scale",
                    ),
                    "max_convex_hull_num": max(
                        1, int(entry.get("max_convex_hull_num", 32))
                    ),
                }
            )
    for entry in _config_entries(scene_config, "articulation"):
        uid = str(entry["uid"])
        articulation = articulation_by_uid.get(uid)
        if articulation is None:
            raise RuntimeError(
                f"Scene USD export did not register articulation {uid!r}."
            )
        runtime_asset = packaged_assets.get(uid)
        if runtime_asset is None:
            raise RuntimeError(
                f"Scene USD export did not package articulation {uid!r}."
            )
        objects.append(
            {
                "uid": uid,
                "kind": "articulation",
                "runtime_name": articulation._entities[0].get_name(),
                "source_asset": entry["fpath"],
                "proxy_asset": entry["proxy_glb_fpath"],
                "runtime_asset": runtime_asset.relative_to(output_root).as_posix(),
                "init_pos": _vector3(entry["init_pos"], f"{uid}.init_pos"),
                "init_rot": _vector3(entry["init_rot"], f"{uid}.init_rot"),
                "body_scale": _vector3(
                    entry.get("body_scale", [1.0, 1.0, 1.0]),
                    f"{uid}.body_scale",
                ),
                "fix_base": entry["fix_base"],
            }
        )
    manifest = {
        "format": _SCENE_USD_FORMAT,
        "scene_usd": scene_usd_path.relative_to(output_root).as_posix(),
        "source_scene_export": "scene_export/scene_config.json",
        "objects": objects,
    }
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
