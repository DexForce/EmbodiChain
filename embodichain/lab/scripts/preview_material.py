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

"""Preview default material presets in a native window or export reference images.

Help and catalog queries do not initialize DexSim or download material assets.
Scene geometry, lighting and cameras belong to this command; the material factory
only loads material definitions.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
import traceback
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from embodichain.lab.sim.material import VisualMaterialCfg

__all__ = ["main", "cli"]


def _positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _material_params(value: str) -> dict[str, Any]:
    try:
        params = json.loads(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--material-params must be a JSON object"
        ) from error
    if not isinstance(params, dict):
        raise argparse.ArgumentTypeError("--material-params must be a JSON object")
    if {"uid", "preset"} & params.keys():
        raise argparse.ArgumentTypeError(
            "Select the preset with --material; preview UIDs are command-owned"
        )
    return params


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="embodichain preview-material",
        description=__doc__,
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--list",
        action="store_true",
        help="Print offline catalog metadata as JSON lines and exit.",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="Export all selected presets without opening a window.",
    )
    selection.add_argument(
        "--material", help="Exact preset name; defaults to GlassClear."
    )
    parser.add_argument(
        "--source",
        choices=("all", "builtin", "resource"),
        default="all",
        help="Filter --list or --all by material source.",
    )
    parser.add_argument(
        "--renderer",
        choices=("rt", "hybrid", "fast-rt"),
        default="rt",
        help="Native renderer; rt (OfflineRT) supports the full catalog.",
    )
    parser.add_argument(
        "--material-params",
        type=_material_params,
        default={},
        help="VisualMaterialCfg overrides as JSON; advanced fields belong inside parameters.",
    )
    parser.add_argument(
        "--compatible-only",
        action="store_true",
        help="With --all, explicitly skip incompatible presets and report their names.",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Export images and exit without opening a native window.",
    )
    parser.add_argument(
        "--max-steps",
        type=_positive_integer,
        help="Bound native-window preview updates; otherwise exit with Ctrl+C.",
    )
    parser.add_argument(
        "--gpu-id",
        "--gpu_id",
        type=int,
        default=0,
        help="GPU index used by DexSim rendering.",
    )
    parser.add_argument(
        "--compare-native",
        action="store_true",
        help="Compare unmodified presets with direct native loading and fail on a mismatch.",
    )
    parser.add_argument(
        "--gallery",
        action="store_true",
        help="Export individual WebP images and grouped galleries (Pillow and matplotlib required).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Optional segmented skin face GLB; not part of the material package.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/default-materials"),
        help="Directory for PNG captures, results.json and optional WebP galleries.",
    )
    parser.add_argument(
        "--size",
        type=_positive_integer,
        default=320,
        help="Square capture resolution in pixels.",
    )
    parser.add_argument(
        "--frames",
        type=_positive_integer,
        default=4,
        help="Accumulation frames per capture.",
    )
    parser.add_argument(
        "--spp",
        type=_positive_integer,
        default=16,
        help="DexSim render iterations per frame.",
    )
    return parser


def _select_materials(
    args: argparse.Namespace,
) -> tuple[list[VisualMaterialCfg], list[dict[str, str]]]:
    """Resolve configurations and renderer requirements before constructing a World."""
    from embodichain._material_catalog import (
        get_default_material_info,
        list_default_materials,
    )
    from embodichain.lab.sim import VisualMaterialCfg
    from embodichain.lab.sim._material_presets import (
        _asset_desc_from_cfg,
        _validate_backend,
    )

    if min(args.size, args.frames, args.spp) < 1:
        raise ValueError("size, frames and spp must be positive")
    if args.gpu_id < 0:
        raise ValueError("gpu-id must be nonnegative")
    if args.compatible_only and not args.all:
        raise ValueError("--compatible-only requires --all")
    if args.source != "all" and not args.all:
        raise ValueError("--source requires --list or --all")
    if args.max_steps is not None and (args.headless or args.all):
        raise ValueError(
            "--max-steps applies to native-window previews; remove --headless/--all"
        )
    if args.compare_native and args.material_params:
        raise ValueError(
            "--compare-native compares authored presets; remove --material-params"
        )
    names = (
        [
            name
            for name in list_default_materials()
            if args.source == "all"
            or get_default_material_info(name)["source"] == args.source
        ]
        if args.all
        else [args.material or "GlassClear"]
    )
    if args.model is not None:
        args.model = args.model.expanduser().resolve()
        if "SkinHumanLight" not in names:
            raise ValueError("--model requires SkinHumanLight or a batch containing it")
        if not args.model.is_file():
            raise FileNotFoundError(args.model)
    configs, skipped = [], []
    for name in names:
        cfg = VisualMaterialCfg.from_preset(
            name, uid=f"preset_{name}", **args.material_params
        )
        desc = _asset_desc_from_cfg(cfg, cfg.uid)
        try:
            _validate_backend(desc, args.renderer)
        except ValueError as error:
            if not args.compatible_only:
                raise ValueError(
                    f"{error} Use --all --compatible-only to explicitly select supported presets."
                ) from error
            skipped.append({"name": name, "reason": str(error)})
        else:
            configs.append(cfg)
    if not configs:
        raise ValueError(
            "No selected presets support this renderer and parameter combination"
        )
    return configs, skipped


def _write_galleries(output: Path, rows: list[dict[str, Any]], renderer: str) -> None:
    """Export captured images without changing their rendering settings.

    Args:
        output: Directory containing the preset PNG captures.
        rows: Material records from the completed rendering run.
        renderer: Renderer name shown in gallery captions.
    """
    import math
    from importlib.metadata import version
    import matplotlib
    from PIL import Image
    from embodichain.data import get_default_material_info

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    for row in rows:
        with Image.open(output / f"{row['name']}_preset.png") as capture:
            capture.save(output / f"{row['name']}.webp", quality=92)
    groups = {
        "default-materials": (rows, 6),
        "builtin-materials": ([row for row in rows if row["source"] == "builtin"], 3),
        "surface-materials": (
            [
                row
                for row in rows
                if row["source"] == "resource"
                and len(get_default_material_info(row["name"])["renderers"]) > 1
            ],
            5,
        ),
        "advanced-materials": (
            [
                row
                for row in rows
                if row["source"] == "resource"
                and get_default_material_info(row["name"])["renderers"] == ["rt"]
            ],
            4,
        ),
    }
    for filename, (records, columns) in groups.items():
        if not records:
            continue
        columns = min(columns, len(records))
        count_rows = math.ceil(len(records) / columns)
        fig, axes = plt.subplots(
            count_rows,
            columns,
            figsize=(3.2 * columns, 3.5 * count_rows),
            squeeze=False,
            facecolor="#171b20",
        )
        for axis in axes.flat:
            axis.axis("off")
        for axis, row in zip(axes.flat, records):
            with Image.open(output / f"{row['name']}_preset.png") as capture:
                axis.imshow(capture.copy())
            axis.set_title(row["name"], fontsize=11, color="white", pad=6)
        renderer_label = {"rt": "OfflineRT", "hybrid": "Hybrid", "fast-rt": "FastRT"}[
            renderer
        ]
        fig.suptitle(
            f"Default materials · DexSim {version('dexsim-engine')} · {renderer_label}",
            color="white",
            fontsize=15,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(
            output / f"{filename}.webp",
            dpi=160,
            facecolor=fig.get_facecolor(),
            pil_kwargs={"quality": 90},
        )
        plt.close(fig)


def _render_materials(
    args: argparse.Namespace,
    configs: list[VisualMaterialCfg],
    skipped: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Capture one fixed studio scene, optionally retaining its native viewer."""
    from embodichain.data import get_default_material_info
    import cv2
    import numpy as np
    import dexsim.render as render
    from importlib.metadata import version
    from embodichain.lab.sim import (
        SimulationManager,
        SimulationManagerCfg,
        VisualMaterialCfg,
    )
    from embodichain.lab.sim.cfg import DenoisingCfg, RenderCfg
    from embodichain.data.assets.materials import DefaultMaterials

    args.output.mkdir(parents=True, exist_ok=True)
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            gpu_id=args.gpu_id,
            device="cpu",
            scene_node_capacity=256,
            startup_summary="off",
            render_cfg=RenderCfg(
                renderer=args.renderer,
                spp=args.spp,
                min_bounces=3,
                max_bounces=14,
                tone_mapping_enabled=True,
                tone_mapping_exposure=0.9,
                denoising=DenoisingCfg(window="optix", offscreen="optix"),
            ),
        )
    )
    world, env = sim.get_world(), sim.get_env()
    camera = sample = inner = native = material = None
    reference_files = None
    rows: list[dict[str, Any]] = []
    try:
        sim.enable_physics(False)
        sim.set_emission_light([0.85, 0.9, 1.0], 24.0)
        # Independent, fixed scene: colored cards reveal transmission, broad
        # rectangular lights reveal roughness and normal-map direction.
        for index, (x, color) in enumerate(
            (
                (-0.85, [0.8, 0.06, 0.02, 1]),
                (0.0, [0.8, 0.8, 0.8, 1]),
                (0.85, [0.02, 0.18, 0.8, 1]),
            )
        ):
            card = env.create_cube(0.7, 0.04, 1.9)
            card.set_name(f"preview_card_{index}")
            card.set_location(x, 1.1, 0.95)
            card_material = sim.create_visual_material(
                VisualMaterialCfg(uid=f"card_{index}", base_color=color, roughness=0.6)
            )
            card.set_material(card_material.inst.mat)
            if index == 1:
                yy, xx = np.indices((256, 256))
                checks = ((xx // 32 + yy // 32) % 2)[..., None]
                pixels = np.where(checks, [225, 225, 225], [45, 45, 45]).astype(
                    np.uint8
                )
                texture = env.create_color_texture(pixels, has_alpha=False)
                card_material.inst.set_base_color_texture(texture_obj=texture)
        for name, location, direction, color, power in (
            ("key", (-2, -2, 3), (2, 2, -2), (1, 0.85, 0.7), 90),
            ("fill", (2, 0, 3), (-2, 0, -2), (0.6, 0.8, 1), 60),
        ):
            light = env.create_light(name, render.LightType.RECT)
            light.set_location(*location)
            light.set_direction(*direction)
            light.set_color(*color)
            light.set_intensity(power)
            light.set_rect_wh(1.8, 1.1)
        camera = env.create_camera(
            "material_camera", args.size, args.size, True, render.ViewFlags.COLOR
        )
        camera.set_near(0.02)
        camera.set_far(20)
        focal = args.size * 1.4
        camera.set_intrinsic(
            np.array(
                [[focal, 0, args.size / 2], [0, focal, args.size / 2], [0, 0, 1]],
                dtype=np.float32,
            )
        )
        camera.look_at([1.5, -3.8, 1.8], [0, 0, 0.8], [0, 0, 1])
        camera.open_camera()
        source = None
        reference_files = (
            TemporaryDirectory(prefix="ec-material-reference-")
            if args.compare_native
            else None
        )
        for cfg in configs:
            name = cfg.preset
            info = get_default_material_info(name)
            started = perf_counter()
            material = sim.create_visual_material(cfg)
            load_ms = (perf_counter() - started) * 1000
            if sample is not None:
                env.remove_actor(sample)
            if inner is not None:
                env.remove_actor(inner)
                inner = None
            skin_segments = None
            if name == "SkinHumanLight" and args.model:
                if not args.model.is_file():
                    raise FileNotFoundError(args.model)
                sample = env.load_actor(str(args.model.resolve()), False, False)
                env.attach(sample.node)
                world.update(0.0)
                body = sample.get_render_body()
                skin_segments = [
                    i
                    for i in range(body.get_mesh_count())
                    if "skin" in body.get_segment_name(i).lower()
                ]
                if (
                    not skin_segments
                    and body.get_mesh_count() == 8
                    and all(not body.get_segment_name(i) for i in range(8))
                ):
                    skin_segments = [0, 1]  # Explicit SkinFaceDemo GLB-order fallback.
                if not skin_segments:
                    raise ValueError("The supplied face model has no Skin_* segments.")
                camera.look_at([0.19, -0.69, 0.165], [0, 0, 0.10], [0, 0, 1])
            else:
                category = get_default_material_info(name)["category"]
                sample = (
                    env.create_cube(1.3, 0.16, 1.3)
                    if category in {"architecture", "wood", "stone", "ground"}
                    else env.create_sphere(0.66, 96, 3, 2)
                )
                sample.set_location(0, 0, 0.8)
                sample.set_name(f"preview_{name}")
                camera.look_at([1.5, -3.8, 1.8], [0, 0, 0.8], [0, 0, 1])
            if name == "SoapBubble":
                # Geometry stays here: a smaller reversed-axis shell supplies
                # the air-facing interface. UVs and nominal sizes are untouched.
                inner = env.create_sphere(0.66, 96, 3, 2)
                inner.set_name("preview_inner_shell")
                inner.set_location(0, 0, 0.8)
                inner.set_scale(-0.975, 0.975, 0.975)

            def bind(instance: Any, inner_instance: Any | None = None) -> None:
                if skin_segments is None:
                    sample.set_material(instance)
                else:
                    for segment in skin_segments:
                        sample.get_render_body().set_mesh_material(segment, instance)
                if inner is not None:
                    inner.set_material(inner_instance)

            def capture(label: str) -> Any:
                for _ in range(args.frames):
                    world.update(0.0)
                    camera.render(render.ViewFlags.COLOR, render.ViewFlags.COLOR, False)
                rgb = np.ascontiguousarray(camera.get_rgb_map()[..., :3])
                saved = cv2.imwrite(
                    str(args.output / f"{name}_{label}.png"),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                )
                if not saved:
                    raise RuntimeError(
                        f"Cannot save {name}_{label}.png to {args.output}."
                    )
                return rgb

            inner_material = (
                sim.create_visual_material(
                    VisualMaterialCfg.from_preset(
                        "SoapBubbleInterior", uid="bubble_inner"
                    )
                )
                if inner is not None
                else None
            )
            bind(material.inst.mat, inner_material.inst.mat if inner_material else None)
            packaged = capture("preset")
            row: dict[str, Any] = {
                "name": name,
                "load_ms": load_ms,
                "ior": material.inst.ior,
                "roughness": material.inst.roughness,
                "metallic": material.inst.mat.get_metallic(),
                "transmission": material.inst.mat.get_transmission(),
                "base_color": material.inst.base_color,
                "native_match": None,
                "source": info["source"],
                "revision": info["revision"],
                "config": cfg.to_dict(),
                "geometry": (
                    "segmented-face"
                    if skin_segments is not None
                    else (
                        "double-shell"
                        if inner is not None
                        else (
                            "slab"
                            if info["category"]
                            in {"architecture", "wood", "stone", "ground"}
                            else "sphere"
                        )
                    )
                ),
            }
            if args.compare_native:
                if info["source"] == "builtin":
                    reference_path = Path(reference_files.name) / name
                    reference_path.mkdir()
                    # Direct native baseline uses the authored definition,
                    # without the runtime resolver or Python default setters.
                    (reference_path / "material.json").write_text(
                        json.dumps(
                            {
                                "schema_version": 1,
                                "parameters": info["parameters"],
                                "textures": {},
                            }
                        ),
                        encoding="utf-8",
                    )
                else:
                    if source is None:
                        source = DefaultMaterials()
                    reference_path = Path(source.get_material_path(name))
                native = env.load_material(str(reference_path), f"reference_{name}")
                native_inner = (
                    env.load_material(
                        source.get_material_path("SoapBubbleInterior"),
                        "reference_inner",
                    )
                    if inner is not None
                    else None
                )
                bind(
                    native.get_inst(), native_inner.get_inst() if native_inner else None
                )
                reference = capture("native")
                repeat = capture("native_repeat")
                crop = np.s_[
                    args.size // 4 : 3 * args.size // 4,
                    args.size // 4 : 3 * args.size // 4,
                    :,
                ]
                difference = float(
                    np.sqrt(
                        np.mean(
                            (
                                packaged[crop].astype(float)
                                - reference[crop].astype(float)
                            )
                            ** 2
                        )
                    )
                )
                noise = float(
                    np.sqrt(
                        np.mean(
                            (repeat[crop].astype(float) - reference[crop].astype(float))
                            ** 2
                        )
                    )
                )
                row.update(
                    rmse=difference,
                    native_repeat_rmse=noise,
                    threshold=max(3.0, 3 * noise + 0.5),
                    native_match=difference <= max(3.0, 3 * noise + 0.5),
                )
                for getter in (
                    "get_ior",
                    "get_roughness",
                    "get_metallic",
                    "get_transmission",
                    "get_base_color",
                ):
                    actual, expected = (
                        getattr(material.inst.mat, getter)(),
                        getattr(native.get_inst(), getter)(),
                    )
                    matches = (
                        expected is actual
                        if expected is None or actual is None
                        else np.allclose(actual, expected, atol=1e-5)
                    )
                    if not matches:
                        raise AssertionError((name, getter, actual, expected))
                if not row["native_match"]:
                    raise AssertionError(row)
            bind(material.inst.mat, inner_material.inst.mat if inner_material else None)
            rows.append(row)
            print(json.dumps(row), flush=True)
        report = {
            "schema_version": 1,
            "scene": "material-studio-v1",
            "renderer": args.renderer,
            "dexsim": version("dexsim-engine"),
            "size": args.size,
            "frames": args.frames,
            "spp": args.spp,
            "materials": rows,
            "skipped_materials": skipped,
            "model": str(args.model) if args.model is not None else None,
        }
        (args.output / "results.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        if not args.headless and not args.all:
            from embodichain.lab.scripts._preview import _run_preview_loop

            sim.prepare()
            if not sim.open_window():
                raise RuntimeError(
                    "The native material-preview window could not be opened"
                )
            eye, target = (
                ([0.19, -0.69, 0.165], [0, 0, 0.10])
                if args.model is not None
                else ([1.5, -3.8, 1.8], [0, 0, 0.8])
            )
            world.get_windows().set_look_at(eye, target, [0, 0, 1])
            print(
                "Material preview ready. Press Ctrl+C to exit.",
                file=sys.stderr,
                flush=True,
            )
            _run_preview_loop(sim, None, interactive=True, max_steps=args.max_steps)
    finally:
        try:
            if camera is not None:
                camera.close_camera()
        finally:
            try:
                sim.destroy(exit_process=False)
            finally:
                if reference_files is not None:
                    reference_files.cleanup()
    return rows


def main(args: argparse.Namespace) -> None:
    """Query or preview default materials using the production preset resolver.

    Args:
        args: Parsed material-preview arguments. Batch mode always exports
            without a window; single previews use one unless headless is set.

    Raises:
        ValueError: Selection, override or renderer requirements are invalid.
        OSError: A supplied model or output directory cannot be accessed.
        RuntimeError: Native rendering or window creation fails.
        AssertionError: A requested direct-native comparison fails.
    """
    from embodichain._material_catalog import (
        get_default_material_info,
        list_default_materials,
    )

    if args.list:
        for name in list_default_materials():
            info = get_default_material_info(name)
            if args.source == "all" or info["source"] == args.source:
                print(json.dumps(info))
        return
    configs, skipped = _select_materials(args)
    for item in skipped:
        print(f"Skipping {item['name']}: {item['reason']}", file=sys.stderr)
    from embodichain.lab.sim import SimulationManager

    failure: BaseException | None = None
    try:
        rows = _render_materials(args, configs, skipped)
    except BaseException as error:
        # Native views in unwound frames must be released before deferred World
        # destruction. Preserve the original error while clearing its views.
        pending = [error]
        seen: set[int] = set()
        while pending:
            current = pending.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            traceback.clear_frames(current.__traceback__)
            pending.extend(
                cause
                for cause in (current.__cause__, current.__context__)
                if cause is not None
            )
        failure = error.with_traceback(None)
    try:
        SimulationManager.flush_cleanup_queue()
    except BaseException as error:
        if failure is None:
            raise
        print(
            f"Deferred preview cleanup failed after {failure!r}: {error!r}",
            file=sys.stderr,
        )
    if failure is not None:
        raise failure
    if args.gallery:
        _write_galleries(args.output, rows, args.renderer)


def cli(argv: Sequence[str] | None = None) -> None:
    """Run the material-preview command.

    Args:
        argv: Arguments excluding the command name, or None for sys.argv.
    """
    parser = _create_parser()
    args = parser.parse_args(argv)
    try:
        main(args)
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    cli()
