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

"""Preview packaged materials and compare them with direct native loading.

Run ``python examples/sim/rendering/default_materials.py --list`` for offline
metadata. Use ``--all --compare-native --renderer rt`` for optical qualification,
and repeat with ``hybrid`` / ``fast-rt`` for the ten basic materials. Each
renderer must run in a separate process. Scene geometry, lighting and cameras
belong to this example; the material factory only loads material definitions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    """Build preview options before importing simulator dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--material", default="GlassClear")
    parser.add_argument("--renderer", choices=("rt", "hybrid", "fast-rt"), default="rt")
    parser.add_argument("--compare-native", action="store_true")
    parser.add_argument(
        "--model",
        type=Path,
        help="Optional segmented skin face GLB; not part of the material package.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/default-materials")
    )
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--spp", type=int, default=16)
    return parser


def main(args: argparse.Namespace) -> None:
    """Render one material or the renderer-compatible catalog with fixed settings."""
    from embodichain.data import get_default_material_info, list_default_materials

    if args.list:
        for name in list_default_materials():
            print(json.dumps(get_default_material_info(name)))
        return
    if min(args.size, args.frames, args.spp) < 1:
        raise ValueError("size, frames and spp must be positive")
    names = (
        [
            name
            for name in list_default_materials()
            if args.renderer in get_default_material_info(name)["renderers"]
        ]
        if args.all
        else [args.material]
    )
    # Validate a single incompatible request through production creation rather
    # than changing renderers or silently excluding it.
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
    rows: list[dict[str, Any]] = []
    try:
        sim.enable_physics(False)
        sim.set_emission_light([0.85, 0.9, 1.0], 12.0)
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
        for name, location, direction, color, power in (
            ("key", (-2, -2, 3), (2, 2, -2), (1, 0.85, 0.7), 60),
            ("fill", (2, 0, 3), (-2, 0, -2), (0.6, 0.8, 1), 40),
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
        source = DefaultMaterials() if args.compare_native else None
        for name in names:
            started = perf_counter()
            material = sim.create_visual_material(
                VisualMaterialCfg.from_preset(name, uid=f"preset_{name}")
            )
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
                cv2.imwrite(
                    str(args.output / f"{name}_{label}.png"),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
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
                "native_match": None,
            }
            if source:
                native = env.load_material(
                    source.get_material_path(name), f"reference_{name}"
                )
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
                ):
                    actual, expected = (
                        getattr(material.inst.mat, getter)(),
                        getattr(native.get_inst(), getter)(),
                    )
                    if expected is None or actual is None:
                        assert expected is actual, (name, getter, actual, expected)
                    else:
                        assert np.isclose(actual, expected, atol=1e-5), (
                            name,
                            getter,
                            actual,
                            expected,
                        )
                assert row["native_match"], row
            rows.append(row)
            print(json.dumps(row), flush=True)
        # Exercise every unsupported advanced preset before native loading in
        # real-time processes, including an advanced variation of a basic one.
        rejected = []
        if args.renderer != "rt":
            for name in list_default_materials():
                if args.renderer not in get_default_material_info(name)["renderers"]:
                    try:
                        sim.create_visual_material(VisualMaterialCfg.from_preset(name))
                    except ValueError:
                        rejected.append(name)
                    else:
                        raise AssertionError(f"Backend accepted advanced preset {name}")
            try:
                sim.create_visual_material(
                    VisualMaterialCfg.from_preset(
                        "BrickRed", parameters={"clearcoat": 0.0}
                    )
                )
            except ValueError:
                rejected.append("BrickRed:clearcoat")
            else:
                raise AssertionError("Backend accepted an advanced parameter variation")
        report = {
            "renderer": args.renderer,
            "dexsim": version("dexsim-engine"),
            "size": args.size,
            "frames": args.frames,
            "spp": args.spp,
            "materials": rows,
            "backend_rejections": rejected,
        }
        (args.output / "results.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        if camera is not None:
            camera.close_camera()
        sim.destroy(exit_process=False)
        del sample, inner, native, material, camera
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main(build_parser().parse_args())
