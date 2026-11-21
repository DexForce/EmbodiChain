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

"""Build a tabletop, repair an invalid layout, and generate three seeded variants.

Run ``embodichain run-example expansion/scene_layout --output /tmp/scene-layout``.
The example composes primitive configurations, calls the shared XY/AABB solver,
generates addressed SceneRecipe candidates, applies them through scene services,
and checks native settled poses. It needs no model service or task environment.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.scene import LayoutCfg, SceneFragment, SceneRecipe

_PARTS = tuple(f"part_{letter}" for letter in "ABCDEF")
_COLORS = (
    (0.15, 0.40, 0.95),
    (0.02, 0.70, 0.45),
    (1.0, 0.65, 0.02),
    (0.55, 0.25, 0.95),
    (0.95, 0.20, 0.55),
    (0.0, 0.70, 0.85),
)
_LOWER, _UPPER = np.array([-0.86, -0.56]), np.array([0.86, 0.56])
_TOP_Z, _TOLERANCE = 0.78, 1e-4
_WIDTH, _HEIGHT = 800, 600


def _build() -> (
    tuple[list[SceneFragment], SceneRecipe, LayoutCfg, dict[str, np.ndarray]]
):
    from embodichain.lab.sim import VisualMaterialCfg
    from embodichain.lab.sim.cfg import RigidObjectCfg, VisualObjectCfg
    from embodichain.lab.sim.shapes import CubeCfg
    from embodichain.lab.sim.scene import (
        SceneFragment,
        SceneRecipe,
        LayoutProblem,
        LayoutCfg,
        PlanarRelation,
        compose,
    )

    def box(uid, size, position, color, *, dynamic=False):
        return RigidObjectCfg(
            uid=uid,
            shape=CubeCfg(
                size=size,
                visual_material=VisualMaterialCfg(
                    uid=f"material_{uid}",
                    base_color=[*color, 1.0],
                    roughness=0.7,
                    metallic=0.0,
                ),
            ),
            init_pos=position,
            body_type="dynamic" if dynamic else "static",
        )

    table = [
        box("work_surface", (1.8, 1.2, 0.06), (0.0, 0.0, 0.75), (0.78, 0.83, 0.88))
    ]
    table.extend(
        box(f"leg_{i}", (0.08, 0.08, 0.72), (x, y, 0.36), (0.18, 0.23, 0.30))
        for i, (x, y) in enumerate(
            ((-0.78, -0.48), (-0.78, 0.48), (0.78, -0.48), (0.78, 0.48))
        )
    )
    table_fragment = SceneFragment(table)

    shell = SceneFragment(
        [
            box("floor", (0.36, 0.30, 0.02), (0, 0, 0), (0.30, 0.38, 0.48)),
            *[
                box(name, size, xyz, (0.30, 0.38, 0.48))
                for name, size, xyz in (
                    ("left", (0.02, 0.30, 0.14), (-0.17, 0, 0.08)),
                    ("right", (0.02, 0.30, 0.14), (0.17, 0, 0.08)),
                    ("front", (0.32, 0.02, 0.14), (0, -0.14, 0.08)),
                    ("back", (0.32, 0.02, 0.14), (0, 0.14, 0.08)),
                )
            ],
        ]
    )
    trays = []
    for prefix, x in (("tray_left", -0.60), ("tray_right", 0.60)):
        transform = np.eye(4)
        transform[:3, 3] = (x, 0.28, 0.79)
        trays.append(compose([shell], namespace=prefix, arena_from_fragment=transform))
    zone = VisualObjectCfg(
        uid="keep_out",
        shape=CubeCfg(
            size=(0.42, 0.28, 0.001),
            visual_material=VisualMaterialCfg(
                uid="keep_out_paint", base_color=[0.95, 0.08, 0.06, 1.0], roughness=0.9
            ),
        ),
        init_pos=(0.0, 0.25, 0.781),
    )
    fixtures = compose([*trays, SceneFragment([zone])])
    positions = (
        (-0.05, -0.10),
        (0.02, -0.10),
        (0.08, -0.12),
        (-0.05, 0.20),
        (0.14, 0.22),
        (0.87, -0.40),
    )
    dimensions = (
        (0.12, 0.12, 0.12),
        (0.13, 0.10, 0.10),
        (0.14, 0.11, 0.10),
        (0.12, 0.10, 0.10),
        (0.11, 0.13, 0.12),
        (0.16, 0.10, 0.12),
    )
    parts = SceneFragment(
        [
            box(uid, size, (*xy, 0.88), color, dynamic=True)
            for uid, size, xy, color in zip(
                _PARTS, dimensions, positions, _COLORS, strict=True
            )
        ]
    )
    fragments = [table_fragment, fixtures, parts]
    scene = compose(fragments)
    seeds = dict(zip(_PARTS, positions, strict=True))
    fixed = {
        "tray_left__floor": (-0.60, 0.28),
        "tray_right__floor": (0.60, 0.28),
        "keep_out": (0.0, 0.25),
    }
    seeds.update(fixed)
    half = {
        uid: tuple(np.asarray(size[:2]) / 2)
        for uid, size in zip(_PARTS, dimensions, strict=True)
    }
    half.update(
        {
            "tray_left__floor": (0.18, 0.15),
            "tray_right__floor": (0.18, 0.15),
            "keep_out": (0.21, 0.14),
        }
    )
    problem = LayoutProblem(
        support_uid="work_surface",
        seed_xy_by_uid=seeds,
        half_extents_xy_by_uid=half,
        bounds_xy_by_uid={uid: (tuple(_LOWER), tuple(_UPPER)) for uid in _PARTS},
        fixed_xy_by_uid=fixed,
        relations=(
            PlanarRelation("part_A", "left_of", "part_B"),
            PlanarRelation("part_B", "left_of", "part_C"),
            PlanarRelation("part_D", "behind", "part_E"),
        ),
    )
    policy = LayoutCfg(
        collision_margin_m=0.04,
        relation_clearance_m=0.04,
        feasibility_tolerance=_TOLERANCE,
    )
    sizes = {
        entity.uid: np.asarray(entity.shape.size, dtype=float)
        for entity in scene.entities
    }
    return (
        fragments,
        SceneRecipe("tabletop-layout-example/v1", scene, problem),
        policy,
        sizes,
    )


def _observe(
    sim: SimulationManager,
    recipe: SceneRecipe,
    policy: LayoutCfg,
    sizes: dict[str, np.ndarray],
    row: int,
) -> dict[str, Any]:
    from embodichain.lab.sim.scene import SceneView, check_layout

    entities = recipe.fragment.entities
    view = SceneView(
        sim, env_ids=(row,), entity_uids=tuple(entity.uid for entity in entities)
    )
    poses = {
        entity.uid: view.pose(entity.uid)[0].detach().cpu().numpy()
        for entity in entities
    }
    boxes = {}
    for uid in recipe.layout.seed_xy_by_uid:
        pose = poses[uid]
        half = np.abs(pose[:3, :3]) @ (sizes[uid] / 2)
        boxes[uid] = np.stack((pose[:2, 3] - half[:2], pose[:2, 3] + half[:2]))
    checked = check_layout(recipe, poses, cfg=policy, actual_aabbs_xy_by_uid=boxes)

    def overlaps(first, second):
        return bool(
            np.all(
                np.minimum(first[1], second[1]) - np.maximum(first[0], second[0])
                > _TOLERANCE
            )
        )

    counts = {
        "overlapping_part_pairs": sum(
            overlaps(boxes[a], boxes[b])
            for i, a in enumerate(_PARTS)
            for b in _PARTS[i + 1 :]
        ),
        "parts_in_keep_out": sum(
            overlaps(boxes[uid], boxes["keep_out"]) for uid in _PARTS
        ),
        "parts_outside_bounds": sum(
            bool(
                np.any(boxes[uid][0] < _LOWER - _TOLERANCE)
                or np.any(boxes[uid][1] > _UPPER + _TOLERANCE)
            )
            for uid in _PARTS
        ),
    }
    fixed = [entity for entity in entities if entity.uid not in _PARTS]
    fixed_unchanged = all(
        np.allclose(poses[entity.uid], entity.init_local_pose, atol=1e-6, rtol=0)
        for entity in fixed
    )
    support_gap = max(
        abs(
            poses[uid][2, 3]
            - (np.abs(poses[uid][:3, :3]) @ (sizes[uid] / 2))[2]
            - _TOP_Z
        )
        for uid in _PARTS
    )
    return {
        "layout_valid": checked.passed,
        "counts": counts,
        "fixed_components_unchanged": fixed_unchanged,
        "max_support_height_error_m": float(support_gap),
        "part_xy_m": {uid: poses[uid][:2, 3].tolist() for uid in _PARTS},
        "checks": [
            {"name": c.name, "status": c.status, "metrics": dict(c.metrics)}
            for c in checked.checks
        ],
    }


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


class _Preview:
    def __init__(self, sim: SimulationManager, output: Path) -> None:
        from embodichain.lab.sim.sensors import CameraCfg

        self.sim, self.output = sim, output
        self.camera = sim.add_sensor(
            CameraCfg(
                uid="layout_camera",
                width=_WIDTH,
                height=_HEIGHT,
                intrinsics=(900.0, 900.0, _WIDTH / 2, _HEIGHT / 2),
                extrinsics=CameraCfg.ExtrinsicsCfg(
                    eye=(1.7, -2.0, 2.3), target=(0, 0, 0.78), up=(0, 0, 1)
                ),
            )
        )
        self.writer = imageio.get_writer(
            output / "preview.mp4", fps=20, codec="libx264", quality=8
        )
        self.images: dict[str, np.ndarray] = {}
        self.phases: list[dict[str, Any]] = []
        self.labels = ["authored scene"] * 4

    def capture(self, stage: str, *, hold: float = 0.0) -> None:
        clock = self.sim.simulation_time
        with self.sim.render_frame(capture_camera_images=False):
            self.camera.update()
            rgb = self.camera.get_data()["color"][..., :3].detach().cpu().numpy()
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb * (255 if rgb.max() <= 1 else 1), 0, 255)
            rgb = np.array(rgb, dtype=np.uint8, copy=True)
        if self.sim.simulation_time != clock:
            raise RuntimeError("Rendering advanced physics")
        image = Image.new("RGB", (2 * _WIDTH, 80 + 2 * (_HEIGHT + 36)), "#101826")
        draw = ImageDraw.Draw(image)
        draw.text(
            (18, 12), stage.upper().replace("_", " "), fill="white", font=_font(27)
        )
        draw.text(
            (18, 47),
            f"native RGB | physical time {clock:.3f} s | all geometry in metres",
            fill="#bdc8da",
            font=_font(17),
        )
        for row in range(4):
            x, y = row % 2 * _WIDTH, 80 + row // 2 * (_HEIGHT + 36)
            draw.text((x + 12, y + 6), self.labels[row], fill="#bdc8da", font=_font(18))
            image.paste(Image.fromarray(rgb[row]), (x, y + 36))
        for _ in range(max(1, round(hold * 20))):
            self.writer.append_data(np.asarray(image))
        if hold:
            self.images[stage] = rgb
            image.save(self.output / f"{stage}.png")
            self.phases.append({"stage": stage, "physical_time_s": clock})
            print(f"[scene-layout] {stage} at {clock:.3f} s", flush=True)

    def close(self) -> None:
        self.writer.close()

    def publish(self, report: dict[str, Any]) -> None:
        canvas = Image.new("RGB", (2 * _WIDTH, _HEIGHT + 135), "#101826")
        draw = ImageDraw.Draw(canvas)
        draw.text((18, 14), "BEFORE: CONFLICTED INPUT", fill="#ffaaa0", font=_font(25))
        draw.text(
            (_WIDTH + 18, 14),
            "AFTER: SOLVED + PHYSICALLY SETTLED",
            fill="#99e6bf",
            font=_font(24),
        )
        canvas.paste(Image.fromarray(self.images["constructed_input"][0]), (0, 57))
        canvas.paste(
            Image.fromarray(self.images["validated_variants"][0]), (_WIDTH, 57)
        )
        for col, data in enumerate((report["input"], report["final_layouts"][0])):
            counts = data["counts"]
            draw.text(
                (col * _WIDTH + 18, _HEIGHT + 72),
                f"overlap pairs: {counts['overlapping_part_pairs']}  |  keep-out violations: {counts['parts_in_keep_out']}  |  outside bounds: {counts['parts_outside_bounds']}",
                fill="white",
                font=_font(17),
            )
            draw.text(
                (col * _WIDTH + 18, _HEIGHT + 104),
                f"shared layout check: {'PASS' if data['layout_valid'] else 'FAIL'}",
                fill="#99e6bf" if data["layout_valid"] else "#ffaaa0",
                font=_font(18),
            )
        canvas.save(self.output / "comparison.png")
        (self.output / "index.html").write_text(
            """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>场景构建、布局求解与扩增</title>
<style>body{max-width:1600px;margin:32px auto;padding:0 20px;background:#101826;color:#e5e7eb;font:16px/1.7 system-ui}img,video{width:100%;border-radius:8px}video{max-height:75vh}a{color:#93c5fd}</style>
<h1>场景构建 → 约束布局求解 → 三种布局扩增</h1>
<p>依次构建桌面、两个容器、禁放区和六个彩色部件。初始布局有重叠、禁区侵入与越界。
共享求解器修复后，使用同一套资产和约束生成三个带种子的布局变体。</p>
<img src="comparison.png" alt="原生仿真中的布局求解前后对照">
<video controls preload="metadata" poster="validated_variants.png" src="preview.mp4"></video>
<p>红色区域是无碰撞体的禁放标记，两个灰色容器固定。几何求解和布局写入不推进物理；
视频按阶段停留，布局切换不伪装成物体运动。所有有效布局最后经过真实重力沉降与实测检查。</p>
<img src="validated_variants.png" alt="修复布局及三种扩增布局的原生渲染">
<p><a href="report.json">实测报告与布局坐标</a> · <a href="plans.json">输入约束、求解结果与候选身份</a></p></html>""",
            encoding="utf-8",
        )


def _run(output: Path, seed: int) -> dict[str, Any]:
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    if "XDG_RUNTIME_DIR" not in os.environ:
        os.environ["XDG_RUNTIME_DIR"] = tempfile.mkdtemp(prefix="embodichain-xdg-")
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import DefaultPhysicsCfg, RenderCfg, LightCfg
    from embodichain.lab.visualization import VisualizationCfg
    from embodichain.lab.sim.scene import (
        SceneView,
        SceneChangeSet,
        SettlingCfg,
        declare_fragment,
        solve_layout,
        generate,
        apply_changes,
        settle_scene,
    )
    from embodichain.lab.sim.scene.expansion import apply_scene_variants

    fragments, recipe, policy, sizes = _build()
    solution = solve_layout(recipe.layout, policy)
    if solution.status != "solved":
        raise RuntimeError(solution.message)
    candidates = [generate(recipe, seed=seed, ordinal=i, cfg=policy) for i in range(3)]
    if any(candidate.variant is None for candidate in candidates):
        raise RuntimeError("An augmentation candidate failed its layout solve")
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device="cpu",
            num_envs=4,
            physics_cfg=DefaultPhysicsCfg(),
            render_cfg=RenderCfg(renderer="hybrid"),
            visualization=VisualizationCfg(backend="none"),
        )
    )
    preview = None
    report: dict[str, Any] = {"seed": seed, "passed": False}
    try:
        sim.add_light(
            LightCfg(
                uid="sun", light_type="sun", direction=(0.3, 0.1, -1), intensity=5.0
            )
        )
        declare_fragment(sim, fragments[0])
        sim.prepare()
        preview = _Preview(sim, output)
        preview.capture("construction_table", hold=0.8)
        declare_fragment(sim, fragments[1])
        sim.prepare()
        preview.capture("construction_fixtures", hold=0.8)
        declare_fragment(sim, fragments[2])
        sim.prepare()
        initial = _observe(sim, recipe, policy, sizes, 0)
        report["input"] = initial
        preview.capture("constructed_input", hold=1.5)
        clock = sim.simulation_time
        changes = {}
        for uid in _PARTS:
            pose = (
                sim.get_rigid_object(uid)
                .get_local_pose(to_matrix=True)[0]
                .detach()
                .cpu()
                .numpy()
                .copy()
            )
            pose[:2, 3] = solution.xy_by_uid[uid]
            changes[uid] = pose.tolist()
        apply_changes(
            SceneView(sim, env_ids=(0, 1, 2, 3)), SceneChangeSet(changes, _PARTS)
        )
        repaired = _observe(sim, recipe, policy, sizes, 0)
        preview.labels = ["constraint-solved authored input"] * 4
        preview.capture("constraint_solved", hold=1.5)
        apply_scene_variants(
            sim,
            [candidate.variant for candidate in candidates],
            env_ids=(1, 2, 3),
            movable_entity_ids=_PARTS,
        )
        preview.labels = [
            "REPAIRED ORIGINAL",
            *[f"AUGMENTATION {i + 1} | seed {seed}, ordinal {i}" for i in range(3)],
        ]
        preview.capture("generated_variants", hold=1.5)
        report["layout_operations_did_not_step"] = sim.simulation_time == clock
        steps = 0

        def step(count: int) -> None:
            nonlocal steps
            sim.update(step=count)
            steps += count
            if steps % 5 == 0:
                preview.capture("gravity_settling")

        stability = settle_scene(
            SceneView(sim, env_ids=(0, 1, 2, 3)),
            step,
            SettlingCfg(mode="convergence", max_steps=600, consecutive_steps=10),
        )
        final = [_observe(sim, recipe, policy, sizes, row) for row in range(4)]
        preview.capture("validated_variants", hold=2.0)
        ids = [candidate.variant.variant_id for candidate in candidates]
        base_xy = np.array([final[0]["part_xy_m"][uid] for uid in _PARTS])
        differences = [
            float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            (
                                np.array([item["part_xy_m"][uid] for uid in _PARTS])
                                - base_xy
                            )
                            ** 2,
                            axis=1,
                        )
                    )
                )
            )
            for item in final[1:]
        ]
        report.update(
            entity_count=sum(len(fragment.entities) for fragment in fragments),
            repaired=repaired,
            final_layouts=final,
            augmentation_variant_ids=ids,
            augmentation_rms_displacement_m=differences,
            physics_steps=steps,
            stable=stability.passed,
            stability_checks=[
                {"name": c.name, "status": c.status, "metrics": dict(c.metrics)}
                for c in stability.checks
            ],
            phases=preview.phases,
        )
        report["passed"] = bool(
            not initial["layout_valid"]
            and repaired["layout_valid"]
            and all(
                item["layout_valid"]
                and item["fixed_components_unchanged"]
                and item["max_support_height_error_m"] < 0.005
                for item in final
            )
            and stability.passed
            and report["layout_operations_did_not_step"]
            and len(set(ids)) == 3
            and min(differences) > 0.1
        )
        plans = {
            "layout_input": asdict(recipe.layout),
            "solver_result": asdict(solution),
            "margin_m": policy.collision_margin_m,
            "geometry_tolerance_m": policy.feasibility_tolerance,
            "variants": [
                {
                    "variant_id": candidate.variant.variant_id,
                    **asdict(candidate.variant),
                }
                for candidate in candidates
            ],
        }
        (output / "plans.json").write_text(json.dumps(plans, indent=2) + "\n")
        preview.close()
        preview.publish(report)
        if not report["passed"]:
            raise RuntimeError("Native layout checks failed; inspect report.json")
        return report
    finally:
        if preview is not None:
            preview.close()
        (output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


def main() -> None:
    """Run the native construction/layout comparison in an empty output folder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/scene_layout"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("output must be empty; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    report = _run(output, args.seed)
    print(json.dumps({"passed": report["passed"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
