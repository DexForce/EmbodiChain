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
"""Recheck recorded Atomic initialization evidence and render its figures on CPU.

Run: python scripts/benchmark/motion_generation/initial_state_evidence.py
The hulls and articulation observations are native captures, not simulated by
this script. Current placement heights are read from the production YAML files.
No task execution or task-success measurement is performed here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import yaml

__all__: list[str] = []
EVIDENCE = Path(__file__).parent / "evidence/initial_state"
REPO = Path(__file__).resolve().parents[3]


def _world_hulls(capture: dict, pose: np.ndarray) -> list[np.ndarray]:
    """Apply native local and object transforms to already scaled vertices."""
    result = []
    for shape in capture["native_shapes"]:
        assert shape["shape_type"] == "CONVEX"
        local = np.asarray(shape["local_pose"], dtype=np.float64)
        vertices = np.asarray(shape["vertices"], dtype=np.float64)
        vertices = vertices @ local[:3, :3].T + local[:3, 3]
        result.append(vertices @ pose[:3, :3].T + pose[:3, 3])
    return result


def _overlap_count(a: list[np.ndarray], b: list[np.ndarray]) -> int:
    """Count positive-volume hull intersections using a 3D interior witness."""
    count = 0
    for first in a:
        for second in b:
            if np.any(first.max(0) < second.min(0)) or np.any(
                second.max(0) < first.min(0)
            ):
                continue
            planes = np.concatenate(
                [ConvexHull(first).equations, ConvexHull(second).equations]
            )
            answer = linprog(
                [0, 0, 0, -1],
                A_ub=np.column_stack(
                    [planes[:, :3], np.linalg.norm(planes[:, :3], axis=1)]
                ),
                b_ub=-planes[:, 3],
                bounds=[(None, None)] * 4,
                method="highs-ds",
                options={
                    "primal_feasibility_tolerance": 1e-9,
                    "dual_feasibility_tolerance": 1e-9,
                },
            )
            assert answer.success, answer.message
            count += answer.x[3] > 1e-7
    return int(count)


def _draw_hulls(ax: plt.Axes, hulls: list[np.ndarray], color: str) -> None:
    """Draw exact side projections, rather than bounding-box approximations."""
    for vertices in hulls:
        points = np.unique(vertices[:, [0, 2]], axis=0)
        if len(points) < 3:
            continue
        polygon = points[ConvexHull(points).vertices]
        ax.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=0.25, linewidth=0.6)
        closed = np.vstack([polygon, polygon[0]])
        ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=0.7)


def _geometry_figure(
    captures: dict, observations: dict, suite_root: Path, output: Path
) -> list[dict]:
    """Compare original birth geometry with the selected production recipes."""
    suites = suite_root / "extensions/embodichain_benchmarks/motion_generation/suites"
    recipes = {}
    for robot in ("ur5", "franka"):
        path = suites / f"atomic_dual_{robot}_pgi_coordinated_placement.yaml"
        suite = yaml.safe_load(path.read_text())
        recipes[robot] = {
            value["id"]: value for value in suite["tracks"][0]["config"]["objects"]
        }
    table = _world_hulls(captures["table"], np.asarray(captures["table"]["pose"]))
    table_top = max(vertices[:, 2].max() for vertices in table)
    fig, axes = plt.subplots(3, 2, figsize=(11, 9.6), sharey=True)
    results = []
    for row, (robot, name) in enumerate(
        [("ur5", "bread"), ("ur5", "pan"), ("franka", "pan")]
    ):
        key = f"{robot}_{name}"
        original = np.asarray(observations[f"{name}_pose_template"], dtype=np.float64)
        original[2, 3] = np.float32(observations["original_heights_m"][key])
        recipe = recipes[robot][name]
        np.testing.assert_allclose(recipe["position"][:2], original[:2, 3], atol=1e-6)
        np.testing.assert_allclose(
            Rotation.from_euler(
                "XYZ", recipe["rotation_deg"], degrees=True
            ).as_matrix(),
            original[:3, :3],
            atol=1e-6,
        )
        selected = original.copy()
        selected[2, 3] = np.float32(recipe["position"][2])
        objects = [_world_hulls(captures[name], pose) for pose in (original, selected)]
        vertices = np.concatenate(objects[1])
        xmin, xmax = vertices[:, 0].min() - 0.025, vertices[:, 0].max() + 0.025
        item = {"object": key}
        for col, (label, hulls) in enumerate(
            zip(("Original recipe", "Selected recipe"), objects)
        ):
            count = _overlap_count(hulls, table)
            gap = min(v[:, 2].min() for v in hulls) - table_top
            item[label] = {
                "intersecting_hull_pairs": count,
                "vertical_gap_m": float(gap),
            }
            ax = axes[row, col]
            _draw_hulls(ax, table, "#64748b")
            _draw_hulls(ax, hulls, "#d97706" if col == 0 else "#0284c7")
            ax.axhline(table_top, color="#475569", linestyle="--", linewidth=0.9)
            ax.set(
                xlim=(xmin, xmax),
                ylim=(0.61, 0.79),
                xlabel="World X (m)",
                ylabel="World Z (m)",
            )
            ax.set_title(
                f"{robot.upper()} {name} | {label}\n3D overlap pairs: {count}; Z gap: {1000 * gap:+.2f} mm",
                fontsize=10,
            )
            ax.grid(alpha=0.15)
        results.append(item)
    fig.suptitle("Recorded native collision hulls: placement birth state", fontsize=15)
    fig.text(
        0.5,
        0.012,
        "Side projections show geometry; overlap counts use full 3D convex halfspaces.\nSelected heights come from YAML. Positive Z gaps certify object/table separation only, before settling.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.955))
    fig.savefig(output / "placement_collision_hulls.png", dpi=170)
    plt.close(fig)
    return results


def _articulation_figure(observations: dict, output: Path) -> None:
    """Plot actual recorded coordinates with each joint's original unit."""
    values = observations["articulation"]["observations"]
    before, after = values["before"], values["after"]
    np.testing.assert_array_equal(
        before["settled_root_pose"], after["settled_root_pose"]
    )
    np.testing.assert_array_equal(
        after["actual_after_method_qpos"], after["expected_initial_qpos"]
    )
    np.testing.assert_array_equal(
        after["target_after_method_qpos"], after["expected_initial_qpos"]
    )
    assert not np.any(after["actual_after_method_qvel"])
    assert not np.any(after["target_after_method_qvel"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for joint, ax in enumerate(axes):
        factor = 1000 if joint == 0 else 1
        for side, color, marker in (
            ("before", "#d97706", "o"),
            ("after", "#0284c7", "s"),
        ):
            recorded = (
                np.asarray(values[side]["actual_after_method_qpos"])[:, joint] * factor
            )
            ax.plot(
                range(8),
                recorded,
                marker=marker,
                color=color,
                label=side,
                linewidth=1.7,
            )
        ax.axhline(0, color="#475569", linewidth=0.7, linestyle="--")
        ax.set(
            xlabel="Original environment row",
            ylabel="Button position (mm)" if joint == 0 else "Door position (rad)",
            xticks=range(8),
        )
        ax.set_title(
            "Button already pressed at initialization"
            if joint == 0
            else "Hinge drift during the same settling window"
        )
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle(
        "Native post-randomization state | B8, seed 11, object_roll_pitch", fontsize=13
    )
    fig.text(
        0.5,
        0.01,
        "Captured actual coordinates, not generated animation. Settled root poses are exactly equal; after restores actual and target joints.\nThese state observations do not establish a task-success improvement.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.92))
    fig.savefig(output / "articulation_initial_coordinates.png", dpi=170)
    plt.close(fig)


def main() -> None:
    """Regenerate the evidence figures and reject remaining object/table overlap."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, default=REPO)
    parser.add_argument("--output-dir", type=Path, default=EVIDENCE)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    captures = json.loads((EVIDENCE / "collision_geometry.json").read_text())
    observations = json.loads((EVIDENCE / "observations.json").read_text())
    geometry = _geometry_figure(
        captures, observations, args.suite_root, args.output_dir
    )
    _articulation_figure(observations, args.output_dir)
    print(
        json.dumps(
            {"geometry": geometry, "restored_actual_and_target_joints": True}, indent=2
        )
    )
    assert all(
        item["Original recipe"]["intersecting_hull_pairs"] > 0 for item in geometry
    )
    assert all(
        item["Selected recipe"]["intersecting_hull_pairs"] == 0 for item in geometry
    )
    assert all(item["Selected recipe"]["vertical_gap_m"] > 0 for item in geometry)


if __name__ == "__main__":
    main()
