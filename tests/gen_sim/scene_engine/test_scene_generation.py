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

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image, ImageDraw
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation

from embodichain.gen_sim.scene_engine.core.scene_graph import (
    SceneGraph,
    SceneGraphNode,
)
from embodichain.gen_sim.scene_engine.core.scene import Scene, SceneObject
from embodichain.gen_sim.scene_engine.pipeline.generation.scene_generation import (
    _align_table_roots_individually,
    _generate_articulated_usdcs,
    _generate_coarse_results_from_masks,
    _apply_visual_yaws_to_simready_asset_layouts,
    _apply_root_layout_updates_to_descendant_subtrees,
    _lift_table_subtrees,
    _optimize_simready_asset_visual_yaws,
    _project_child_aabb_centers_into_parent_aabb,
    _refine_on_children_bfs,
    _table_on_asset_ids,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.visual_yaw_optimizer import (
    VisualYawOptimizer,
)
from embodichain.gen_sim.asset_engine.utils.articulated_usdc_utils import (
    _canonicalize_articulated_usdc_bottom_center,
    _articulation_root_bottom_z,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_generation_utils import (
    layout_object_to_transform_matrix,
    transform_matrix_to_layout_object,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_layout_utils import (
    measure_scene_object_z_up_world_aabb,
    rotate_scene_z_up_world,
    scene_object_y_up_layout,
    y_up_to_z_up_matrix,
)
from embodichain.gen_sim.services.contracts import (
    GeometryRequest,
    GeometryResult,
    ReconstructedObject,
)


def test_coarse_layout_records_replayable_source_and_normalized_pose(
    tmp_path: Path,
) -> None:
    coarse = tmp_path / "coarse_geometry"
    coarse.mkdir()
    image = tmp_path / "image.png"
    image.touch()
    mask = tmp_path / "mask.png"
    mask.touch()
    quarter_turn = np.sqrt(0.5)
    mesh = coarse / "table.glb"
    mesh.touch()

    class Geometry:
        def reconstruct(self, request: GeometryRequest) -> GeometryResult:
            assert request.object_masks == (("table", mask),)
            return GeometryResult(
                (
                    ReconstructedObject(
                        "table",
                        mesh,
                        (1.0, 2.0, 3.0),
                        (0.0, 0.0, quarter_turn, quarter_turn),
                        (2.0, 3.0, 4.0),
                        source_quaternion_wxyz=(quarter_turn, 0, 0, -quarter_turn),
                        pose_coordinate_system="glb_y_up",
                        pose_convention="sam3d_row_conjugated",
                        pose_convention_source="configuration",
                    ),
                )
            )

    _generate_coarse_results_from_masks(
        image,
        tmp_path,
        coarse,
        Scene(
            [
                SceneObject(
                    id="table",
                    kind="table",
                    category="table",
                    name="table",
                    description="Tabletop",
                    mask_path=str(mask),
                )
            ]
        ),
        geometry_generation_client=Geometry(),
    )

    (row,) = json.loads((coarse / "coarse_layout.json").read_text())
    expected = np.array([[0, -3, 0, 1], [2, 0, 0, 2], [0, 0, 4, 3], [0, 0, 0, 1]])
    np.testing.assert_allclose(row["matrix"], expected, atol=1e-12)
    np.testing.assert_allclose(
        layout_object_to_transform_matrix(row), expected, atol=1e-12
    )
    assert row["source"]["rotation_quaternion_wxyz"] == pytest.approx(
        [quarter_turn, 0, 0, -quarter_turn]
    )
    assert row["source"]["pose_convention"] == "sam3d_row_conjugated"
    assert row["source"]["pose_convention_source"] == "configuration"
    assert coarse / row["source"]["geometry_path"] == mesh


@pytest.mark.parametrize("up_axis, expected", [("Y", -0.2), ("Z", -0.1)])
def test_root_height_uses_collision_geometry_not_usd_root_translation(
    tmp_path: Path, up_axis: str, expected: float
) -> None:
    path = tmp_path / "height.usdc"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(stage, up_axis)
    root.AddTranslateOp().Set(Gf.Vec3d(4, 5, 6))
    body = UsdGeom.Xform.Define(stage, "/Asset/base")
    body.AddTranslateOp().Set(Gf.Vec3d(0, 0, 2))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/base/collision")
    mesh.CreatePointsAttr([(-1, -0.1, -0.05), (1, 0.1, 0.05)])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    stage.GetRootLayer().Save()
    assert _articulation_root_bottom_z(path, [2, 2, 2], [0, 0, 35]) == pytest.approx(
        expected
    )
    root.GetOrderedXformOps()[0].Set(Gf.Vec3d(9, 10, 11))
    stage.GetRootLayer().Save()
    assert _articulation_root_bottom_z(path, [2, 2, 2], [0, 0, 35]) == pytest.approx(
        expected
    )


def test_z_up_canonicalization_uses_z_bottom(tmp_path: Path) -> None:
    path = tmp_path / "z_up.usdc"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    cube = UsdGeom.Cube.Define(stage, "/Asset/mesh")
    cube.CreateSizeAttr(2)
    cube.AddTranslateOp().Set(Gf.Vec3d(2, 3, 4))
    stage.GetRootLayer().Save()
    _canonicalize_articulated_usdc_bottom_center(path)
    stage = Usd.Stage.Open(str(path))
    translation = (
        UsdGeom.Xformable(stage.GetDefaultPrim()).GetOrderedXformOps()[0].Get()
    )
    assert list(translation) == pytest.approx([-2, -3, -3])


def _z_up_rotation_from_y_up_layout(layout: dict[str, object]) -> np.ndarray:
    y_up_to_z_up_matrix = np.eye(4)
    y_up_to_z_up_matrix[:3, :3] = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    return (
        y_up_to_z_up_matrix
        @ layout_object_to_transform_matrix(layout)
        @ np.linalg.inv(y_up_to_z_up_matrix)
    )[:3, :3]


def test_visual_yaw_optimizer_returns_zero_for_unobserved_asset(
    tmp_path,
) -> None:
    glb_path = tmp_path / "book_001.glb"
    glb_path.write_bytes(b"glTF")
    scene_object = SceneObject(
        id="book_001",
        kind="asset",
        category="book",
        name="book",
        description="book",
        simready_glb_path=str(glb_path),
    )

    yaw_delta_degrees = VisualYawOptimizer(
        scene_object=scene_object,
        baked_scale_y_up=[1.0, 1.0, 1.0],
        vlm_client=object(),
        debug_output_root=tmp_path / "visual_yaw",
    ).optimize_z_up_yaw_degrees()

    assert yaw_delta_degrees == 0.0


def test_visual_yaw_optimizer_queries_vlm_and_saves_yawed_debug_image(
    monkeypatch,
    tmp_path,
) -> None:
    glb_path = tmp_path / "book_001.glb"
    glb_path.write_bytes(b"glTF")
    rgba_path = tmp_path / "book_001_rgba.png"
    Image.new("RGBA", (64, 64), (255, 0, 0, 255)).save(rgba_path)

    rendered_yaws: list[float] = []

    def fake_render(*, z_up_yaw_degrees, output_path, **_) -> None:
        rendered_yaws.append(z_up_yaw_degrees)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (512, 512), "black").save(output_path)

    class FakeVLM:
        image_paths: list[Path] = []

        def complete(self, *, image_path, **_) -> str:
            self.image_paths.append(Path(image_path))
            return '{"clockwise_yaw_degrees": 90, "reason": "long axis"}'

    monkeypatch.setattr(
        "embodichain.gen_sim.scene_engine.pipeline.utils.visual_yaw_optimizer._render_canonical_oblique_view",
        fake_render,
    )
    fake_vlm = FakeVLM()
    yaw_degrees = VisualYawOptimizer(
        scene_object=SceneObject(
            id="book_001",
            kind="asset",
            category="book",
            name="book",
            description="book",
            simready_glb_path=str(glb_path),
            visible_rgba_path=str(rgba_path),
        ),
        baked_scale_y_up=[1.0, 1.0, 1.0],
        vlm_client=fake_vlm,
        debug_output_root=tmp_path / "visual_yaw",
    ).optimize_z_up_yaw_degrees()

    assert yaw_degrees == -90.0
    assert rendered_yaws == [0.0, -90.0]
    assert fake_vlm.image_paths == [tmp_path / "visual_yaw" / "book_001_vlm_input.png"]
    assert (tmp_path / "visual_yaw" / "book_001_yaw_result.png").is_file()


def test_simready_visual_yaw_queries_every_asset(monkeypatch, tmp_path) -> None:
    queried_asset_ids: list[str] = []

    class FakeVisualYawOptimizer:
        def __init__(
            self,
            *,
            scene_object,
            baked_scale_y_up,
            vlm_client,
            debug_output_root,
        ) -> None:
            assert baked_scale_y_up == [1.0, 2.0, 3.0]
            assert vlm_client is fake_vlm_client
            assert debug_output_root == tmp_path / "visual_yaw"
            queried_asset_ids.append(scene_object.id)

        def optimize_z_up_yaw_degrees(self) -> float:
            return 15.0

    monkeypatch.setattr(
        "embodichain.gen_sim.scene_engine.pipeline.generation.scene_generation.VisualYawOptimizer",
        FakeVisualYawOptimizer,
    )
    scene = Scene(
        objects=[
            SceneObject(
                id="table",
                kind="table",
                category="table",
                name="table",
                description="table",
            ),
            SceneObject(
                id="book_001",
                kind="asset",
                category="book",
                name="book",
                description="book",
                simready_glb_path=str(tmp_path / "book_001.glb"),
            ),
            SceneObject(
                id="cup_001",
                kind="asset",
                category="cup",
                name="cup",
                description="cup",
                simready_glb_path=str(tmp_path / "cup_001.glb"),
            ),
        ]
    )
    fake_vlm_client = object()

    yaw_deltas_by_id = _optimize_simready_asset_visual_yaws(
        scene=scene,
        simready_assets_layout=[{"id": "book_001"}, {"id": "cup_001"}],
        coarse_layout_by_id={
            "book_001": {"scale": [1.0, 2.0, 3.0]},
            "cup_001": {"scale": [1.0, 2.0, 3.0]},
        },
        vlm_client=fake_vlm_client,
        debug_output_root=tmp_path / "visual_yaw",
    )

    assert queried_asset_ids == ["book_001", "cup_001"]
    assert yaw_deltas_by_id == {"book_001": 15.0, "cup_001": 15.0}


def test_visual_yaws_replace_coarse_rotations_but_preserve_positions() -> None:
    yawed_layout = _apply_visual_yaws_to_simready_asset_layouts(
        simready_assets_layout=[
            {
                "id": "book_001",
                "rot": [20.0, -15.0, 40.0],
                "pos": [0.1, 0.2, 0.3],
                "scale": [1.0, 1.0, 1.0],
            }
        ],
        z_up_yaws_degrees_by_id={"book_001": 45.0},
    )[0]

    expected_z_up_yaw = Rotation.from_euler("z", 45.0, degrees=True).as_matrix()
    assert np.allclose(_z_up_rotation_from_y_up_layout(yawed_layout), expected_z_up_yaw)
    assert np.allclose(yawed_layout["pos"], [0.1, 0.2, 0.3])


@pytest.mark.parametrize(
    "pivot_xy, expected_translation_xy",
    [(None, [0.0, 0.0]), ([1.0, 2.0], [2.0, 4.0])],
    ids=["world-origin", "table-center"],
)
def test_rotate_scene_z_up_world_rotates_complete_scene_and_support_metadata(
    pivot_xy: list[float] | None, expected_translation_xy: list[float]
) -> None:
    scene = Scene(
        objects=[
            SceneObject(
                id="table",
                kind="table",
                category="table",
                name="table",
                description="table",
                rot=[0.0, 0.0, 0.0],
                pos=[0.0, 0.0, 0.0],
                scale=[1.0, 1.0, 1.0],
                center_xy=[1.0, 2.0],
                support_surface_z=0.4,
                support_contour_xy=[[1.0, 2.0], [-1.0, 2.0]],
                support_optimization_rect_xy=[[1.0, 1.0], [-1.0, 1.0]],
            ),
            SceneObject(
                id="book_001",
                kind="asset",
                category="book",
                name="book",
                description="book",
                rot=[10.0, 20.0, 30.0],
                pos=[1.0, 2.0, 3.0],
                scale=[1.0, 2.0, 3.0],
                center_xy=[3.0, 4.0],
            ),
        ]
    )
    basis = y_up_to_z_up_matrix()
    inverse_basis = np.linalg.inv(basis)
    original_z_up_transforms = {
        scene_object.id: (
            basis
            @ layout_object_to_transform_matrix(scene_object_y_up_layout(scene_object))
            @ inverse_basis
        )
        for scene_object in scene.objects
    }
    expected_world_rotation = np.eye(4)
    expected_world_rotation[:3, :3] = Rotation.from_euler(
        "z", 180.0, degrees=True
    ).as_matrix()
    expected_world_rotation[:2, 3] = expected_translation_xy

    if pivot_xy is None:
        rotate_scene_z_up_world(scene=scene, rotation_degrees=180.0)
    else:
        rotate_scene_z_up_world(scene=scene, rotation_degrees=180.0, pivot_xy=pivot_xy)

    for scene_object in scene.objects:
        actual_z_up_transform = (
            basis
            @ layout_object_to_transform_matrix(scene_object_y_up_layout(scene_object))
            @ inverse_basis
        )
        assert np.allclose(
            actual_z_up_transform,
            expected_world_rotation @ original_z_up_transforms[scene_object.id],
        )
    translation_xy = np.asarray(expected_translation_xy)
    assert np.allclose(scene.table.center_xy, [-1.0, -2.0] + translation_xy)
    assert np.allclose(scene.assets[0].center_xy, [-3.0, -4.0] + translation_xy)
    assert np.allclose(
        scene.table.support_contour_xy, [[-1.0, -2.0], [1.0, -2.0]] + translation_xy
    )
    assert np.allclose(
        scene.table.support_optimization_rect_xy,
        [[-1.0, -1.0], [1.0, -1.0]] + translation_xy,
    )
    assert scene.table.support_surface_z == 0.4


def test_scene_heading_optional_preview_preserves_rigid_scene_quarter_turns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from embodichain.gen_sim.scene_engine.pipeline.utils import scene_heading

    table = trimesh.creation.box(extents=[2.0, 0.8, 1.0])
    table.apply_translation([0.75, 0.0, 0.375])
    table.export(tmp_path / "table.glb")
    table_layout = {
        "id": "table",
        "rot": [0.0, 0.0, 0.0],
        "pos": [2.0, 0.6, 1.5],
        "scale": [2.0, 1.0, 4.0],
    }
    assets_layout = [
        {
            "id": "tray_001",
            "rot": [0.0, 32.0, 0.0],
            "pos": [3.0, 1.0, 5.0],
            "scale": [1.0, 1.0, 1.0],
        },
        {
            "id": "cup_001",
            "rot": [15.0, 0.0, 0.0],
            "pos": [3.2, 1.3, 4.8],
            "scale": [1.0, 1.0, 1.0],
        },
    ]
    # Scaling and translating the offset table center gives Y-up (3.5, 0.6, 3.0).
    expected_pivot_xy = [3.5, -3.0]
    source_path = tmp_path / "source.png"
    Image.new("RGB", (16, 16)).save(source_path)
    previews = []

    def render_comparison(*, image_path, candidate_layouts, geometry_root, output_path):
        assert image_path == source_path
        assert geometry_root == tmp_path
        assert list(candidate_layouts) == [0, 90, 180, 270]
        assert candidate_layouts[0] == [table_layout, *assets_layout]
        original_poses = [
            layout_object_to_transform_matrix(i) for i in candidate_layouts[0]
        ]
        expected_asset_positions = {
            90: [5.5, 1.0, 3.5],
            180: [4, 1, 1],
            270: [1.5, 1, 2.5],
        }
        for angle, expected_position in expected_asset_positions.items():
            assert [item["id"] for item in candidate_layouts[angle]] == [
                "table",
                "tray_001",
                "cup_001",
            ]
            turned_poses = [
                layout_object_to_transform_matrix(i) for i in candidate_layouts[angle]
            ]
            assert np.allclose(turned_poses[1][:3, 3], expected_position)
            for index, original_pose in enumerate(original_poses):
                for other_index in range(index + 1, len(original_poses)):
                    assert np.allclose(
                        np.linalg.solve(turned_poses[index], turned_poses[other_index]),
                        np.linalg.solve(original_pose, original_poses[other_index]),
                    )
        Image.new("RGB", (16, 16)).save(output_path)
        previews.append(output_path)
        return output_path

    monkeypatch.setattr(
        scene_heading, "render_scene_heading_comparison", render_comparison
    )
    decision, pivot_xy = scene_heading.review_scene_heading(
        image_path=source_path,
        table_layout=table_layout,
        assets_layout=assets_layout,
        geometry_root=tmp_path,
        debug_output_root=tmp_path / "debug",
        render_candidates=True,
    )
    selection = json.loads(
        (tmp_path / "debug" / "scene_heading_selection.json").read_text()
    )

    assert decision == 0
    assert pivot_xy == pytest.approx(expected_pivot_xy)
    assert previews == [tmp_path / "debug" / "scene_heading_comparison.png"]
    assert selection["yaw_degrees"] == 0
    assert selection["method"] == "mask_geometry"
    assert selection["pivot_xy_z_up_m"] == pytest.approx(expected_pivot_xy)


def _heading_mask_fixture(
    tmp_path: Path,
    other_position: tuple[float, float, float],
    other_mask_center: tuple[int, int] | None,
) -> dict[str, object]:
    image_path = tmp_path / "source.png"
    Image.new("RGB", (100, 60)).save(image_path)

    def layout(uid: str, position: tuple[float, float, float]) -> dict[str, object]:
        trimesh.creation.box().export(tmp_path / f"{uid}.glb")
        return {"id": uid, "pos": list(position), "rot": [0, 0, 0], "scale": [1, 1, 1]}

    masks = {"first": (12, 24)}
    if other_mask_center is not None:
        masks["other"] = other_mask_center
    mask_paths = {}
    for uid, (image_x, image_y) in masks.items():
        mask = Image.new("L", (100, 60))
        mask.paste(255, (image_x - 4, image_y - 4, image_x + 5, image_y + 5))
        # One stray pixel cannot move the median to the opposite image endpoint.
        mask.putpixel((99 - image_x, 0), 255)
        mask_paths[uid] = tmp_path / f"{uid}.png"
        mask.save(mask_paths[uid])
    return {
        "image_path": image_path,
        "table_layout": layout("table", (0.0, 0.0, 0.0)),
        "assets_layout": [
            layout("first", (0.0, 0.0, 0.0)),
            layout("other", other_position),
        ],
        "geometry_root": tmp_path,
        "debug_output_root": tmp_path / "debug",
        "mask_paths": mask_paths,
    }


@pytest.mark.parametrize(
    ("other_x", "other_z", "other_mask_center", "expected_yaw"),
    [
        (0.6, 0.0, (88, 24), 0),
        (0.0, 0.6, (88, 24), 90),
        (-0.6, 0.0, (88, 24), 180),
        (0.0, -0.6, (88, 24), 270),
        (0.0, 0.6, (12, 48), 0),
        (0.0, -0.6, (12, 48), 180),
    ],
)
def test_scene_heading_selects_from_mask_geometry_without_rendering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    other_x: float,
    other_z: float,
    other_mask_center: tuple[int, int],
    expected_yaw: int,
) -> None:
    from embodichain.gen_sim.scene_engine.pipeline.utils import scene_heading

    def reject_render(**_):
        pytest.fail("The program heading proposal must not render candidates.")

    monkeypatch.setattr(scene_heading, "render_scene_heading_comparison", reject_render)
    inputs = _heading_mask_fixture(tmp_path, (other_x, 0.0, other_z), other_mask_center)
    yaw, _ = scene_heading.review_scene_heading(**inputs)
    result = json.loads(
        (tmp_path / "debug" / "scene_heading_selection.json").read_text()
    )

    assert yaw == expected_yaw
    assert result["method"] == "mask_geometry"
    assert result["mask_evidence"]["suggested_yaw_degrees"] == expected_yaw


@pytest.mark.parametrize(
    ("other_position", "other_mask_center"),
    [((0.0, 0.0, 0.6), None), ((0.6, 0.0, 0.6), (88, 24))],
    ids=["one-mask", "tied-diagonal-headings"],
)
def test_scene_heading_keeps_zero_for_missing_or_ambiguous_mask_evidence(
    tmp_path: Path,
    other_position: tuple[float, float, float],
    other_mask_center: tuple[int, int] | None,
) -> None:
    from embodichain.gen_sim.scene_engine.pipeline.utils import scene_heading

    inputs = _heading_mask_fixture(tmp_path, other_position, other_mask_center)
    yaw, _ = scene_heading.review_scene_heading(**inputs)
    result = json.loads(
        (tmp_path / "debug" / "scene_heading_selection.json").read_text()
    )

    assert yaw == 0
    assert result["method"] == "mask_geometry"
    assert result["mask_evidence"].get("suggested_yaw_degrees") is None


def test_scene_heading_rejects_weak_lateral_endpoint_projection(tmp_path: Path) -> None:
    from embodichain.gen_sim.scene_engine.pipeline.utils import scene_heading

    # Both 0 and 90 preserve left/right order. Only 90 explains the clearly
    # lateral source pair: 0 projects 4 cm versus the best candidate's 60 cm.
    # The 4 cm already passes the separate 2%-of-table-span absolute gate.
    inputs = _heading_mask_fixture(tmp_path, (0.04, 0.0, 0.6), (88, 24))
    yaw, _ = scene_heading.review_scene_heading(**inputs)
    result = json.loads(
        (tmp_path / "debug" / "scene_heading_selection.json").read_text()
    )

    assert yaw == 90
    assert result["mask_evidence"]["suggested_yaw_degrees"] == 90


def _stub_settled_heading_render(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, object]]:
    from embodichain.gen_sim.scene_engine.pipeline.utils import scene_heading

    calls = []

    def render(**kwargs):
        calls.append(kwargs)
        assert list(kwargs["candidate_layouts"]) == [0, 90, 180, 270]
        output = kwargs["output_path"]
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16)).save(output)
        Image.new("RGB", (16, 16)).save(
            output.with_name(scene_heading.SCENE_HEADING_CANDIDATES_FILENAME)
        )
        return output

    monkeypatch.setattr(scene_heading, "render_scene_heading_comparison", render)
    return calls


@pytest.mark.parametrize("mask_count", [0, 1, 2])
def test_settled_heading_vision_overrides_program_even_without_multiple_masks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mask_count: int
) -> None:
    from embodichain.gen_sim.scene_engine.pipeline.utils import scene_heading
    from embodichain.gen_sim.services import (
        ChatResponse,
        GenSimServices,
        GenSimServicesCfg,
        ImagePart,
    )
    from embodichain.gen_sim.services.compat import VisionFacade

    inputs = _heading_mask_fixture(
        tmp_path, (0.0, 0.0, 0.6), (88, 24) if mask_count == 2 else None
    )
    if mask_count == 0:
        inputs["mask_paths"] = {}
    proposed, pivot = scene_heading.review_scene_heading(**inputs)
    assert proposed == (90 if mask_count == 2 else 0)
    selection_path = tmp_path / "debug/scene_heading_selection.json"
    program_evidence = json.loads(selection_path.read_text())["mask_evidence"]
    render_calls = _stub_settled_heading_render(monkeypatch)
    model_images = []

    class Chat:
        def complete(self, request):
            model_images.extend(
                part.source
                for part in request.messages[1].content
                if isinstance(part, ImagePart)
            )
            return ChatResponse(
                '{"yaw_degrees": 270, "reason": "Visible object fronts match 270."}'
            )

    # Only scene_understanding is injected. Keeping the original orientation
    # role would fail rather than accidentally contacting a configured provider.
    with GenSimServices(
        GenSimServicesCfg(), injected={"scene_understanding": Chat()}
    ) as services:
        decision = scene_heading.review_settled_scene_heading(
            **{key: value for key, value in inputs.items() if key != "mask_paths"},
            vision_client=VisionFacade(services, "scene_asset_orientation"),
            proposed_yaw_degrees=proposed,
            pivot_xy=pivot,
        )
    selection = json.loads(selection_path.read_text())

    assert decision == 270
    assert selection["method"] == "mask_geometry_vlm"
    assert selection["program_yaw_degrees"] == proposed
    assert selection["yaw_degrees"] == 270
    assert selection["mask_evidence"] == program_evidence
    assert len(render_calls) == 1
    assert render_calls[0]["proposed_yaw_degrees"] == proposed
    assert render_calls[0]["candidate_layouts"][0] == [
        inputs["table_layout"],
        *inputs["assets_layout"],
    ]
    assert model_images == [
        inputs["image_path"],
        render_calls[0]["output_path"].with_name(
            scene_heading.SCENE_HEADING_CANDIDATES_FILENAME
        ),
    ]
    assert selection["model_images"] == [str(path) for path in model_images]
    assert isinstance(selection["vision_review"]["status"], str)
    assert "Visible object fronts match 270." in json.dumps(selection["vision_review"])


@pytest.mark.parametrize(
    ("response", "expected_yaw"),
    [
        ('{"yaw_degrees": 0, "reason": "The original heading matches."}', 0),
        ('{"yaw_degrees": null, "reason": "The object is symmetric."}', 90),
        ("not JSON", 90),
        ('{"yaw_degrees": true, "reason": "Invalid boolean heading."}', 90),
        ('{"yaw_degrees": 45, "reason": "Invalid non-quarter heading."}', 90),
        ('{"yaw_degrees": 180, "reason": " "}', 90),
        (None, 90),
    ],
    ids=[
        "zero-override",
        "abstain",
        "malformed",
        "boolean",
        "non-quarter",
        "blank-reason",
        "service-failure",
    ],
)
def test_settled_heading_complete_only_review_and_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    response: str | None,
    expected_yaw: int,
) -> None:
    from embodichain.gen_sim.scene_engine.pipeline.utils import scene_heading
    from embodichain.gen_sim.services import ServiceTransportError

    inputs = _heading_mask_fixture(tmp_path, (0.0, 0.0, 0.6), (88, 24))
    proposed, pivot = scene_heading.review_scene_heading(**inputs)
    assert proposed == 90
    selection_path = tmp_path / "debug/scene_heading_selection.json"
    program_evidence = json.loads(selection_path.read_text())["mask_evidence"]
    render_calls = _stub_settled_heading_render(monkeypatch)
    model_images = []

    class Vision:
        def complete(self, *, image_path, **_):
            model_images.append(image_path)
            if response is None:
                raise ServiceTransportError("Mock vision service unavailable.")
            return response

    decision = scene_heading.review_settled_scene_heading(
        **{key: value for key, value in inputs.items() if key != "mask_paths"},
        vision_client=Vision(),
        proposed_yaw_degrees=proposed,
        pivot_xy=pivot,
    )
    selection = json.loads(selection_path.read_text())

    assert decision == expected_yaw
    assert selection["program_yaw_degrees"] == 90
    assert selection["yaw_degrees"] == expected_yaw
    assert selection["method"] == (
        "mask_geometry_vlm" if expected_yaw == 0 else "mask_geometry"
    )
    assert selection["mask_evidence"] == program_evidence
    assert model_images == [render_calls[0]["output_path"]]
    assert selection["model_images"] == [str(path) for path in model_images]
    assert isinstance(selection["vision_review"]["status"], str)


def test_table_only_refinement_still_reviews_and_applies_scene_heading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from embodichain.gen_sim.scene_engine.pipeline.generation import scene_generation

    geometry_path = tmp_path / "table.glb"
    trimesh.creation.box(extents=[2.0, 0.4, 1.0]).export(geometry_path)
    source = tmp_path / "source.png"
    Image.new("RGB", (8, 8)).save(source)
    initial_layout = {
        "id": "table",
        "pos": [1.0, 0.2, 3.0],
        "rot": [0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
    }
    (tmp_path / "simready_layout.json").write_text(json.dumps([initial_layout]))
    scene = Scene(
        objects=[
            SceneObject(
                id="table",
                kind="table",
                category="table",
                name="table",
                description="Rectangular table",
                simready_glb_path=str(geometry_path),
            )
        ]
    )
    graph = SceneGraph(nodes=[SceneGraphNode(object_id="table", parent_id=None)])
    client = object()
    reviews = []

    def review_settled(
        *, table_layout, assets_layout, vision_client, proposed_yaw_degrees, **_
    ):
        assert proposed_yaw_degrees == 0
        assert vision_client is client
        assert assets_layout == []
        # The same finalization boundary must synchronize the table before review.
        np.testing.assert_allclose(
            layout_object_to_transform_matrix(scene_object_y_up_layout(scene.table)),
            layout_object_to_transform_matrix(table_layout),
        )
        reviews.append(table_layout)
        return 90

    monkeypatch.setattr(
        scene_generation, "review_settled_scene_heading", review_settled
    )
    final_table, assets = scene_generation._layout_refinement(
        scene=scene,
        scene_graph=graph,
        image_path=source,
        simready_geometry_output_root=tmp_path,
        debug_output_root=tmp_path,
        z_up_yaws_degrees_by_id={},
        vlm_client=client,
    )

    assert len(reviews) == 1
    assert assets == []
    assert layout_object_to_transform_matrix(final_table)[:3, :3] == pytest.approx(
        Rotation.from_euler("y", 90, degrees=True).as_matrix()
    )
    snapshots = json.loads((tmp_path / "layout_pose_snapshots.json").read_text())
    assert snapshots["frames"]["before_heading"] == reviews
    assert snapshots["frames"]["final"] == [final_table]


@pytest.mark.parametrize(
    ("side", "expected"),
    [
        ("left", [-0.92, 0.0, -0.3]),
        ("right", [1.52, 0.0, -0.3]),
        ("front", [0.0, -2.32, -0.3]),
        ("back", [0.0, 1.92, -0.3]),
    ],
)
def test_floor_side_uses_the_final_camera_axes(side, expected):
    from embodichain.gen_sim.scene_engine.pipeline.utils.floor_placement import (
        floor_placement_delta,
    )

    table = np.array([[-1.0, -2.0, 0.0], [1.0, 2.0, 0.75]])
    asset = np.array([[-0.5, 0.1, 0.3], [-0.1, 0.3, 0.8]])
    assert floor_placement_delta(table, asset, side) == pytest.approx(expected)


def test_floor_only_asset_settles_using_the_final_table_edge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from embodichain.gen_sim.scene_engine.pipeline.generation import scene_generation

    def asset(uid, kind, extents, image_box):
        geometry = tmp_path / f"{uid}.glb"
        trimesh.creation.box(extents=extents).export(geometry)
        mask_path = tmp_path / f"{uid}_mask.png"
        mask = Image.new("L", (64, 64))
        ImageDraw.Draw(mask).rectangle(image_box, fill=255)
        mask.save(mask_path)
        return SceneObject(
            id=uid,
            kind=kind,
            category=uid,
            name=uid,
            description=uid,
            simready_glb_path=str(geometry),
            mask_path=str(mask_path),
        )

    table = asset("table", "table", [2.0, 0.6, 1.0], [18, 10, 62, 57])
    bin_asset = asset("bin_001", "asset", [0.2, 0.4, 0.2], [1, 19, 10, 40])
    child = asset("child_001", "asset", [0.1, 0.1, 0.1], [2, 20, 8, 27])
    table.support_surface_z = 0.3
    table.support_contour_xy = [[-1, -0.5], [1, -0.5], [1, 0.5], [-1, 0.5]]
    table.support_optimization_rect_xy = table.support_contour_xy
    scene = Scene(objects=[table, bin_asset, child])
    graph = SceneGraph(
        nodes=[
            SceneGraphNode(object_id="table", parent_id=None),
            SceneGraphNode(
                object_id=bin_asset.id, parent_id="table", parent_relation="on"
            ),
            SceneGraphNode(
                object_id=child.id, parent_id=bin_asset.id, parent_relation="on"
            ),
        ]
    )
    original_graph = graph.to_dict()
    image = tmp_path / "source.png"
    overlay = tmp_path / "asset_masks_with_ids.png"
    Image.new("RGB", (64, 64)).save(image)
    Image.new("RGB", (64, 64)).save(overlay)
    layouts = [
        {"id": obj.id, "pos": pos, "rot": [0.0] * 3, "scale": [1.0] * 3}
        for obj, pos in (
            (table, [0.0] * 3),
            (bin_asset, [0.9, 0.1, 0.3]),
            (child, [0.94, 0.32, 0.3]),
        )
    ]
    (tmp_path / "simready_layout.json").write_text(json.dumps(layouts))

    class Vision:
        def complete_images(self, *, user_prompt, image_paths, **_):
            assert image_paths == [image, overlay]
            assert set(json.loads(user_prompt)["candidates"]) == {bin_asset.id}
            return '{"bin_001": "left"}'

    def choose_heading(*, proposed_yaw_degrees, assets_layout, **_):
        assert proposed_yaw_degrees == 0
        assert assets_layout == []
        return 90

    settled_ids = []

    def settle_floor_root(self):
        assert self.dynamic_asset_ids == {bin_asset.id}
        assert self.static_asset_ids == set()
        assert self.table_body.scene_object.id == table.id
        # The table base is at -0.30 m in the scene and at Z=0 in simulation.
        assert self.table_body.y_up_layout["pos"] == pytest.approx([0.0, 0.3, 0.0])
        (body,) = self.participant_bodies
        assert body.scene_object.id == bin_asset.id
        x, y, z = body.y_up_layout["pos"]
        assert x == pytest.approx(-0.62)
        assert y == pytest.approx(0.22)  # Half height plus the 2 cm release.
        settled_ids.append(bin_asset.id)
        # Tipping widens the bin towards the table. Its final collision pose
        # leaves a 3 mm visual gap above the simulation plane.
        return {bin_asset.id: {"pos": [x + 0.03, 0.103, z], "rot": [0.0, 0.0, 90.0]}}

    monkeypatch.setattr(scene_generation, "review_asset_posture", lambda **_: None)
    monkeypatch.setattr(
        scene_generation, "review_settled_scene_heading", choose_heading
    )
    monkeypatch.setattr(scene_generation.GravitySettler, "settle", settle_floor_root)
    scene_generation._layout_refinement(
        scene=scene,
        scene_graph=graph,
        image_path=image,
        simready_geometry_output_root=tmp_path,
        debug_output_root=tmp_path,
        z_up_yaws_degrees_by_id={},
        vlm_client=Vision(),
    )

    assert settled_ids == [bin_asset.id]
    table_lo, _ = measure_scene_object_z_up_world_aabb(scene_object=table)
    bin_lo, bin_hi = measure_scene_object_z_up_world_aabb(scene_object=bin_asset)
    assert table.pos == pytest.approx([0.0, 0.0, 0.0])
    assert table_lo[0] == pytest.approx(-0.5)
    assert bin_hi[0] == pytest.approx(-0.52)
    assert bin_lo[2] == pytest.approx(table_lo[2] + 0.003)
    assert bin_asset.rot == pytest.approx([0.0, 0.0, 90.0])
    assert bin_asset.placement_support == "floor"
    assert child.placement_support is None
    assert graph.to_dict() == original_graph
    frames = json.loads((tmp_path / "layout_pose_snapshots.json").read_text())["frames"]
    assert frames["after_y_align"][1]["pos"] == pytest.approx(layouts[1]["pos"])
    _, bin_before, child_before = frames["before_heading"]
    _, bin_final, child_final = frames["final"]
    assert np.linalg.solve(
        layout_object_to_transform_matrix(bin_final),
        layout_object_to_transform_matrix(child_final),
    ) == pytest.approx(
        np.linalg.solve(
            layout_object_to_transform_matrix(bin_before),
            layout_object_to_transform_matrix(child_before),
        )
    )


@pytest.mark.parametrize(
    ("program_heading_degrees", "heading_degrees", "planar_turn"),
    [
        (180, 90, [[0, -1], [1, 0]]),
        (0, 180, [[-1, 0], [0, -1]]),
        (90, 270, [[0, 1], [-1, 0]]),
    ],
)
def test_refinement_levels_and_grounds_bases_with_descendants_before_heading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    program_heading_degrees: int,
    heading_degrees: int,
    planar_turn: list[list[int]],
) -> None:
    from embodichain.gen_sim.scene_engine.pipeline.generation import scene_generation

    table_id = "table"
    seed_positions = {
        table_id: [0.0, 0.0, 0.0],
        "low_001": [0.42, -0.25, 0.24],
        "high_001": [-0.38, 0.12, -0.21],
        "child_001": [0.44, -0.50, 0.26],
    }
    parent_by_id = {
        "low_001": table_id,
        "high_001": table_id,
        "child_001": "low_001",
    }
    low_yaw = Rotation.from_euler("y", 35.0, degrees=True)
    rest = Rotation.from_euler("x", 90.0, degrees=True)
    tilted = low_yaw * Rotation.from_euler("x", 100.0, degrees=True)
    high_yaw = Rotation.from_euler("y", -10.0, degrees=True)
    seed_rotations = {"low_001": tilted, "high_001": high_yaw}
    tabletop = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
    support_height = 0.20
    table_max_y = 0.50
    table_span = 2.0
    placement_clearance = 0.02
    asset_half_height = 0.05
    low_half_height_after_turn = 0.15
    objects = []
    layouts = []
    for object_id, pos in seed_positions.items():
        is_table = object_id == table_id
        mesh_extents = (
            [table_span, 2 * support_height, table_span]
            if is_table
            else [2 * asset_half_height] * 3
        )
        if object_id == "low_001":
            mesh_extents[2] = 2 * low_half_height_after_turn
        geometry_path = tmp_path / f"{object_id}.glb"
        mesh = trimesh.creation.box(extents=mesh_extents)
        if is_table:
            protrusion_height = table_max_y - support_height
            protrusion = trimesh.creation.box(extents=[0.10, protrusion_height, 0.10])
            protrusion.apply_translation(
                [1.1, support_height + protrusion_height / 2, 0.9]
            )
            mesh = trimesh.util.concatenate([mesh, protrusion])
        mesh.export(geometry_path)
        objects.append(
            SceneObject(
                id=object_id,
                kind="table" if is_table else "asset",
                category=object_id,
                name=object_id,
                description=object_id,
                simready_glb_path=str(geometry_path),
                support_surface_z=support_height if is_table else None,
                support_contour_xy=tabletop if is_table else None,
                support_optimization_rect_xy=tabletop if is_table else None,
            )
        )
        layouts.append(
            {
                "id": object_id,
                "rot": seed_rotations.get(object_id, Rotation.identity())
                .as_euler("xyz", degrees=True)
                .tolist(),
                "pos": pos,
                "scale": [1.0] * 3,
            }
        )
    (tmp_path / "simready_layout.json").write_text(
        json.dumps(layouts), encoding="utf-8"
    )
    scene = Scene(objects=objects)
    scene_graph = SceneGraph(
        nodes=[
            SceneGraphNode(object_id=table_id, parent_id=None),
            *[
                SceneGraphNode(
                    object_id=asset_id, parent_id=parent_id, parent_relation="on"
                )
                for asset_id, parent_id in parent_by_id.items()
            ],
        ]
    )

    reviewed = []

    def review_base(*, glb_path, **_):
        reviewed.append(glb_path.stem)
        return rest.as_matrix() if glb_path.stem == "low_001" else np.eye(3)

    low_rotation = low_yaw * rest
    child_delta = low_rotation * tilted.inv()
    expected_low_y = support_height + low_half_height_after_turn + placement_clearance
    expected_low_pos = [
        seed_positions["low_001"][0],
        expected_low_y,
        seed_positions["low_001"][2],
    ]
    expected_child_pos = expected_low_pos + child_delta.apply(
        np.asarray(seed_positions["child_001"]) - seed_positions["low_001"]
    )

    def choose_heading(*, table_layout, assets_layout, **_):
        assert table_layout["pos"] == pytest.approx(seed_positions[table_id])
        assert reviewed == ["low_001", "high_001"]
        by_id = {
            layout["id"]: layout_object_to_transform_matrix(layout)
            for layout in assets_layout
        }
        assert by_id["low_001"][:3, 3] == pytest.approx(expected_low_pos)
        assert by_id["low_001"][:3, :3] == pytest.approx(low_rotation.as_matrix())
        assert by_id["high_001"][1, 3] == pytest.approx(
            support_height + asset_half_height + placement_clearance
        )
        assert by_id["high_001"][:3, :3] == pytest.approx(high_yaw.as_matrix())
        assert by_id["child_001"][:3, 3] == pytest.approx(expected_child_pos)
        assert by_id["child_001"][:3, :3] == pytest.approx(child_delta.as_matrix())
        return program_heading_degrees, [0.0, 0.0]

    settled = []

    def settle_vertically(self):
        assert self.dynamic_asset_ids == {"low_001", "high_001"}
        results = {}
        for body in self.participant_bodies:
            x, y, z = body.y_up_layout["pos"]
            half_height = (
                low_half_height_after_turn
                if body.scene_object.id == "low_001"
                else asset_half_height
            )
            assert y - half_height == pytest.approx(
                support_height + placement_clearance
            )
            results[body.scene_object.id] = {
                "pos": [x, y - placement_clearance, z],
                "rot": body.y_up_layout["rot"],
            }
        settled.append(results)
        return results

    def keep_child_poses(*, assets_layout, z_up_yaws_degrees_by_id, **_):
        assert z_up_yaws_degrees_by_id == {"child_001": -program_heading_degrees}
        return assets_layout

    heading_client = object()
    vision_reviews = []

    def review_settled(
        *,
        table_layout,
        assets_layout,
        vision_client,
        proposed_yaw_degrees,
        pivot_xy,
        **_,
    ):
        assert vision_client is heading_client
        assert proposed_yaw_degrees == program_heading_degrees
        assert pivot_xy == [0.0, 0.0]
        assert len(settled) == 1
        by_id = {
            layout["id"]: layout_object_to_transform_matrix(layout)
            for layout in assets_layout
        }
        assert by_id["low_001"][:3, 3] == pytest.approx(
            np.asarray(expected_low_pos) - [0.0, placement_clearance, 0.0]
        )
        assert by_id["child_001"][:3, 3] == pytest.approx(
            expected_child_pos - [0.0, placement_clearance, 0.0]
        )
        for asset in scene.assets:
            np.testing.assert_allclose(
                layout_object_to_transform_matrix(scene_object_y_up_layout(asset)),
                by_id[asset.id],
                atol=1e-8,
            )
        vision_reviews.append([table_layout, *assets_layout])
        return heading_degrees

    monkeypatch.setattr(scene_generation, "review_asset_posture", review_base)
    monkeypatch.setattr(scene_generation, "review_scene_heading", choose_heading)
    monkeypatch.setattr(
        scene_generation, "review_settled_scene_heading", review_settled
    )
    monkeypatch.setattr(scene_generation.GravitySettler, "settle", settle_vertically)
    monkeypatch.setattr(scene_generation, "_refine_on_children_bfs", keep_child_poses)
    scene_generation._layout_refinement(
        scene=scene,
        scene_graph=scene_graph,
        image_path=tmp_path / "source.png",
        simready_geometry_output_root=tmp_path,
        debug_output_root=tmp_path,
        z_up_yaws_degrees_by_id={"child_001": 0.0},
        vlm_client=heading_client,
    )

    snapshots = json.loads((tmp_path / "layout_pose_snapshots.json").read_text())
    assert snapshots["up_axis"] == "Y"
    assert len(vision_reviews) == 1
    frames = {
        stage: {layout["id"]: layout for layout in layouts}
        for stage, layouts in snapshots["frames"].items()
    }
    low_before = frames["before_posture"]["low_001"]
    low_posture = frames["before_y_align"]["low_001"]
    low_placed = frames["after_y_align"]["low_001"]
    assert low_before["rot"] != pytest.approx(low_posture["rot"])
    assert low_placed["rot"] == pytest.approx(low_posture["rot"])
    assert low_placed["pos"][1] != pytest.approx(low_posture["pos"][1])

    yaw = Rotation.from_euler("y", heading_degrees, degrees=True)
    whole_scene_turn = np.eye(4)
    whole_scene_turn[:3, :3] = yaw.as_matrix()
    assert frames["before_heading"].keys() == frames["final"].keys()
    for uid, settled in frames["before_heading"].items():
        np.testing.assert_allclose(
            layout_object_to_transform_matrix(frames["final"][uid]),
            whole_scene_turn @ layout_object_to_transform_matrix(settled),
            atol=1e-8,
        )
    assert scene.table is not None
    table_pose = layout_object_to_transform_matrix(
        scene_object_y_up_layout(scene.table)
    )
    assert table_pose[:3, :3] == pytest.approx(yaw.as_matrix())
    expected_support = np.asarray(tabletop) @ np.asarray(planar_turn).T
    np.testing.assert_allclose(scene.table.support_contour_xy, expected_support)
    np.testing.assert_allclose(
        scene.table.support_optimization_rect_xy, expected_support
    )
    by_id = {asset.id: asset for asset in scene.assets}
    low = layout_object_to_transform_matrix(scene_object_y_up_layout(by_id["low_001"]))
    assert low[:3, :3] == pytest.approx((yaw * low_rotation).as_matrix())
    assert low[:3, 3] == pytest.approx(
        yaw.apply(np.asarray(expected_low_pos) - [0.0, placement_clearance, 0.0])
    )
    child = layout_object_to_transform_matrix(
        scene_object_y_up_layout(by_id["child_001"])
    )
    assert child[:3, :3] == pytest.approx((yaw * child_delta).as_matrix())
    assert child[:3, 3] == pytest.approx(
        yaw.apply(expected_child_pos - [0.0, placement_clearance, 0.0])
    )


def test_rotate_scene_z_up_world_rejects_non_finite_angle() -> None:
    with pytest.raises(ValueError, match="rotation_degrees must be finite"):
        rotate_scene_z_up_world(scene=Scene(), rotation_degrees=float("nan"))


def test_runtime_revolute_limits_convert_degrees_without_mutating_usd(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdPhysics
    from embodichain.gen_sim.asset_engine.utils.articulated_usdc_utils import (
        _read_revolute_qpos_limits,
    )

    path = tmp_path / "limits.usdc"
    stage = Usd.Stage.CreateNew(str(path))
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/rocker")
    joint.CreateLowerLimitAttr(-8.0)
    joint.CreateUpperLimitAttr(8.0)
    slider = UsdPhysics.PrismaticJoint.Define(stage, "/slider")
    slider.CreateLowerLimitAttr(0.0)
    slider.CreateUpperLimitAttr(0.1)
    stage.GetRootLayer().Save()
    before = path.read_bytes()
    limits = _read_revolute_qpos_limits(path)
    assert limits["rocker"] == pytest.approx(np.deg2rad([-8.0, 8.0]))
    assert "slider" not in limits
    assert path.read_bytes() == before


def test_articulated_usdcs_use_visible_rgba_in_scene_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeArticulatedGenerationClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str | Path, str | Path]] = []

        def generate_articulated_usdc(
            self,
            *,
            prompt: str,
            image_path: str | Path,
            output_path: str | Path,
        ) -> Path:
            self.calls.append((prompt, image_path, output_path))
            resolved_output_path = Path(output_path)
            resolved_output_path.write_bytes(b"USDC")
            return resolved_output_path

    drawer_rgba_path = tmp_path / "drawer_rgba.png"
    microwave_rgba_path = tmp_path / "microwave_rgba.png"
    drawer_rgba_path.write_bytes(b"PNG")
    microwave_rgba_path.write_bytes(b"PNG")
    drawer = SceneObject(
        id="drawer_001",
        kind="asset",
        category="drawer",
        name="drawer",
        description="white drawer with a pull handle",
        is_articulated=True,
        visible_rgba_path=str(drawer_rgba_path),
    )
    microwave = SceneObject(
        id="microwave_001",
        kind="asset",
        category="microwave",
        name="microwave",
        description="black microwave with a hinged door",
        is_articulated=True,
        visible_rgba_path=str(microwave_rgba_path),
    )
    static_mug = SceneObject(
        id="mug_001",
        kind="asset",
        category="mug",
        name="mug",
        description="blue ceramic mug",
        visible_rgba_path=str(tmp_path / "mug_rgba.png"),
    )
    client = FakeArticulatedGenerationClient()

    def fake_canonicalize(usdc_path: str | Path) -> Path:
        return Path(usdc_path)

    monkeypatch.setattr(
        "embodichain.gen_sim.scene_engine.pipeline.generation.scene_generation._canonicalize_articulated_usdc_bottom_center",
        fake_canonicalize,
    )

    _generate_articulated_usdcs(
        scene=Scene(objects=[drawer, static_mug, microwave]),
        output_root=tmp_path / "articulated_geometry",
        coarse_scales_y_up_by_id={
            "drawer_001": [1.0, 2.0, 3.0],
            "microwave_001": [4.0, 5.0, 6.0],
        },
        articulated_generation_client=client,  # type: ignore[arg-type]
    )

    assert "white drawer with a pull handle" in client.calls[0][0]
    assert "black microwave with a hinged door" in client.calls[1][0]
    assert all("self-contained USDC" in call[0] for call in client.calls)
    assert all("gen_sim:closedPosition" in call[0] for call in client.calls)
    assert all("closed_position=<endpoint>" in call[0] for call in client.calls)
    assert all("Do not patch USD files" in call[0] for call in client.calls)
    assert drawer.articulated_usdc_path == str(
        tmp_path / "articulated_geometry" / "drawer_001.usdc"
    )
    assert microwave.articulated_usdc_path == str(
        tmp_path / "articulated_geometry" / "microwave_001.usdc"
    )
    assert drawer.to_dict()["articulated_usdc_path"] == drawer.articulated_usdc_path
    assert drawer.articulated_usdc_scale == [1.0, 2.0, 3.0]
    assert microwave.articulated_usdc_scale == [4.0, 5.0, 6.0]
    assert static_mug.articulated_usdc_path is None


def test_articulated_usdc_canonicalization_moves_bottom_center_to_origin(
    tmp_path: Path,
) -> None:
    """A root-level translation preserves the hierarchy while normalizing its origin."""
    usdc_path = tmp_path / "offset_drawer.usdc"
    stage = Usd.Stage.CreateNew(str(usdc_path))
    UsdGeom.Xform.Define(stage, "/World")
    root = UsdGeom.Xform.Define(stage, "/World/drawer")
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    cube = UsdGeom.Cube.Define(stage, "/World/drawer/housing")
    cube.CreateSizeAttr(2.0)
    UsdGeom.Xformable(cube).AddTranslateOp().Set(Gf.Vec3d(2.0, 3.0, 4.0))
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().Save()

    _canonicalize_articulated_usdc_bottom_center(usdc_path)

    reopened_stage = Usd.Stage.Open(str(usdc_path))
    bounds = (
        UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        )
        .ComputeLocalBound(reopened_stage.GetDefaultPrim())
        .ComputeAlignedBox()
    )
    minimum, maximum = bounds.GetMin(), bounds.GetMax()
    assert minimum[1] == pytest.approx(0.0)
    assert (minimum[0] + maximum[0]) / 2.0 == pytest.approx(0.0)
    assert (minimum[2] + maximum[2]) / 2.0 == pytest.approx(0.0)


def test_table_root_update_propagates_its_pose_delta_to_descendants() -> None:
    scene_graph = SceneGraph(
        nodes=[
            SceneGraphNode(object_id="table", parent_id=None),
            SceneGraphNode(
                object_id="book_001",
                parent_id="table",
                parent_relation="on",
            ),
            SceneGraphNode(
                object_id="pen_001",
                parent_id="book_001",
                parent_relation="on",
            ),
        ]
    )
    book_layout = {
        "id": "book_001",
        "rot": [0.0, 0.0, 0.0],
        "pos": [0.1, 0.2, 0.3],
        "scale": [1.0, 1.0, 1.0],
    }
    pen_layout = {
        "id": "pen_001",
        "rot": [0.0, 0.0, 0.0],
        "pos": [0.2, 0.25, 0.35],
        "scale": [1.0, 1.0, 1.0],
    }
    updated_book_layout = {
        **book_layout,
        "pos": [0.6, -0.1, 0.4],
    }

    refined_layouts = _apply_root_layout_updates_to_descendant_subtrees(
        scene_graph=scene_graph,
        assets_layout=[book_layout, pen_layout],
        updated_root_layouts=[updated_book_layout],
        root_matrices_before_update={
            "book_001": layout_object_to_transform_matrix(book_layout)
        },
    )

    assert _table_on_asset_ids(scene_graph=scene_graph, table_id="table") == {
        "book_001"
    }
    assert np.allclose(
        layout_object_to_transform_matrix(refined_layouts[1])[:3, 3],
        [0.7, -0.05, 0.45],
    )


def test_recovered_table_ground_shift_applies_to_entire_supported_subtree() -> None:
    graph = SceneGraph(
        nodes=[
            SceneGraphNode(object_id="table", parent_id=None),
            SceneGraphNode(object_id="tray", parent_id="table", parent_relation="on"),
            SceneGraphNode(object_id="cup", parent_id="tray", parent_relation="on"),
        ]
    )
    table = {"id": "table", "pos": [1, 2, 3], "rot": [0, 0, 90], "scale": [1, 1, 1]}
    assets = [
        {"id": "tray", "pos": [2, 2, 3], "rot": [0, 40, 0], "scale": [1, 1, 1]},
        {"id": "cup", "pos": [2, 3, 3], "rot": [0, 10, 0], "scale": [1, 1, 1]},
    ]

    lifted = _lift_table_subtrees(
        scene_graph=graph,
        table_layout=table,
        assets_layout=assets,
        table_root_ids={"tray"},
        ground_shift=0.75,
    )

    # A +Y shift in this table's frame is -X in the reconstruction frame.
    np.testing.assert_allclose(
        [item["pos"] for item in lifted], [[1.25, 2, 3], [1.25, 3, 3]]
    )
    np.testing.assert_allclose(
        [item["rot"] for item in lifted], [[0, 40, 0], [0, 10, 0]]
    )
    assert assets[0]["pos"] == [2, 2, 3]


def test_table_roots_align_independently_and_move_only_their_descendants(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeAssetsGroupTableAligner:
        aligned_root_ids: list[str] = []

        def __init__(
            self, *, table_layout, assets_layout, geometry_root, support_top_z
        ) -> None:
            assert geometry_root == tmp_path / "geometry"
            assert len(assets_layout) == 1
            assert support_top_z == 0.42
            self.table_layout = table_layout
            self.root_layout = assets_layout[0]

        def align(self):
            root_id = self.root_layout["id"]
            assert isinstance(root_id, str)
            self.aligned_root_ids.append(root_id)
            vertical_delta_by_id = {"board_001": 0.4, "bottle_001": -0.2}
            return self.table_layout, [
                {
                    **self.root_layout,
                    "pos": [
                        *self.root_layout["pos"][:2],
                        self.root_layout["pos"][2] + vertical_delta_by_id[root_id],
                    ],
                }
            ]

    monkeypatch.setattr(
        "embodichain.gen_sim.scene_engine.pipeline.generation.scene_generation.AssetsGroupTableAligner",
        FakeAssetsGroupTableAligner,
    )
    scene_graph = SceneGraph(
        nodes=[
            SceneGraphNode(object_id="table", parent_id=None),
            SceneGraphNode(
                object_id="board_001", parent_id="table", parent_relation="on"
            ),
            SceneGraphNode(
                object_id="knife_001", parent_id="board_001", parent_relation="on"
            ),
            SceneGraphNode(
                object_id="bottle_001", parent_id="table", parent_relation="on"
            ),
        ]
    )
    table_layout = {
        "id": "table",
        "rot": [0.0, 0.0, 0.0],
        "pos": [0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
    }
    assets_layout = [
        {
            "id": "board_001",
            "rot": [0.0, 0.0, 0.0],
            "pos": [0.1, 0.2, 0.3],
            "scale": [1.0, 1.0, 1.0],
        },
        {
            "id": "knife_001",
            "rot": [0.0, 0.0, 0.0],
            "pos": [0.15, 0.25, 0.35],
            "scale": [1.0, 1.0, 1.0],
        },
        {
            "id": "bottle_001",
            "rot": [0.0, 0.0, 0.0],
            "pos": [0.5, 0.6, 0.8],
            "scale": [1.0, 1.0, 1.0],
        },
    ]

    _, aligned_assets_layout = _align_table_roots_individually(
        scene_graph=scene_graph,
        table_layout=table_layout,
        assets_layout=assets_layout,
        table_root_ids={"board_001", "bottle_001"},
        geometry_root=tmp_path / "geometry",
        support_top_z=0.42,
    )

    aligned_layouts_by_id = {
        str(asset_layout["id"]): asset_layout for asset_layout in aligned_assets_layout
    }
    assert FakeAssetsGroupTableAligner.aligned_root_ids == [
        "board_001",
        "bottle_001",
    ]
    assert np.allclose(aligned_layouts_by_id["board_001"]["pos"], [0.1, 0.2, 0.7])
    assert np.allclose(aligned_layouts_by_id["knife_001"]["pos"], [0.15, 0.25, 0.75])
    assert np.allclose(aligned_layouts_by_id["bottle_001"]["pos"], [0.5, 0.6, 0.6])


def test_on_children_bfs_uses_parent_aabb_then_moves_xz_and_settles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from embodichain.gen_sim.scene_engine.pipeline.generation import scene_generation

    events: list[tuple[str, str]] = []
    inherited_by_id = {
        "pen_001": [0.01, 0.20, 0.03],
        "eraser_001": [0.14, 0.31, 0.08],
    }
    # Parent top + 2 cm clearance + 5 cm from child bottom to its origin.
    release_height_by_id = {"pen_001": 0.22, "eraser_001": 0.33}
    parent_top_by_child_id = {"pen_001": 0.15, "eraser_001": 0.26}
    parent_xy_by_child_id = {
        "pen_001": [[-0.4, -0.4], [0.4, 0.4]],
        "eraser_001": [[0.05, -0.20], [0.25, 0.0]],
    }
    solved_xy_by_id = {"pen_001": [0.12, -0.08], "eraser_001": [0.18, -0.10]}
    settled_by_id = {
        "pen_001": {"pos": [0.15, 0.21, 0.10], "rot": [0.0, 180.0, 0.0]},
        "eraser_001": {"pos": [0.18, 0.30, 0.10], "rot": [0.0, 180.0, 0.0]},
    }

    class FakeParentSurfaceLayoutOptimizer:
        def optimize(self, problem):
            (child_id,) = problem.child_ids
            events.append(("optimize_xz", child_id))
            x, _, z = inherited_by_id[child_id]
            child = problem.assets_by_id[child_id]
            assert child.pos == pytest.approx([x, release_height_by_id[child_id], z])
            assert problem.parent_top_z == pytest.approx(
                parent_top_by_child_id[child_id]
            )
            np.testing.assert_allclose(
                problem.parent_aabb_xy, parent_xy_by_child_id[child_id], atol=1e-9
            )
            if child_id == "eraser_001":
                assert Rotation.from_euler(
                    "xyz", child.rot, degrees=True
                ).as_matrix() == (pytest.approx(np.diag([-1.0, 1.0, -1.0])))
            assert problem.child_seed_xy_by_id[child_id] == pytest.approx([x, -z])
            return {child_id: solved_xy_by_id[child_id]}

        def settle_dynamic_children(self, *, problem, dynamic_child_ids, **_):
            (child_id,) = dynamic_child_ids
            events.append(("settle", child_id))
            x, negative_z = solved_xy_by_id[child_id]
            assert problem.assets_by_id[child_id].pos == pytest.approx(
                [x, release_height_by_id[child_id], -negative_z]
            )
            return {child_id: settled_by_id[child_id]}

    def fake_measure_scene_object_z_up_world_aabb(*, scene_object: SceneObject):
        assert scene_object.pos is not None
        x, y_up, z_up_negative_y = scene_object.pos
        horizontal_half_extent = {
            "book_001": 0.4,
            "pen_001": 0.1,
            "eraser_001": 0.05,
        }[scene_object.id]
        return [
            [
                x - horizontal_half_extent,
                -z_up_negative_y - horizontal_half_extent,
                y_up - 0.05,
            ],
            [
                x + horizontal_half_extent,
                -z_up_negative_y + horizontal_half_extent,
                y_up + 0.05,
            ],
        ]

    monkeypatch.setattr(
        scene_generation,
        "ParentSurfaceLayoutOptimizer",
        FakeParentSurfaceLayoutOptimizer,
    )
    monkeypatch.setattr(
        scene_generation,
        "measure_scene_object_z_up_world_aabb",
        fake_measure_scene_object_z_up_world_aabb,
    )
    scene = Scene(
        objects=[
            SceneObject(
                id="table",
                kind="table",
                category="table",
                name="table",
                description="table",
            ),
            SceneObject(
                id="book_001",
                kind="asset",
                category="book",
                name="book",
                description="book",
            ),
            SceneObject(
                id="pen_001",
                kind="asset",
                category="pen",
                name="pen",
                description="pen",
            ),
            SceneObject(
                id="eraser_001",
                kind="asset",
                category="eraser",
                name="eraser",
                description="eraser",
            ),
        ]
    )
    scene_graph = SceneGraph(
        nodes=[
            SceneGraphNode(object_id="table", parent_id=None),
            SceneGraphNode(
                object_id="book_001", parent_id="table", parent_relation="on"
            ),
            SceneGraphNode(
                object_id="pen_001", parent_id="book_001", parent_relation="on"
            ),
            SceneGraphNode(
                object_id="eraser_001", parent_id="pen_001", parent_relation="on"
            ),
        ]
    )
    table_layout = {
        "id": "table",
        "rot": [0.0, 0.0, 0.0],
        "pos": [0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
    }
    assets_layout = [
        {
            "id": "book_001",
            "rot": [0.0, 0.0, 0.0],
            "pos": [0.0, 0.1, 0.0],
            "scale": [1.0, 1.0, 1.0],
        },
        {
            "id": "pen_001",
            "rot": [0.0, 0.0, 0.0],
            "pos": [0.01, 0.2, 0.03],
            "scale": [1.0, 1.0, 1.0],
        },
        {
            "id": "eraser_001",
            "rot": [0.0, 0.0, 0.0],
            "pos": [0.02, 0.3, 0.05],
            "scale": [1.0, 1.0, 1.0],
        },
    ]

    refined_layout = _refine_on_children_bfs(
        scene=scene,
        scene_graph=scene_graph,
        table_layout=table_layout,
        assets_layout=assets_layout,
        table_root_ids={"book_001"},
        debug_output_root=tmp_path,
    )

    assert events == [
        ("optimize_xz", "pen_001"),
        ("settle", "pen_001"),
        ("optimize_xz", "eraser_001"),
        ("settle", "eraser_001"),
    ]
    eraser_layout = next(
        layout for layout in refined_layout if layout["id"] == "eraser_001"
    )
    assert eraser_layout["pos"] == pytest.approx(settled_by_id["eraser_001"]["pos"])
    assert (tmp_path / "parent_book_001_child_aabb_projection_2d.png").is_file()
    assert (tmp_path / "parent_book_001_child_aabb_optimization_2d.png").is_file()


def test_parent_aabb_projection_uses_the_nearest_valid_child_center() -> None:
    projected_centers, projected_aabbs = _project_child_aabb_centers_into_parent_aabb(
        parent_aabb_xy=[[0.0, 0.0], [1.0, 1.0]],
        child_aabbs_xy_by_id={
            "pen_001": np.array([[-0.4, 0.3], [0.0, 0.7]]),
        },
    )

    assert projected_centers == {"pen_001": [0.2, 0.5]}
    assert np.allclose(projected_aabbs["pen_001"], [[0.0, 0.3], [0.4, 0.7]])


def test_parent_aabb_projection_rejects_a_child_that_cannot_fit() -> None:
    with pytest.raises(ValueError, match="cannot fit inside its support bounds"):
        _project_child_aabb_centers_into_parent_aabb(
            parent_aabb_xy=[[0.0, 0.0], [1.0, 1.0]],
            child_aabbs_xy_by_id={
                "book_001": np.array([[0.0, 0.0], [1.1, 0.2]]),
            },
        )


@pytest.mark.parametrize(
    ("angle", "relation", "region"),
    [
        (180.0, "right_of", "right_back"),
        (90.0, "behind", "left_back"),
        (-90.0, "in_front_of", "right_front"),
    ],
)
def test_world_rotation_updates_saved_graph_directions(angle, relation, region):
    from embodichain.gen_sim.scene_engine.core.scene_graph import SceneGraphRelation

    objects = [
        SceneObject(
            id=uid,
            kind="table" if uid == "table" else "asset",
            category=uid,
            name=uid,
            description=uid,
            pos=[0.0, 0.0, 0.0],
            rot=[0.0, 0.0, 0.0],
            scale=[1.0, 1.0, 1.0],
        )
        for uid in ("table", "a", "b")
    ]
    graph = SceneGraph(
        nodes=[
            SceneGraphNode("table", None),
            SceneGraphNode("a", "table", "on", table_region="left_front"),
            SceneGraphNode("b", "table", "on"),
        ],
        relations=[SceneGraphRelation("a", "left_of", "b")],
    )
    rotate_scene_z_up_world(
        scene=Scene(objects=objects), scene_graph=graph, rotation_degrees=angle
    )
    assert graph.relations[0].relation == relation
    assert graph.nodes[1].table_region == region
    assert graph.nodes[1].parent_relation == "on"


def test_directional_rotation_rejects_unrepresentable_angle_before_pose_change():
    graph = SceneGraph(
        nodes=[
            SceneGraphNode("table", None),
            SceneGraphNode("a", "table", "on", table_region="left_front"),
        ]
    )
    scene = Scene()
    with pytest.raises(ValueError, match="multiple of 90"):
        rotate_scene_z_up_world(scene=scene, scene_graph=graph, rotation_degrees=45)
    assert graph.nodes[1].table_region == "left_front"
