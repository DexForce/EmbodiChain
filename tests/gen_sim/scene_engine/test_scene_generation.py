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

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
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
    _apply_visual_yaws_to_simready_asset_layouts,
    _apply_root_layout_updates_to_descendant_subtrees,
    _optimize_simready_asset_visual_yaws,
    _project_child_aabb_centers_into_parent_aabb,
    _refine_on_children_bfs,
    _table_on_asset_ids,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.visual_yaw_optimizer import (
    VisualYawOptimizer,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.articulated_usdc_utils import (
    _canonicalize_articulated_usdc_bottom_center,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_generation_utils import (
    layout_object_to_transform_matrix,
    transform_matrix_to_layout_object,
)


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

    assert [call[0] for call in client.calls] == [
        "white drawer with a pull handle",
        "black microwave with a hinged door",
    ]
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


def test_table_roots_align_independently_and_move_only_their_descendants(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeAssetsGroupTableAligner:
        aligned_root_ids: list[str] = []

        def __init__(self, *, table_layout, assets_layout, geometry_root) -> None:
            assert geometry_root == tmp_path / "geometry"
            assert len(assets_layout) == 1
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


def test_on_children_bfs_refines_every_non_table_parent(
    monkeypatch,
    tmp_path,
) -> None:
    class FakeParentSurfaceLayoutOptimizer:
        refined_child_ids_by_call: list[list[str]] = []

        def optimize(self, problem):
            self.refined_child_ids_by_call.append(problem.child_ids)
            return problem.child_seed_xy_by_id

        def settle_dynamic_children(self, **_: object):
            return {}

    def fake_measure_scene_object_z_up_world_aabb(*, scene_object: SceneObject):
        assert scene_object.pos is not None
        x, y_up, z_up_negative_y = scene_object.pos
        return [
            [x - 0.05, -z_up_negative_y - 0.05, y_up - 0.05],
            [x + 0.05, -z_up_negative_y + 0.05, y_up + 0.05],
        ]

    def fake_place_on_support(
        *,
        scene_object: SceneObject,
        support_region_z: float,
        center_xy: list[float],
        clearance_m: float,
    ) -> None:
        scene_object.pos = [
            center_xy[0],
            support_region_z + clearance_m,
            -center_xy[1],
        ]

    monkeypatch.setattr(
        "embodichain.gen_sim.scene_engine.pipeline.generation.scene_generation.ParentSurfaceLayoutOptimizer",
        FakeParentSurfaceLayoutOptimizer,
    )
    monkeypatch.setattr(
        "embodichain.gen_sim.scene_engine.pipeline.generation.scene_generation.measure_scene_object_z_up_world_aabb",
        fake_measure_scene_object_z_up_world_aabb,
    )
    monkeypatch.setattr(
        "embodichain.gen_sim.scene_engine.pipeline.generation.scene_generation.update_scene_object_y_up_pose_from_z_up_support",
        fake_place_on_support,
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
            "pos": [0.01, 0.2, 0.0],
            "scale": [1.0, 1.0, 1.0],
        },
        {
            "id": "eraser_001",
            "rot": [0.0, 0.0, 0.0],
            "pos": [0.02, 0.3, 0.0],
            "scale": [1.0, 1.0, 1.0],
        },
    ]

    _refine_on_children_bfs(
        scene=scene,
        scene_graph=scene_graph,
        table_layout=table_layout,
        assets_layout=assets_layout,
        table_root_ids={"book_001"},
        debug_output_root=tmp_path,
    )

    assert FakeParentSurfaceLayoutOptimizer.refined_child_ids_by_call == [
        ["pen_001"],
        ["eraser_001"],
    ]
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
    with pytest.raises(ValueError, match="cannot fit inside its parent AABB"):
        _project_child_aabb_centers_into_parent_aabb(
            parent_aabb_xy=[[0.0, 0.0], [1.0, 1.0]],
            child_aabbs_xy_by_id={
                "book_001": np.array([[0.0, 0.0], [1.1, 0.2]]),
            },
        )
