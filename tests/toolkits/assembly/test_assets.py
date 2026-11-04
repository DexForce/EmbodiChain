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
"""Portable assembly geometry, ownership and standalone format regressions."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from embodichain.toolkits.assembly import AssemblyAsset, import_mjcf

SOURCE = (
    Path(__file__).resolve().parents[2] / "fixtures/assembly/body.xml"
).read_text()


@pytest.mark.parametrize("export_usd", [False, True])
def test_portable_formats_do_not_require_simulation_or_generation(
    tmp_path: Path, export_usd: bool
) -> None:
    if export_usd:
        pytest.importorskip("pxr")
    program = textwrap.dedent("""
        import importlib.abc
        from pathlib import Path
        import sys

        class RuntimeBlocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                blocked = (
                    "embodichain.gen_sim", "embodichain.lab",
                    "embodichain_tasks", "embodichain_benchmarks",
                    "mjfit", "mujoco", "dexsim",
                )
                if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
                    raise ImportError("Unexpected runtime dependency: " + fullname)

        sys.meta_path.insert(0, RuntimeBlocker())
        from embodichain.toolkits.assembly import (
            AssemblyAsset, AssemblyBody, AssemblyGeometry,
        )

        asset = AssemblyAsset(bodies=[AssemblyBody(
            name="part", geometries=[AssemblyGeometry(
                kind="mesh", vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                triangles=[[0, 1, 2]],
            )],
        )])
        folder = Path(sys.argv[1])
        restored = AssemblyAsset.load(asset.save(folder / "asset.json"))
        assert restored.content_hash == asset.content_hash
        assert restored.bodies[0].geometries[0].triangles == [[0, 1, 2]]
        if sys.argv[2] == "True":
            restored.write_usd(folder / "asset.usda")
        """)
    result = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path), str(export_usd)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_import_preserves_authored_mass_and_rotated_inertia() -> None:
    pytest.importorskip("mujoco")
    asset = import_mjcf(SOURCE)
    body = asset.bodies[0]
    assert body.mass == 2.0
    np.testing.assert_allclose(body.com, [0.001, 0.002, 0.003], atol=1e-15)
    # MuJoCo's default Euler sequence is intrinsic XYZ, independently formed
    # here rather than reading the converter's compiled inertia back.
    rotation = Rotation.from_euler("XYZ", [0.1, 0.2, 0.3]).as_matrix()
    expected = rotation @ np.diag([0.01, 0.02, 0.025]) @ rotation.T
    np.testing.assert_allclose(body.inertia, expected, atol=1e-14)
    assert len(body.geometries) == 2
    assert body.geometries[1].kind == "cylinder"


def test_compiled_mesh_recentering_preserves_body_local_vertices() -> None:
    """MuJoCo's mesh principal frame must not be applied a second time."""
    pytest.importorskip("mujoco")
    source_vertices = np.array(
        [
            [0.1, 0.0, 0.0],
            [0.11, 0.0, 0.0],
            [0.1, 0.03, 0.0],
            [0.1, 0.0, 0.04],
        ]
    )
    xml = """<mujoco><compiler angle="radian"/>
    <asset><mesh name="offset_tetra" vertex=".1 0 0 .11 0 0 .1 .03 0 .1 0 .04"
      face="0 2 1 0 1 3 0 3 2 1 2 3"/></asset>
    <worldbody><body name="part"><freejoint name="joint"/>
      <geom name="offset_mesh" type="mesh" mesh="offset_tetra" pos=".02 -.03 .04" euler=".2 -.3 .4"/>
    </body></worldbody></mujoco>"""
    geom = import_mjcf(xml).bodies[0].geometries[0]
    expected = source_vertices @ Rotation.from_euler(
        "XYZ", [0.2, -0.3, 0.4]
    ).as_matrix().T + [0.02, -0.03, 0.04]
    actual = (
        np.asarray(geom.vertices) @ Rotation.from_quat(geom.quaternion).as_matrix().T
        + geom.position
    )
    # Convex compilation may permute vertices; compare their body-local set.
    pairwise_distance = np.linalg.norm(actual[:, None] - expected[None], axis=-1)
    assert np.max(np.min(pairwise_distance, axis=1)) < 1e-8
    assert np.max(np.min(pairwise_distance, axis=0)) < 1e-8


def test_manifest_roundtrip_does_not_import_generator_and_detects_tampering(
    tmp_path, monkeypatch
) -> None:
    pytest.importorskip("mujoco")
    asset = import_mjcf(SOURCE)
    target = asset.save(tmp_path / "asset.json")
    monkeypatch.setitem(sys.modules, "mjfit", None)
    monkeypatch.setitem(sys.modules, "mujoco", None)
    restored = AssemblyAsset.load(target)
    assert restored.content_hash == asset.content_hash
    data = json.loads(target.read_text())
    data["asset"]["bodies"][0]["mass"] = 3.0
    target.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum"):
        AssemblyAsset.load(target)


def test_usd_preserves_body_ownership_vertices_and_mass(tmp_path) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("pxr")
    from embodichain_benchmarks.assembly.precision import compare_usd_geometry

    asset = import_mjcf(SOURCE)
    path = asset.write_usd(tmp_path / "part.usda")
    measured = compare_usd_geometry(asset, path)
    assert measured["passed"]
    assert measured["colliders"] == 2
    assert measured["mass_relative_error"] < 1e-6
    assert measured["inertia_relative_error"] < 1e-6


def test_usd_static_compound_has_one_body_not_one_body_per_collider(tmp_path) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("pxr")
    from dexsim.kit.usd import parse_usd

    asset = import_mjcf(SOURCE.replace('<freejoint name="part_joint"/>', ""))
    path = asset.write_usd(tmp_path / "static.usda")
    parsed = parse_usd(str(path), root_name="test")
    assert len(parsed.mesh_objects) == 1
    assert len(parsed.mesh_objects[0].collisions) == 2


def test_unsupported_equality_constraints_fail_before_conversion() -> None:
    pytest.importorskip("mujoco")
    xml = SOURCE.replace(
        "</mujoco>", '<equality><weld body1="part"/></equality></mujoco>'
    )
    with pytest.raises(ValueError, match="equality"):
        import_mjcf(xml)


def test_nonstandard_collision_masks_cannot_silently_change_filtering() -> None:
    pytest.importorskip("mujoco")
    xml = SOURCE.replace('name="piece_a"', 'name="piece_a" contype="2" conaffinity="2"')
    with pytest.raises(ValueError, match="collision masks"):
        import_mjcf(xml)


def test_usd_mesh_keeps_sharp_face_normals_in_geometry_local_frame(tmp_path) -> None:
    """USD shading must retain hard edges without applying the geom pose twice."""
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom

    from embodichain.toolkits.assembly import AssemblyBody, AssemblyGeometry

    geometry = AssemblyGeometry(
        vertices=[[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]],
        triangles=[[0, 1, 2], [0, 3, 1]],
        position=[1, 2, 3],
        quaternion=[0, 0, np.sqrt(0.5), np.sqrt(0.5)],
    )
    body = AssemblyBody(name="hard_corner", geometries=[geometry])
    path = AssemblyAsset(bodies=[body]).write_usd(tmp_path / "corner.usda")
    stage = Usd.Stage.Open(str(path))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Assembly/hard_corner/geom_0"))

    np.testing.assert_allclose(mesh.GetPointsAttr().Get(), geometry.vertices)
    np.testing.assert_array_equal(
        mesh.GetFaceVertexIndicesAttr().Get(), [0, 1, 2, 0, 3, 1]
    )
    assert mesh.GetNormalsInterpolation() == UsdGeom.Tokens.faceVarying
    normals = np.asarray(mesh.GetNormalsAttr().Get())
    # The two source planes face +Z and +Y; their common edge stays sharp.
    np.testing.assert_allclose(normals, [[0, 0, 1]] * 3 + [[0, 1, 0]] * 3)
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim())
    world_normals = [
        transform.TransformDir(Gf.Vec3d(*normal.tolist())) for normal in normals
    ]
    # Geometry rotation, not authored normal data, maps +Y to world -X.
    np.testing.assert_allclose(
        world_normals, [[0, 0, 1]] * 3 + [[-1, 0, 0]] * 3, atol=1e-6
    )


def test_many_pegs_preserves_official_sampling_prefixes_and_grid_placements(
    tmp_path,
) -> None:
    """A grid is distinct bodies with their own source geometry and inertia."""
    mujoco = pytest.importorskip("mujoco")
    mjfit = pytest.importorskip("mjfit")
    from embodichain.toolkits.assembly import build_many_pegs

    asset = build_many_pegs(2, seed=0, family="primitive", cache_dir=tmp_path)
    configs = mjfit.sample_pegs(2, seed=0, family="primitive")
    source = mjfit.grid(
        [mjfit.build(config, standalone=False) for config in configs], spacing=0.3
    )
    source.compile()
    model = mujoco.MjModel.from_xml_string(source.to_xml())

    assert [item["config"] for item in asset.config["assemblies"]] == configs
    assert [body.name for body in asset.bodies] == [
        "p0_plate",
        "p0_peg",
        "p1_plate",
        "p1_peg",
    ]
    # Two cells in the source's square grid occupy the top row, not two
    # coincident rigid facades or independent simulated arenas.
    np.testing.assert_allclose(
        [item["position"] for item in asset.config["assemblies"]],
        [[-0.15, 0.15, 0], [0.15, 0.15, 0]],
        atol=1e-15,
    )
    assert sum(len(body.geometries) for body in asset.bodies) == model.ngeom - 1
    for body in asset.bodies:
        source_body = model.body(body.name)
        assert body.mass == source_body.mass[0]
        np.testing.assert_allclose(body.position, source_body.pos, atol=1e-15)
        assert body.joint.kind == ("free" if body.name.endswith("peg") else "fixed")
    assert AssemblyAsset.load(next(tmp_path.glob("*.json"))).content_hash == (
        asset.content_hash
    )

    # The primitive sampler does not accept max_error; it belongs to the
    # public curved shape spec. Its tighter curve approximation must
    # produce more collider pieces while retaining the same sampled outline.
    precise = build_many_pegs(
        2, seed=0, family="primitive", geometry_parameters={"max_error": 1e-5}
    )
    assert precise.config["geometry_parameters"] == {"max_error": 1e-5}
    assert precise.config["assemblies"][0]["sampled_config"] == configs[0]
    assert precise.config["assemblies"][0]["config"]["hole"] == {
        **configs[0]["hole"],
        "max_error": 1e-5,
    }
    assert sum(len(body.geometries) for body in precise.bodies) > model.ngeom - 1
