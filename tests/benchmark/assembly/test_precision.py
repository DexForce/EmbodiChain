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
"""Assembly precision runner, reference options and endpoint measurements."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from embodichain.toolkits.assembly import AssemblyAsset, import_mjcf

SOURCE = (
    Path(__file__).resolve().parents[2] / "fixtures/assembly/body.xml"
).read_text()


def test_usd_precision_detects_changed_primitive_dimensions(tmp_path) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom
    from embodichain_benchmarks.assembly.precision import compare_usd_geometry

    asset = import_mjcf(SOURCE)
    path = asset.write_usd(tmp_path / "wrong_radius.usda")
    stage = Usd.Stage.Open(str(path))
    cylinder = UsdGeom.Cylinder(stage.GetPrimAtPath("/Assembly/part/geom_1"))
    cylinder.GetRadiusAttr().Set(0.012)  # Source radius is 0.010 m.
    stage.GetRootLayer().Save()

    measured = compare_usd_geometry(asset, path)

    assert not measured["passed"]
    assert measured["geometry_error_m"] >= 0.0019


def test_precision_host_matches_source_floor_and_disables_missing_floor() -> None:
    pytest.importorskip("mujoco")
    from embodichain_benchmarks.assembly.precision import (
        PrecisionRunCfg,
        _source_floor_attributes,
    )

    cfg = PrecisionRunCfg(backend="newton", source_contact=True)
    enabled, attrs = _source_floor_attributes(import_mjcf(SOURCE), cfg)
    assert not enabled and attrs is None
    floor_source = SOURCE.replace(
        "<worldbody>",
        '<worldbody><geom name="floor" type="plane" size="1 1 .01" friction="1 .01 .001" solref=".002 1" solimp=".99 .999 .001 .5 2"/>',
    )
    asset = import_mjcf(floor_source)
    enabled, attrs = _source_floor_attributes(asset, cfg)
    assert len(asset.bodies) == 1  # Ground remains a host-owned scene property.
    assert enabled
    assert attrs.material_props.dynamic_friction == pytest.approx(1)
    assert attrs.material_props.torsional_friction == pytest.approx(0.01)
    assert attrs.collision_props.solref == pytest.approx([0.002, 1])
    assert attrs.collision_props.solimp == pytest.approx([0.99, 0.999, 0.001, 0.5, 2])
    assert attrs.collision_props.contact_offset == 0


@pytest.mark.parametrize("backend", ["unknown", "fake"])
def test_invalid_probe_backend_is_not_reported_as_skipped(backend) -> None:
    from embodichain_benchmarks.assembly.precision import (
        PrecisionRunCfg,
        run_passive_probe,
    )

    with pytest.raises(ValueError, match="Unknown precision backend"):
        run_passive_probe(AssemblyAsset(), PrecisionRunCfg(backend=backend))


@pytest.mark.parametrize("scale,bottom", [(1.0, 0.0), (2.0, 0.004)])
def test_peg_drop_distinguishes_seating_from_rim_and_outside_falls(
    scale: float, bottom: float
) -> None:
    """A tiny penetration after the spawn gap must not become drop success."""
    from embodichain_benchmarks.assembly.precision import PrecisionRunCfg, _score

    # Pinned peg() defaults: 10 mm plate, 30 mm insertion section, 5 mm
    # spawn gap. The blind-hole bottom and all dimensions scale together.
    thickness, length, gap = np.array([0.01, 0.03, 0.005]) * scale
    bottom *= scale
    plate = np.array([0.0, 0.0, bottom + thickness / 2])
    initial_peg = np.array([0.0, 0.0, bottom + thickness + gap + length / 2])
    seated = [0.0, 0.0, bottom + length / 2]
    rim_stop = [0.02 * scale, 0.0, bottom + thickness + length / 2 - 1e-7]
    outside_fall = [0.05 * scale, 0.0, bottom + length / 2]
    asset = AssemblyAsset(config={"kind": "peg", "scale": scale})

    measured = _score(
        asset,
        {"peg": initial_peg, "plate": plate},
        {"peg": np.array([seated, rim_stop, outside_fall]), "plate": plate},
        PrecisionRunCfg(),
    )

    # Every endpoint fell farther than 5 mm, but only the centered peg reached
    # the bottom of the bore. This expected count is independent of scoring.
    assert measured["success_rate"] == pytest.approx(1 / 3)
    assert measured["required_insertion_depth_m"] == pytest.approx(thickness)


def test_many_pegs_scores_each_translated_cell_and_rejects_rim_or_floor_passage() -> (
    None
):
    """Heterogeneous cells cannot use one global plate position or thickness."""
    from embodichain_benchmarks.assembly._many_metrics import _score_many_pegs

    cells = [
        ([-0.3, 0.3, 0.005], 0.01, 0.03),
        ([0.3, -0.3, 0.01], 0.02, 0.06),
    ]
    assemblies, initial, final = [], {}, {}
    for index, (plate, thickness, length) in enumerate(cells):
        plate_name, peg_name = f"p{index}_plate", f"p{index}_peg"
        assemblies.append(
            {
                "index": index,
                "plate_body": plate_name,
                "peg_body": peg_name,
                "config": {"plate_thickness": thickness, "peg_length": length},
            }
        )
        initial[plate_name] = final[plate_name] = np.array(plate)
        initial[peg_name] = np.array(
            [plate[0], plate[1], thickness + 0.005 + length / 2]
        )
        # Four independent endpoint cases: fully seated, on the rim, fallen
        # outside the plate, and tunneled through the floor.
        final[peg_name] = np.array(
            [
                [plate[0], plate[1], length / 2],
                [plate[0] + 0.02, plate[1], thickness + length / 2],
                [plate[0] + 0.10, plate[1], length / 2],
                [plate[0], plate[1], length / 2 - 0.01],
            ]
        )
    measured = _score_many_pegs(
        AssemblyAsset(config={"kind": "many_pegs", "assemblies": assemblies}),
        initial,
        final,
    )

    assert measured["assembly_count"] == 2
    assert measured["success_rate"] == pytest.approx(0.25)
    assert [row["success_rate"] for row in measured["per_assembly"]] == [0.25, 0.25]
    assert [row["required_insertion_depth_m"] for row in measured["per_assembly"]] == [
        0.01,
        0.02,
    ]


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"backend": "default", "solver": "cg"}, "Default does not accept"),
        ({"warp_reference_initialization": "unknown"}, "initialization must be"),
        (
            {
                "backend": "mujoco_cpu",
                "warp_reference_initialization": "warp_recomputed",
            },
            "only for MuJoCo Warp references",
        ),
        (
            {"backend": "newton", "warp_reference_initialization": "warp_recomputed"},
            "only for MuJoCo Warp references",
        ),
        ({"backend": "default", "solver_tolerance": 1e-6}, "Default does not accept"),
        ({"backend": "default", "enable_multiccd": False}, "Default does not accept"),
        ({"solver_tolerance": -1.0}, "tolerance must be finite and non-negative"),
        (
            {"solver_tolerance": float("inf")},
            "tolerance must be finite and non-negative",
        ),
        (
            {"solver_tolerance": float("nan")},
            "tolerance must be finite and non-negative",
        ),
        ({"enable_multiccd": "false"}, "enable_multiccd must be a bool"),
        ({"backend": "mujoco_cpu", "use_mujoco_cpu": True}, "Newton CPU integration"),
        ({"backend": "newton", "use_mujoco_cpu": True, "num_envs": 2}, "one world"),
    ],
)
def test_precision_solver_comparisons_reject_ignored_or_unsupported_settings(
    settings, message
) -> None:
    from embodichain_benchmarks.assembly.precision import (
        PrecisionRunCfg,
        run_passive_probe,
    )

    # These errors must be diagnosed before importing or starting a simulator.
    with pytest.raises(ValueError, match=message):
        run_passive_probe(AssemblyAsset(), PrecisionRunCfg(**settings))


@pytest.mark.parametrize(
    ("source_multiccd", "overrides", "expected_tolerance", "expected_multiccd"),
    [
        (True, {}, 7e-9, True),
        (False, {}, 7e-9, False),
        (False, {"solver_tolerance": 0.0, "enable_multiccd": True}, 0.0, True),
        (True, {"solver_tolerance": 1e-6, "enable_multiccd": False}, 1e-6, False),
    ],
)
def test_precision_reference_preserves_source_options_unless_explicitly_overridden(
    source_multiccd, overrides, expected_tolerance, expected_multiccd
) -> None:
    """Unspecified calibration knobs must not rewrite the recorded source baseline."""
    mujoco = pytest.importorskip("mujoco")
    from embodichain_benchmarks.assembly.precision import PrecisionRunCfg, _run_mujoco

    flag = "enable" if source_multiccd else "disable"
    source = SOURCE.replace(
        "<mujoco>",
        f'<mujoco><option tolerance="7e-9"><flag multiccd="{flag}"/></option>',
    )
    asset = import_mjcf(source, config={"kind": "options_inspection"})

    class OptionsInspected(Exception):
        pass

    def inspect(model, data, step):
        assert step == 0
        assert model.opt.tolerance == expected_tolerance
        multiccd_disabled = bool(
            model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_MULTICCD
        )
        assert multiccd_disabled is not expected_multiccd
        raise OptionsInspected

    with pytest.raises(OptionsInspected):
        _run_mujoco(asset, PrecisionRunCfg(**overrides), on_step=inspect)


def test_failed_precision_measurements_preserve_json_and_failure_reason(
    monkeypatch,
) -> None:
    from embodichain_benchmarks.assembly import precision

    # Real sub-micron meshes can fail backend cooking; the matrix must retain
    # that diagnostic instead of terminating without a result file.
    def unsupported(*args):
        raise ValueError("planar mesh collider")

    monkeypatch.setattr(precision, "_run_mujoco", unsupported)
    result = precision.run_passive_probe(AssemblyAsset(), precision.PrecisionRunCfg())
    assert result["status"] == "failed"
    assert "planar mesh collider" in result["reason"]

    def nonfinite(*args):
        return {
            "status": "failed",
            "metrics": {"finite": False, "distance": float("nan")},
            "final_state": {"peg": [0.0, float("inf"), 0.0]},
        }

    monkeypatch.setattr(precision, "_run_mujoco", nonfinite)
    result = precision.run_passive_probe(AssemblyAsset(), precision.PrecisionRunCfg())
    json.dumps(result, allow_nan=False)
    assert result["status"] == "failed"
    assert result["metrics"]["finite"] is False
    assert result["metrics"]["distance"] is None
    assert result["final_state"]["peg"] == [0.0, None, 0.0]
    assert result["nonfinite_measurement_count"] == 2


def test_compiled_robot_tree_preserves_nonzero_joint_frame(tmp_path: Path) -> None:
    import xml.etree.ElementTree as ET
    import torch

    mujoco = pytest.importorskip("mujoco")
    pk = pytest.importorskip("pytorch_kinematics")
    from embodichain_benchmarks.assembly.robot_reference import _append_model

    source = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="base"><geom type="box" size=".1 .1 .1"/>
       <body name="arm" pos=".1 .2 .3" euler="15 20 5">
        <joint name="joint" axis="0 1 0" pos=".01 .02 .03" range="-90 90"/>
        <geom type="box" pos=".02 0 .1" size=".02 .03 .1" mass="1"/>
        <body name="tool" pos=".1 0 .2"><geom type="sphere" size=".01" mass=".01"/></body>
       </body>
      </body></worldbody></mujoco>""")
    urdf = ET.Element("robot", name="reference")
    _append_model(urdf, source, tmp_path)
    chain = pk.build_serial_chain_from_urdf(
        ET.tostring(urdf, encoding="unicode"),
        end_link_name="tool",
        root_link_name="base",
    ).to(dtype=torch.float64)
    data = mujoco.MjData(source)
    for angle in [-0.7, 0, 0.65]:
        data.qpos[0] = angle
        mujoco.mj_forward(source, data)
        actual = (
            chain.forward_kinematics(torch.tensor([[angle]], dtype=torch.float64))
            .get_matrix()[0]
            .numpy()
        )
        tool_id = mujoco.mj_name2id(source, mujoco.mjtObj.mjOBJ_BODY, "tool")
        expected = np.eye(4)
        expected[:3, 3] = data.xpos[tool_id]
        expected[:3, :3] = data.xmat[tool_id].reshape(3, 3)
        np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-7)
