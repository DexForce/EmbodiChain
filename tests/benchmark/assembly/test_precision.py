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
import argparse
from pathlib import Path
import subprocess
import sys

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


@pytest.mark.parametrize("source_contact", [False, True])
def test_precision_host_matches_source_floor_and_disables_missing_floor(
    source_contact: bool,
) -> None:
    pytest.importorskip("mujoco")
    from embodichain_benchmarks.assembly.precision import (
        PrecisionRunCfg,
        _source_floor_attributes,
    )

    cfg = PrecisionRunCfg(backend="newton", source_contact=source_contact)
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
    if source_contact:
        assert attrs.material_props.solref == pytest.approx([0.002, 1])
        assert attrs.material_props.solimp == pytest.approx(
            [0.99, 0.999, 0.001, 0.5, 2]
        )
        assert attrs.material_props.solmix == pytest.approx(1)
    else:
        assert attrs.material_props.solref is None
        assert attrs.material_props.solimp is None
        assert attrs.material_props.solmix is None
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
    over_inserted = [0.0, 0.0, bottom + length / 2 - thickness]
    below_floor = [0.0, 0.0, bottom - length]
    asset = AssemblyAsset(config={"kind": "peg", "scale": scale})

    measured = _score(
        asset,
        {"peg": initial_peg, "plate": plate},
        {
            "peg": np.array(
                [seated, rim_stop, outside_fall, over_inserted, below_floor]
            ),
            "plate": plate,
        },
        PrecisionRunCfg(),
    )

    # Every endpoint fell farther than 5 mm, but only the centered peg reached
    # the bottom of the bore. This expected count is independent of scoring.
    assert measured["success_rate"] == pytest.approx(1 / 5)
    assert measured["required_insertion_depth_m"] == pytest.approx(thickness)


def _matrix_args(output: Path, manifest: Path) -> argparse.Namespace:
    return argparse.Namespace(
        output=str(output),
        configs=[str(manifest)],
        clearances=None,
        backends=["mujoco_cpu"],
        routes=["direct"],
        num_envs=[1],
        timesteps=[0.001],
        duration=0.002,
        contact_offset=0.0,
        rest_offset=0.0,
        solver_iterations=100,
        contact_capacity=4096,
        backend_contacts=False,
        solver=None,
        use_mujoco_cpu=False,
        solver_tolerance=None,
        multiccd=None,
        warp_reference_initialization="cpu_compiled",
        probe="passive",
        directions=4,
        cell_timeout=60,
    )


def test_usd_comparison_registers_schemas_in_a_fresh_process(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("pxr")
    pytest.importorskip("dexsim.kit.usd")
    asset = import_mjcf(SOURCE)
    manifest = asset.save(tmp_path / "asset.json")
    usd = asset.write_usd(tmp_path / "asset.usda")
    program = """
import json
import sys
from embodichain.toolkits.assembly import AssemblyAsset
from embodichain_benchmarks.assembly.precision import compare_usd_geometry
result = compare_usd_geometry(AssemblyAsset.load(sys.argv[1]), sys.argv[2])
assert result['passed'], result
print('FRESH_USD_COMPARISON_PASSED')
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(manifest), str(usd)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FRESH_USD_COMPARISON_PASSED" in result.stdout


def test_fresh_matrix_records_usd_failure_and_runs_cpu_cells(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("pxr")
    # Include a fixed plate: current SDKs can fail its ownership check, while
    # the independent CPU reference remains meaningful and must still run.
    source = SOURCE.replace('name="part"', 'name="peg"').replace(
        "<worldbody>",
        '<worldbody><body name="plate" pos="0 0 -1"><geom type="box" size=".02 .02 .005"/></body>',
    )
    asset = import_mjcf(source, config={"kind": "peg"})
    manifest = asset.save(tmp_path / "asset.json")
    output = tmp_path / "results"
    script = (
        Path(__file__).resolve().parents[3] / "scripts/benchmark/assembly/precision.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--configs",
            str(manifest),
            "--backends",
            "mujoco_cpu",
            "--duration",
            "0.002",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    raw = next(output.glob("run_*/assembly_precision_*.json"))
    report = json.loads(raw.read_text())
    assert len(report["cells"]) == 1
    cell = report["cells"][0]
    assert cell["asset_sha256"] == asset.content_hash
    assert cell["metrics"]["finite"] is True
    assert cell["backend"] == "mujoco_cpu"
    assert "schemas were registered" not in report["geometry"][0].get("reason", "")
    assert raw.with_suffix(".md").is_file()


@pytest.mark.parametrize("outcome", ["valid", "crash", "wrong_asset", "wrong_config"])
def test_matrix_isolates_runs_and_checks_worker_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("pxr")
    from scripts.benchmark.assembly import precision as matrix

    asset = import_mjcf(SOURCE)
    manifest = asset.save(tmp_path / "asset.json")
    output = tmp_path / "results"
    output.mkdir()
    old = {
        "backend": "mujoco_cpu",
        "route": "direct",
        "status": "passed",
        "asset_sha256": "previous_run",
    }
    (output / "cell_0.json").write_text(json.dumps(old))

    def unavailable_geometry(*args):
        raise ValueError("ownership unavailable")

    original_run = subprocess.run

    def run_worker(command, **kwargs):
        if "--cell-output" not in command:
            return original_run(command, **kwargs)
        if outcome != "crash":
            config = json.loads(
                Path(command[command.index("--cell-config") + 1]).read_text()
            )
            config.pop("constrained_probe")
            config.pop("num_directions")
            if outcome == "wrong_config":
                config["duration"] = 123.0
            result = {
                "backend": "mujoco_cpu",
                "route": "direct",
                "status": "passed",
                "asset_sha256": (
                    "other_asset" if outcome == "wrong_asset" else asset.content_hash
                ),
                "config": config,
            }
            Path(command[command.index("--cell-output") + 1]).write_text(
                json.dumps(result)
            )
        return subprocess.CompletedProcess(
            command,
            139 if outcome == "crash" else 0,
            "",
            "worker crashed" if outcome == "crash" else "",
        )

    monkeypatch.setattr(matrix, "compare_usd_geometry", unavailable_geometry)
    monkeypatch.setattr(matrix.subprocess, "run", run_worker)
    paths = [
        matrix.run_all_benchmarks(_matrix_args(output, manifest)) for _ in range(2)
    ]
    assert paths[0].parent != paths[1].parent
    assert json.loads((output / "cell_0.json").read_text()) == old
    for path in paths:
        report = json.loads(path.with_suffix(".json").read_text())
        assert report["geometry"][0]["passed"] is False
        assert report["geometry"][0]["reason"] == "ownership unavailable"
        assert "unmeasured" in path.read_text()
        assert report["cells"][0]["status"] == (
            "passed" if outcome == "valid" else "failed"
        )


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
