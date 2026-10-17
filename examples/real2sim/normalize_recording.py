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

"""Optional real-recording example; no dependency on any reconstruction experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from embodichain.data_pipeline.real2sim import (
    EefPoseSamples,
    Real2SimTrajectory,
    RobotJointLayout,
    SourcePhase,
)
from embodichain.data_pipeline.real2sim.normalization import (
    Real2SimNormalizationCfg,
    normalize_real2sim,
)

__all__: list[str] = []


def main() -> None:
    """Normalize one external NPZ recording as explicitly unverified evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--joint-names", nargs="+", required=True)
    parser.add_argument(
        "--joint-units", nargs="+", choices=["rad", "deg", "m", "mm"], required=True
    )
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--robot-revision", required=True)
    parser.add_argument("--time-unit", choices=["s", "ms", "us", "ns"], required=True)
    parser.add_argument("--control-dt", type=float, required=True)
    parser.add_argument("--positions-key", default="joint_positions")
    parser.add_argument("--timestamps-key", default="timestamps")
    parser.add_argument("--velocities-key", default="joint_velocities")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stop-index", type=int)
    parser.add_argument("--eef-key")
    parser.add_argument(
        "--eef-description",
        type=Path,
        help="Explicit EefPoseSamples frame/TCP/unit description as JSON",
    )
    parser.add_argument(
        "--phases",
        type=Path,
        help="Optional SourcePhase declarations, indexed relative to the selected slice",
    )
    args = parser.parse_args()
    input_path = args.input.resolve()
    output = args.output.resolve()
    if output == input_path.parent or input_path.parent in output.parents:
        parser.error("Output must be outside the original recording directory")
    if bool(args.eef_key) != bool(args.eef_description):
        parser.error("--eef-key and --eef-description must be supplied together")
    with np.load(input_path, allow_pickle=False) as data:
        total = len(data[args.positions_key])
        stop = total if args.stop_index is None else args.stop_index
        if not 0 <= args.start_index < stop <= total:
            parser.error("Invalid source slice")
        section = slice(args.start_index, stop)
        q = data[args.positions_key][section]
        t = data[args.timestamps_key][section]
        v = data[args.velocities_key][section] if args.velocities_key in data else None
        eef = None
        if args.eef_key:
            description = json.loads(args.eef_description.read_text())
            for key in ("reference_to_base", "link_to_tcp"):
                description[key] = np.asarray(description[key], dtype=float)
            eef = EefPoseSamples(poses=data[args.eef_key][section], **description)
    phases = (
        ()
        if args.phases is None
        else tuple(SourcePhase(**item) for item in json.loads(args.phases.read_text()))
    )
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    raw = Real2SimTrajectory(
        source_id="external_recording",
        source_revision=digest,
        unit_id=f"samples_{args.start_index}_{stop}",
        robot_id=args.robot_id,
        robot_revision=args.robot_revision,
        joint_names=tuple(args.joint_names),
        joint_units=tuple(args.joint_units),
        positions=q,
        timestamps=t,
        time_unit=args.time_unit,
        calibration_revision="unverified",
        annotation_revision="unverified",
        phases=phases,
        velocities=v,
        eef=eef,
        provenance={
            "recording_sha256": digest,
            "source_sample_range": [args.start_index, stop],
            "scope": "Recorded channels only; no full robot or physical scene verified",
        },
    )
    factors = {
        "rad": ("rad", 1.0),
        "deg": ("rad", np.pi / 180),
        "m": ("m", 1.0),
        "mm": ("m", 0.001),
    }
    layout = RobotJointLayout(
        args.robot_id,
        args.robot_revision,
        tuple(args.joint_names),
        tuple(factors[u][0] for u in args.joint_units),
        q[0] * np.array([factors[u][1] for u in args.joint_units]),
    )
    unit = normalize_real2sim(
        raw,
        layout,
        Real2SimNormalizationCfg(control_dt=args.control_dt, allow_unverified=True),
    )
    output.mkdir(parents=True, exist_ok=True)
    for path in (output / "normalization.json", output / "trajectory.npz"):
        if path.exists():
            raise FileExistsError(
                f"{path} already exists; choose a fresh output directory"
            )
    report = unit.report_dict()
    report["example_scope"] = (
        "Offline diagnostic only; not a qualified Pick or committed dataset episode"
    )
    (output / "normalization.json").write_text(json.dumps(report, indent=2) + "\n")
    arrays = {
        "raw_qpos": raw.positions,
        "raw_timestamps": raw.timestamps,
        "qpos": unit.positions,
        "qvel": unit.velocities,
        "dt": unit.dt,
        "execution_times_s": unit.time_map.execution_times,
        "source_query_times_s": unit.time_map.source_query_times,
    }
    arrays.update(
        {
            "map_" + name: getattr(unit.time_map, name)
            for name in unit.time_map.__dataclass_fields__
        }
    )
    if raw.velocities is not None:
        arrays["raw_qvel"] = raw.velocities
    if raw.eef is not None:
        arrays["raw_eef"] = raw.eef.poses
        arrays["eef_pose"] = unit.eef_poses
    np.savez_compressed(output / "trajectory.npz", **arrays)
    print(
        json.dumps(
            {
                "source_samples": len(raw.positions),
                "execution_samples": len(unit.positions),
                "input_prerequisites_verified": unit.eligible,
                "revision": unit.revision,
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
