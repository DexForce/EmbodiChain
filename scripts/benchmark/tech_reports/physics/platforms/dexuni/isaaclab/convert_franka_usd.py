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
"""Convert a selected FR3 URDF into a new external USD input snapshot.

Run with the Isaac Lab interpreter:
    python scripts/benchmark/tech_reports/physics/platforms/dexuni/isaaclab/convert_franka_usd.py --urdf FILE --out-dir NEW_DIR

This is input preparation, not a timed simulation. The legacy conversion
normalizes mimic joints; G0/G1 must qualify its dynamics before comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    """Convert once without overwriting an existing input revision."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    urdf = args.urdf.expanduser().resolve(strict=True)
    out = args.out_dir.expanduser().resolve()
    repo = Path(__file__).resolve().parents[7]
    if repo / "scripts" in out.parents or out == repo / "scripts":
        raise ValueError("Generated USD must be outside the source tree")
    out.mkdir(parents=True, exist_ok=False)
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    cfg = UrdfConverterCfg(
        asset_path=str(urdf),
        usd_dir=str(out),
        usd_file_name="fr3_franka_hand.usda",
        fix_base=True,
        merge_fixed_joints=False,
        convert_mimic_joints_to_normal_joints=True,
        joint_drive=None,
        force_usd_conversion=True,
    )
    converter = UrdfConverter(cfg)
    source = {
        "qualification": "unqualified_pilot",
        "urdf": str(urdf),
        "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
        "usd": converter.usd_path,
        "options": {
            "fix_base": True,
            "merge_fixed_joints": False,
            "convert_mimic_joints_to_normal_joints": True,
            "joint_drive": None,
        },
        "files_sha256": {
            p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out.rglob("*"))
            if p.is_file()
        },
    }
    (out / "source.json").write_text(json.dumps(source, indent=2) + "\n")
    print(json.dumps(source, indent=2))


if __name__ == "__main__":
    main()
