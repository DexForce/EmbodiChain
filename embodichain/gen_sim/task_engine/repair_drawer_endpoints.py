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

"""Opt-in nearest-zero endpoint annotation for generated drawer exports.

Run with ``python -m embodichain.gen_sim.task_engine.repair_drawer_endpoints``.
This heuristic does not certify which endpoint physically closes a drawer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile

from .endpoint_annotations import (
    MANIFEST_SCHEMA,
    _prismatic_joints,
    migrate_endpoint_manifest,
    write_endpoint_report,
)

__all__: list[str] = []


def _manifest(root: Path) -> dict:
    assets = {}
    for config_path in sorted(root.rglob("scene_export/scene_config.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        for item in config.get("articulation", []):
            label = " ".join(
                str(item.get(key, "")).lower()
                for key in ("uid", "name", "description", "category")
            )
            if not any(
                word in label
                for word in ("drawer", "cabinet", "chest", "storage", "organizer")
            ):
                continue
            path = (config_path.parent / item["fpath"]).resolve()
            relative = path.relative_to(root).as_posix()
            if relative in assets:
                continue
            if path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
                raise ValueError(f"Expected a USD drawer asset: {path}")
            try:
                joints = _prismatic_joints(path)
            except ValueError as error:
                if "has no prismatic joints:" in str(error):
                    continue
                raise
            decisions = []
            for name, joint in joints.items():
                closed = joint["closed_position"]
                if closed is None:
                    lower, upper = joint["limits"]
                    if math.isclose(abs(lower), abs(upper), abs_tol=1e-6):
                        raise ValueError(
                            f"Ambiguous nearest-zero endpoint: {path}:{name}"
                        )
                    closed = min((lower, upper), key=abs)
                    evidence = (
                        "Heuristic nearest-zero limit; NOT semantically reviewed."
                    )
                else:
                    evidence = "Preserved existing authored closed endpoint."
                decisions.append(
                    {"joint": name, "closed_position": closed, "evidence": [evidence]}
                )
            assets[relative] = {
                "path": relative,
                "expected_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "joints": decisions,
            }
    return {"schema_version": MANIFEST_SCHEMA, "assets": list(assets.values())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write missing endpoints using the nearest-zero heuristic; back up first.",
    )
    args = parser.parse_args(argv)
    root = Path(args.asset_root).expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Asset root is not a directory: {root}")
    manifest = _manifest(root)
    if not manifest["assets"]:
        print("No matching drawer assets with prismatic joints found; nothing changed.")
        return 0
    # Keep backups outside the scan root, and never reuse an earlier report.
    audit = Path(tempfile.mkdtemp(prefix=f"{root.name}-endpoints-", dir=root.parent))
    write_endpoint_report(audit / "manifest.json", manifest)
    print(
        f"Heuristic annotation (not semantic verification). Audit directory: {audit}",
        flush=True,
    )
    report = migrate_endpoint_manifest(
        asset_root=root,
        manifest=manifest,
        apply=args.apply,
        backup_root=audit / "backup" if args.apply else None,
    )
    write_endpoint_report(audit / "report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
