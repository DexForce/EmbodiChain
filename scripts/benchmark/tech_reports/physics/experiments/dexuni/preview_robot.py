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
"""Explicit robot asset selection for isolated, untimed report previews."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

__all__ = ["preview_robot"]


def preview_robot(default: Path, args: argparse.Namespace) -> Path:
    """Resolve an optional preview robot and record all referenced mesh hashes.

    Args:
        default: Frozen robot used by the original campaign.
        args: Worker arguments, including robot_urdf and a separate out directory.

    Returns:
        Selected URDF path. Frozen measurement selection is unchanged.

    Raises:
        ValueError: An asset override would enter measurement or existing output.
    """
    selected = getattr(args, "robot_urdf", None)
    if selected is None:
        return default
    if (
        not args.out
        or getattr(args, "measure", False)
        or getattr(args, "pilot_performance", False)
    ):
        raise ValueError("--robot-urdf requires --out and an untimed preview")
    selected = selected.expanduser().resolve(strict=True)
    output = args.out.expanduser().resolve()
    if (output / "manifest.json").exists() or (
        output / "preview/manifest.json"
    ).exists():
        raise ValueError(f"Asset refresh needs a new output directory: {output}")
    files = {selected}
    tree = ET.parse(selected)
    for node in tree.findall(".//mesh"):
        files.add((selected.parent / node.attrib["filename"]).resolve(strict=True))
    hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(files)
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "robot_asset_override.json").write_text(
        json.dumps(
            {
                "scope": "new robot asset; untimed preview; not the historical measurement asset",
                "frozen_robot_urdf": str(default),
                "selected_robot_urdf": str(selected),
                "link_count": len(tree.findall("link")),
                "joint_count": len(tree.findall("joint")),
                "sha256": hashes,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    return selected
