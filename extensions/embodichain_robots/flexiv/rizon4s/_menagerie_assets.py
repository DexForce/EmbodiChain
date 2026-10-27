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

"""Pinned public source assets for the Rizon 4S and Robotiq assembly model."""

from __future__ import annotations

import open3d as o3d

from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATA_ROOT
from embodichain.data.dataset import EmbodiChainDataset

__all__ = ["FlexivRizon4SMenagerie"]


class FlexivRizon4SMenagerie(EmbodiChainDataset):
    """Pinned public Rizon 4S and Robotiq 2F-85 source models with licenses.

    Source files are verified using Git blob identities from the packaged
    manifest. This dataset contains nominal public dynamics, not the assembly
    paper's identified robot model. Historical URDF preparation is owned by
    ``embodichain.gen_sim.assembly.robot_reference``; loading prepared URDFs does not require MuJoCo.
    """

    def __init__(self, data_root: str | None = None) -> None:
        import hashlib
        import json
        from pathlib import Path
        from urllib.request import urlopen

        from embodichain.data.dataset import _dataset_download_lock

        manifest = json.loads(
            Path(__file__).with_name("rizon4s_sources.json").read_text()
        )
        revision = manifest["revision"]
        prefix = type(self).__name__
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root
        # The existing data/download registry uses Open3D DownloadDataset. A
        # small licensed source file is its checked seed; the immutable source
        # manifest preserves paths and verifies every subsequent resource.
        seed_url = (
            "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/"
            f"{revision}/flexiv_rizon4s/LICENSE"
        )
        super().__init__(
            prefix,
            o3d.data.DataDescriptor(seed_url, "233049eed2ea22fb364a87bdc4e04fa9"),
            path,
        )
        with _dataset_download_lock(path, prefix):
            root = Path(self.extract_dir)
            root.mkdir(parents=True, exist_ok=True)
            for relative, expected in manifest["files"].items():
                target = root / relative
                data = target.read_bytes() if target.is_file() else b""
                digest = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
                if digest == expected:
                    continue
                url = (
                    "https://raw.githubusercontent.com/google-deepmind/"
                    f"mujoco_menagerie/{revision}/{relative}"
                )
                with urlopen(url, timeout=60) as response:
                    data = response.read()
                digest = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
                if digest != expected:
                    raise RuntimeError(f"Source integrity mismatch: {relative}")
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix + ".part")
                temporary.write_bytes(data)
                temporary.replace(target)
            (root / "source_manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
