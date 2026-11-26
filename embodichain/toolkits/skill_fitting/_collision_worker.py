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

"""Internal subprocess for reconstructed-scene convex decomposition."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys

import numpy as np

__all__ = []


def _main() -> None:
    from dexsim.kit.meshproc.convex_cache import convex_decomposition_to_obj

    source, output, hulls, algorithm = sys.argv[1:]
    destination = Path(output)
    with np.load(source, allow_pickle=False) as data:
        cooked = convex_decomposition_to_obj(
            data["vertices"],
            data["triangles"],
            max_convex_hulls=int(hulls),
            max_vert_per_convex_hull=64,
            algorithm=algorithm,
            cache_dir=str(destination.parent),
        )
    shutil.copyfile(cooked, destination)


if __name__ == "__main__":
    _main()
