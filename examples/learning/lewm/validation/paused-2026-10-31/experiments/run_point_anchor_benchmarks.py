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
import subprocess, sys, time, json

root = Path("/root/sources/EmbodiChain/outputs/lewm/official-cube")
rows = []
for n, enabled in [(1, True), (25, False), (25, True)]:
    label = "point" if enabled else "default"
    name = f"point-anchor-benchmark-{label}-{n}"
    start = time.monotonic()
    cmd = [
        sys.executable,
        "-m",
        "scripts.benchmark.learning.benchmark_d6_point_drive",
        "--num-envs",
        str(n),
        "--device",
        "cuda:5",
        "--output",
        str(root / "alignment" / f"{name}.json"),
    ]
    if enabled:
        cmd.append("--point-anchors")
    with (root / "alignment" / f"{name}.log").open("w") as out:
        p = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, timeout=300)
    rows.append(
        {
            "n": n,
            "point": enabled,
            "exit_code": p.returncode,
            "elapsed_s": time.monotonic() - start,
        }
    )
    print(rows[-1], flush=True)
    if p.returncode:
        raise SystemExit(p.returncode)
(root / "alignment/point-anchor-benchmark-processes.json").write_text(
    json.dumps(rows, indent=2) + "\n"
)
