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
import os, subprocess, sys, time, json

root = Path("/root/sources/EmbodiChain/outputs/lewm/official-cube")
jobs = [
    ("point", "base", 1),
    ("point", "discrete", 1),
    ("mimic", "base", 1),
    ("both", "base", 1),
    ("both", "discrete", 1),
    ("none", "base", 0),
    ("point", "discrete", 0),
    ("mimic", "base", 0),
    ("both", "base", 0),
    ("both", "discrete", 0),
]
rows = []
for mode, gains, iterations in jobs:
    path = root / "alignment" / f"real14-coupled-{mode}-{gains}-v{iterations}.log"
    start = time.monotonic()
    with path.open("w") as out:
        p = subprocess.run(
            [
                sys.executable,
                "-X",
                "faulthandler",
                "-u",
                str(root / "reference/real14_coupled_impulse.py"),
                "native",
                "--mode",
                mode,
                "--point-gains",
                gains,
                "--velocity-iterations",
                str(iterations),
            ],
            stdout=out,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
    rows.append(
        {
            "mode": mode,
            "gains": gains,
            "velocity_iterations": iterations,
            "exit_code": p.returncode,
            "elapsed_s": time.monotonic() - start,
        }
    )
    print(rows[-1], flush=True)
    if p.returncode:
        raise SystemExit(p.returncode)
(root / "alignment/real14-coupled-impulse/processes.json").write_text(
    json.dumps(rows, indent=2) + "\n"
)
