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
results = []
for mode, point, mimic, vel in [
    ("both", "discrete_bias", "base", 0),
    ("both", "discrete_bias", "base", 1),
    ("mimic", "base", "discrete", 0),
    ("both", "discrete_bias", "discrete", 0),
]:
    suffix = "-timemimic" if mimic == "discrete" else ""
    start = time.monotonic()
    with (
        root / "alignment" / f"real14-coupled-{mode}-{point}{suffix}-v{vel}.log"
    ).open("w") as out:
        proc = subprocess.run(
            [
                sys.executable,
                "-u",
                str(root / "reference/real14_coupled_impulse_bias.py"),
                "native",
                "--mode",
                mode,
                "--point-gains",
                point,
                "--mimic-gains",
                mimic,
                "--velocity-iterations",
                str(vel),
            ],
            stdout=out,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
    results.append(
        {
            "mode": mode,
            "point": point,
            "mimic": mimic,
            "velocity_iterations": vel,
            "exit_code": proc.returncode,
            "elapsed_s": time.monotonic() - start,
        }
    )
    print(results[-1], flush=True)
    if proc.returncode:
        raise SystemExit(proc.returncode)
(root / "alignment/real14-coupled-impulse/bias-processes.json").write_text(
    json.dumps(results, indent=2) + "\n"
)
