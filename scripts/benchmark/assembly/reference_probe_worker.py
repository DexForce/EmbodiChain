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
"""Independent version-comparison worker for exported constrained S2 MJCF.

Run with a minimal MuJoCo/NumPy environment:
    python scripts/benchmark/assembly/reference_probe_worker.py --spec FILE --output FILE

No EmbodiChain, Torch, mjfit, renderer, or native SDK imports occur here.
The parent benchmark owns source preparation and its Markdown report.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform

import mujoco
import numpy as np

__all__: list[str] = []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.spec).resolve()
    cells = json.loads(source.read_text())["cells"]
    results = []
    for cell in cells:
        model = mujoco.MjModel.from_xml_path(str(source.parent / cell["mjcf"]))
        model.opt.timestep = cell["timestep"]
        if cell.get("multiccd", True):
            model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
        else:
            model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
        steps = round(cell["duration"] / model.opt.timestep)
        rows = []
        for angle in cell["directions_rad"]:
            direction = np.array([math.cos(angle), math.sin(angle)])
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            initial_overlap = any(
                contact.dist < -1e-9 for contact in data.contact[: data.ncon]
            )
            data.qfrc_applied[:2] = cell["force_N"] * direction
            values = []
            for step in range(steps):
                mujoco.mj_step(model, data)
                if step >= steps // 2:
                    values.append(float(data.qpos[:2] @ direction))
            error = np.asarray(values) - cell["truth_m"]
            finite = bool(np.isfinite(error).all())
            quarter = max(1, len(error) // 4)
            drift = float(error[-quarter:].mean() - error[:quarter].mean())
            worst = float(np.abs(error).max())
            warnings = sum(int(w.number) for w in data.warning)
            rows.append(
                {
                    "direction_rad": angle,
                    "wall_error_m": float(np.median(error)),
                    "wall_error_min_m": float(error.min()),
                    "wall_error_max_m": float(error.max()),
                    "wall_worst_error_m": worst,
                    "wall_jitter_m": float(error.std()),
                    "wall_drift_m": drift,
                    "finite": finite,
                    "initial_overlap": initial_overlap,
                    "warning_count": warnings,
                    "passed": finite
                    and not warnings
                    and not initial_overlap
                    and worst <= cell["tolerance_m"]
                    and abs(drift) <= cell["tolerance_m"],
                }
            )
        results.append(
            {
                "name": cell["name"],
                "asset_sha256": cell["asset_sha256"],
                "source_mjcf": cell["mjcf"],
                "source_config": cell["source_config"],
                "timestep": model.opt.timestep,
                "duration": cell["duration"],
                "solver_iterations": model.opt.iterations,
                "integrator": int(model.opt.integrator),
                "disableflags": int(model.opt.disableflags),
                "multiccd": not bool(
                    model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
                ),
                "force_N": cell["force_N"],
                "tolerance_m": cell["tolerance_m"],
                "passed": all(row["passed"] for row in rows),
                "rows": rows,
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "mujoco_version": mujoco.__version__,
                "numpy_version": np.__version__,
                "python": platform.python_version(),
                "scope": "independent CPU S2 exported-MJCF version comparison",
                "cells": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Independent reference saved: {output.resolve()}")


if __name__ == "__main__":
    main()
