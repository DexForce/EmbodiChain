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
"""Capture a preview and read initialized model parameters outside measurement.

Run: PLATFORM_PYTHON record_s03_audited_preview.py --platform PLATFORM --batch PATH
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

TECH = Path(__file__).resolve().parents[3]
__all__ = ["main"]


def _save_model(model: object, solver: object, output: Path) -> None:
    """Read model arrays and selected initialized solver settings without edits."""
    output.mkdir(parents=True, exist_ok=True)
    arrays = {}
    labels = {}
    for prefix, source in (
        ("", model),
        ("dexuni.", getattr(model, "dexuni", None)),
        ("mujoco.", getattr(model, "mujoco", None)),
    ):
        if source is None:
            continue
        for name in dir(source):
            if name.startswith("_"):
                continue
            try:
                value = getattr(source, name)
                if hasattr(value, "numpy"):
                    arrays[prefix + name] = value.numpy()
                elif name in ("body_label", "joint_label", "shape_label"):
                    arrays[prefix + name] = np.asarray(value)
            except (TypeError, ValueError, RuntimeError, AttributeError):
                continue
    np.savez_compressed(output / "model.npz", **arrays)
    visited = set()

    def inspect(value: object, depth: int = 0) -> dict:
        if id(value) in visited or depth > 3:
            return {}
        visited.add(id(value))
        row = {"class": type(value).__name__}
        for name, item in vars(value).items() if hasattr(value, "__dict__") else ():
            if any(
                word in name
                for word in (
                    "iteration",
                    "friction_epsilon",
                    "contact_hard",
                    "contact_history",
                    "tile_solve",
                    "fast_path",
                    "mass_scale",
                    "relaxation",
                    "mode",
                )
            ) and isinstance(item, (str, bool, int, float)):
                row[name] = item
            if "model" in name and hasattr(item, "opt"):
                opt = item.opt
                row[name + ".opt"] = {
                    k: (
                        getattr(opt, k).numpy().tolist()
                        if hasattr(getattr(opt, k), "numpy")
                        else str(getattr(opt, k))
                    )
                    for k in (
                        "iterations",
                        "ls_iterations",
                        "integrator",
                        "cone",
                        "solver",
                        "disableflags",
                    )
                    if hasattr(opt, k)
                }
            if "solver" in name or "entries" in name or "views" in name:
                if isinstance(item, dict):
                    row[name] = {str(k): inspect(v, depth + 1) for k, v in item.items()}
                elif isinstance(item, (list, tuple)):
                    row[name] = [inspect(v, depth + 1) for v in item]
                elif item is not None:
                    row[name] = inspect(item, depth + 1)
        return row

    labels["solver"] = inspect(solver)
    labels["scope"] = (
        "read-only initialization audit during scene preview; not a performance sample"
    )
    (output / "solver.json").write_text(json.dumps(labels, indent=2) + "\n")


def main() -> None:
    """Use unchanged physical workers, adding read-only initialization snapshots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform", choices=("embodichain", "isaaclab"), required=True
    )
    parser.add_argument("--batch", type=Path, required=True)
    args = parser.parse_args()
    batch = args.batch.resolve()
    if TECH.parents[2] / "outputs/benchmarks/tech_reports/dexuni" not in batch.parents:
        raise ValueError(
            "Preview batch must be in outputs/benchmarks/tech_reports/dexuni"
        )
    os.environ["DEXUNI_INPUT_ROOT"] = str(batch / "inputs")
    os.environ["DEXUNI_ALLOW_PILOT"] = "1"
    worker = TECH / f"physics/platforms/dexuni/{args.platform}/s03.py"
    sys.argv = [
        str(worker),
        "--preview-only",
        "--images",
        "full",
        "--out-root",
        str(batch / "preview"),
    ]
    spec = importlib.util.spec_from_file_location("s03_preview_worker", worker)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    output = batch / "audit" / args.platform
    if args.platform == "embodichain":
        original = module.newton_model

        def audited_model(sim):
            from dexsim.engine.newton_physics.backend_registry import get_newton_backend

            model = original(sim)
            _save_model(model, get_newton_backend(sim.get_world()).solver, output)
            (output / "diagnostics.json").write_text(
                json.dumps(sim.spawn_result.get_solver_diagnostics(), indent=2) + "\n"
            )
            return model

        module.newton_model = audited_model
        module.main()
    else:
        original = module.describe_model

        def audited_description():
            from isaaclab_newton.physics import NewtonManager

            _save_model(NewtonManager.backend.model, NewtonManager._solver, output)
            return original()

        module.describe_model = audited_description
        module.run(module.args_cli)


if __name__ == "__main__":
    main()
