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
"""Common command-line host for native passive assembly examples."""

from __future__ import annotations

import argparse
import json

from embodichain.toolkits.assembly import AssemblyAsset, build_assembly
from embodichain_benchmarks.assembly.precision import PrecisionRunCfg, run_passive_probe


def run(config: str) -> None:
    """Run the selected physical probe; print measured and unverified fields."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=config)
    parser.add_argument(
        "--backend",
        choices=["mujoco_cpu", "mujoco_warp", "newton", "default"],
        default="newton",
    )
    parser.add_argument("--route", choices=["direct", "usd"], default="direct")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--duration", type=float, default=1.5)
    parser.add_argument("--timestep", type=float, default=0.001)
    parser.add_argument("--solver", choices=["pgs", "cg", "newton"])
    parser.add_argument("--use-mujoco-cpu", action="store_true")
    parser.add_argument("--solver-tolerance", type=float)
    parser.add_argument(
        "--multiccd", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--backend-contacts", action="store_true")
    args = parser.parse_args()
    asset = (
        AssemblyAsset.load(args.config)
        if args.config.endswith(".json")
        else build_assembly(args.config)
    )
    result = run_passive_probe(
        asset,
        PrecisionRunCfg(
            backend=args.backend,
            route=args.route,
            num_envs=args.num_envs,
            duration=args.duration,
            timestep=args.timestep,
            solver=args.solver,
            use_mujoco_cpu=args.use_mujoco_cpu,
            solver_tolerance=args.solver_tolerance,
            enable_multiccd=args.multiccd,
            source_contact=not args.backend_contacts,
        ),
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    if result["status"] in {"failed", "skipped"}:
        raise SystemExit(1)
