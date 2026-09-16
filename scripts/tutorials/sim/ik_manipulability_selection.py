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

"""Manipulability-driven multi-seed IK re-ranking, compared with seed selection.

For each target pose the multi-seed IK pipeline produces many valid joint
solutions and collapses them to one. This example shows the decision that
collapse makes and what manipulability re-ranking changes:

- ``nearest`` (default): keep the successful candidate closest to the seed.
- ``manipulability``: keep the best-conditioned candidate
  (``ik_solution_selection="manipulability"``), scored with
  :func:`embodichain.compute.kinematics.yoshikawa_manipulability`.

Both collapse modes are also combined with database seed selection
(``enable_seed_selection=True``, the SELIK-style retrieval) to separate the
two effects: seed selection improves *which candidates exist*, re-ranking
improves *which candidate is kept*.

Run:
    python scripts/tutorials/sim/ik_manipulability_selection.py --num_targets 50
"""

from __future__ import annotations

import argparse
import time

import torch

from embodichain.compute.kinematics import (
    condition_number,
    yoshikawa_manipulability,
)
from embodichain.data import get_data_path
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import RobotCfg
from embodichain.lab.sim.objects import Robot
from embodichain.lab.visualization.cli import (
    add_viser_args_to_parser,
    visualization_cfg_from_args,
)
from embodichain.utils.utility import reset_all_seeds

# (label, ik_solution_selection, enable_seed_selection)
VARIANTS = (
    ("nearest (default)", "nearest", False),
    ("manipulability re-rank", "manipulability", False),
    ("iksel + nearest", "nearest", True),
    ("iksel + manipulability", "manipulability", True),
)


def _robot_cfg(uid: str, selection: str, seed_selection: bool) -> RobotCfg:
    urdf = get_data_path("DexforceW1V021/DexforceW1_v02_1.urdf")
    return RobotCfg.from_dict(
        {
            "uid": uid,
            "fpath": urdf,
            "control_parts": {"left_arm": [f"LEFT_J{i+1}" for i in range(7)]},
            "solver_cfg": {
                "left_arm": {
                    "class_type": "PytorchSolver",
                    "end_link_name": "left_ee",
                    "root_link_name": "left_arm_base",
                    "num_samples": 30,
                    "ik_solution_selection": selection,
                    "enable_seed_selection": seed_selection,
                },
            },
        }
    )


def _sample_targets(solver, num_targets: int) -> tuple[torch.Tensor, torch.Tensor]:
    """FK poses of interior joint configurations, guaranteed reachable."""
    reset_all_seeds(0)
    lower, upper = solver.lower_qpos_limits, solver.upper_qpos_limits
    span = upper - lower
    q_true = lower + span * (0.25 + 0.5 * torch.rand(num_targets, solver.dof))
    with torch.no_grad():
        return solver.get_fk(q_true), q_true


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_targets", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    add_viser_args_to_parser(parser)
    args = parser.parse_args()

    # This comparison runs headless, so the browser scene is the only way to
    # inspect it. Viser stays off unless --viser is passed, which keeps the
    # reported timings free of capture overhead by default.
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            sim_device=args.device,
            visualization=visualization_cfg_from_args(args),
        )
    )
    try:
        rows = []
        targets = seed_q = None
        for label, selection, seed_selection in VARIANTS:
            uid = f"w1_{selection}_{'sel' if seed_selection else 'rand'}"
            robot: Robot = sim.add_robot(cfg=_robot_cfg(uid, selection, seed_selection))
            solver = robot.get_solver("left_arm")
            if targets is None:
                targets, seed_q = _sample_targets(solver, args.num_targets)

            # Warm up compile caches at the timed batch shape and (for
            # iksel) the seed database, so the timed call measures
            # steady-state solving.
            solver.get_ik(targets, qpos_seed=seed_q[0])

            reset_all_seeds(1)  # identical random seed draws across variants
            start = time.perf_counter()
            success, qpos = solver.get_ik(targets, qpos_seed=seed_q[0])
            elapsed_ms = (time.perf_counter() - start) * 1e3

            qpos = qpos[:, 0, :]
            jac = solver.get_jacobian(qpos)
            scores = yoshikawa_manipulability(jac)
            conditions = condition_number(jac)
            ok = success.to(torch.bool)
            rows.append(
                (
                    label,
                    float(ok.float().mean()) * 100.0,
                    float(scores[ok].mean()),
                    float(scores[ok].min()),
                    float(conditions[ok].mean()),
                    elapsed_ms,
                )
            )

        header = (
            f"{'variant':<26} {'success':>8} {'mean w':>9} {'min w':>9} "
            f"{'mean cond':>10} {'time ms':>9}"
        )
        print("\n=== Multi-seed IK solution selection on DexforceW1 left arm ===")
        print(f"targets: {args.num_targets}, num_samples per target: 30\n")
        print(header)
        print("-" * len(header))
        for label, rate, mean_w, min_w, mean_c, ms in rows:
            print(
                f"{label:<26} {rate:>7.1f}% {mean_w:>9.4f} {min_w:>9.4f} "
                f"{mean_c:>10.2f} {ms:>9.1f}"
            )
        base = rows[0]
        for label, _, mean_w, _, _, _ in rows[1:]:
            gain = (mean_w / base[2] - 1.0) * 100.0
            print(f"\n{label}: mean manipulability {gain:+.1f}% vs nearest baseline")
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
