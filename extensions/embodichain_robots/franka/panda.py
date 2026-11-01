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

from .cfg import FrankaCfg
from embodichain.utils import configclass

__all__ = ["FrankaPandaCfg"]


@configclass
class FrankaPandaCfg(FrankaCfg):
    """Compatibility entry preserving the historical FR3/Panda-Hand preset.

    The legacy ``robot_type=\"panda\"`` selector retains its authored asset
    and defaults. New arm/tool selections use :class:`FrankaCfg`.
    """

    robot_type: str = "panda"

    def _build_defaults(self, init_dict: dict | None = None) -> None:
        values = dict(init_dict or {})
        self.robot_type = values.pop("robot_type", self.robot_type)
        if self.robot_type != "panda":
            raise ValueError(
                f"Unknown Franka robot_type: {self.robot_type!r}; expected panda"
            )
        if (
            values.get("arm_model", "fr3") != "fr3"
            or values.get("gripper", "panda_hand") != "panda_hand"
        ):
            raise ValueError(
                "FrankaPanda preserves its legacy combination; select FrankaCfg for other models"
            )
        values.update(arm_model="fr3", gripper="panda_hand")
        super()._build_defaults(values)


if __name__ == "__main__":
    import argparse

    import numpy as np

    np.set_printoptions(precision=5, suppress=True)

    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import RenderCfg, physics_cfg_for_backend

    parser = argparse.ArgumentParser(description="Launch the Franka Panda robot")
    parser.add_argument(
        "--physics",
        choices=("default", "newton"),
        default="default",
        help="Physics backend to launch (default: default).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Runtime device override; otherwise the selected backend default is used.",
    )
    args = parser.parse_args()

    config = SimulationManagerCfg(
        headless=True,
        device=args.device,
        num_envs=1,
        physics_cfg=physics_cfg_for_backend(args.physics),
        render_cfg=RenderCfg(renderer="hybrid"),
    )
    sim = SimulationManager(config)

    cfg = FrankaPandaCfg.from_dict({"robot_type": "panda"})
    robot = sim.add_robot(cfg=cfg)
    sim.prepare()
    sim.open_window()

    from IPython import embed

    embed()  # noqa: F401
