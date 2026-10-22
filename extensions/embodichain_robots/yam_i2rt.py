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

"""YAM i2rt dual-arm configuration, using the reference routing embodiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING
import torch
from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATA_ROOT
from embodichain.lab.sim.cfg import (
    RobotCfg,
    JointDrivePropertiesCfg,
    RigidBodyPhysicsCfg,
)
from embodichain.lab.sim.motion.solvers import PytorchSolverCfg
from embodichain.lab.sim.utility.cfg_utils import merge_robot_cfg
from embodichain.utils import configclass

if TYPE_CHECKING:
    import pytorch_kinematics as pk

__all__ = ["YamI2rtCfg"]


@configclass
class YamI2rtCfg(RobotCfg):
    """Two six-joint YAM arms with two independently driven fingers per hand.

    Assets retain their MIT license and pinned source provenance. TCP links use
    the reference grasp-site frame. Runtime grasp parameters belong to tasks.
    """

    @classmethod
    def from_dict(cls, init_dict: dict[str, Any]) -> YamI2rtCfg:
        """Construct source defaults and apply explicit robot overrides."""
        cfg = cls()
        cfg._build_defaults(init_dict)
        return merge_robot_cfg(cfg, init_dict)

    def _build_defaults(self, init_dict: dict | None = None) -> None:
        self.uid = "yam"
        self.fpath = (init_dict or {}).get("fpath") or self.fpath
        self.fpath = self._pk_urdf_path
        if not Path(self.fpath).is_file():
            raise FileNotFoundError(
                f"YAM robot asset not found: {self.fpath}. From the EmbodiChain "
                "checkout run `python scripts/tools/prepare_yam_i2rt_assets.py`, "
                "or set the robot's fpath to an already prepared yam_i2rt.urdf."
            )
        self.init_pos = (0.2525, 0, 0.75)
        self.asset_physics_mode = "overlay"
        self.control_parts = {
            f"{side}_arm": [f"{side}_joint{i}" for i in range(1, 7)]
            for side in ("right", "left")
        }
        self.control_parts.update(
            {
                f"{side}_eef": [f"{side}_joint7", f"{side}_joint8"]
                for side in ("right", "left")
            }
        )
        self.solver_cfg = {
            f"{side}_arm": PytorchSolverCfg(
                root_link_name=f"{side}_base",
                end_link_name=f"{side}_tcp",
                num_samples=12,
                # Cartesian retiming measures finite-difference acceleration;
                # loose IK convergence produces visible joint jitter on that grid.
                max_iterations=200,
                pos_eps=1e-5,
                rot_eps=1e-5,
            )
            for side in ("right", "left")
        }
        self.init_qpos = [
            0.0,
            0.0,
            1.4,
            1.4,
            1.2,
            1.2,
            0.0,
            0.0,
            1.1,
            1.1,
            0.0,
            0.0,
            0.0475,
            0.0475,
            0.0475,
            0.0475,
        ]
        self.joint_drive_props = JointDrivePropertiesCfg(
            stiffness={
                "right_joint[1-6]": 300.0,
                "left_joint[1-6]": 300.0,
                ".*_joint[78]": 600.0,
            },
            damping={
                "right_joint[1-6]": 30.0,
                "left_joint[1-6]": 30.0,
                ".*_joint[78]": 20.0,
            },
            max_effort={
                ".*_joint[1-3]": 27.0,
                ".*_joint[4-6]": 10.0,
                ".*_joint[78]": 80.0,
            },
            armature={
                ".*_joint[1-3]": 0.0192,
                ".*_joint[4-6]": 0.0012,
                ".*_joint[78]": 0.05,
            },
            target_mode="position",
        )
        # The source MJCF enables gravcomp=1 on every body.
        self.newton_gravity_compensation = 1.0
        self.attrs = RigidBodyPhysicsCfg.from_dict(
            {"collision_props": {"contact_offset": 0.0002, "rest_offset": 0.0}}
        )

    @property
    def _pk_urdf_path(self) -> str:
        path = (
            Path(self.fpath).expanduser()
            if self.fpath
            else Path(EMBODICHAIN_DEFAULT_DATA_ROOT) / "YamI2rt/yam_i2rt.urdf"
        )
        return str(path)

    def build_pk_serial_chain(
        self, device: torch.device = torch.device("cpu"), **kwargs: Any
    ) -> dict[str, pk.SerialChain]:
        """Build local arm chains from the exact simulation URDF.

        Args:
            device: Kinematics device.
            **kwargs: Reserved configuration hooks.

        Returns:
            Six-joint chains keyed by control part.
        """
        from embodichain.lab.sim.utility.solver_utils import create_pk_serial_chain

        return {
            f"{side}_arm": create_pk_serial_chain(
                urdf_path=self._pk_urdf_path,
                device=device,
                root_link_name=f"{side}_base",
                end_link_name=f"{side}_tcp",
            )
            for side in ("right", "left")
        }
