.. _tutorial_add_robot:

Adding a New Robot
==================

This tutorial walks through adding a new robot config to EmbodiChain. For the
quick-reference checklist, see :doc:`/guides/add_robot`.

A robot config subclasses :class:`~embodichain.lab.sim.cfg.RobotCfg`, which itself
extends ``ArticulationCfg`` → ``ObjectBaseCfg``. You override two hooks and keep
``from_dict`` as a 3-line template. Serialization is free.

Two paths: **single-file** (variant-less robots) and **package with variants**
(robots with versions / arm-kinds / hand-brands).

Path A — single-file robot
--------------------------

A variant-less robot lives in one file, e.g. ``embodichain/lab/sim/robots/my_robot.py``::

    from __future__ import annotations

    import numpy as np
    import torch
    from typing import TYPE_CHECKING, Dict

    from embodichain.lab.sim.cfg import (
        RobotCfg, URDFCfg, JointDrivePropertiesCfg, RigidBodyAttributesCfg,
    )
    from embodichain.lab.sim.solvers import OPWSolverCfg
    from embodichain.lab.sim.utility.cfg_utils import merge_robot_cfg
    from embodichain.data import get_data_path
    from embodichain.utils import configclass

    if TYPE_CHECKING:
        import pytorch_kinematics as pk

    __all__ = ["MyRobotCfg"]


    @configclass
    class MyRobotCfg(RobotCfg):

        @classmethod
        def from_dict(cls, init_dict):
            cfg = cls()
            cfg._build_defaults(init_dict)
            return merge_robot_cfg(cfg, init_dict)

        def _build_defaults(self, init_dict=None):
            init_dict = init_dict or {}
            self.uid = "MyRobot"
            self.urdf_cfg = URDFCfg(
                components=[
                    {
                        "component_type": "arm",
                        "urdf_path": get_data_path("MyRobot/arm.urdf"),
                        "transform": np.eye(4),
                    }
                ]
            )
            self.control_parts = {"arm": ["JOINT[1-6]"]}
            self.solver_cfg = {
                "arm": OPWSolverCfg(
                    end_link_name="link6",
                    root_link_name="base_link",
                    tcp=np.eye(4),
                )
            }
            self.drive_pros = JointDrivePropertiesCfg(stiffness={"JOINT[1-6]": 1e4})

        @property
        def _pk_urdf_path(self) -> str:
            return get_data_path("MyRobot/arm.urdf")

        def build_pk_serial_chain(
            self, device: torch.device = torch.device("cpu"), **kwargs
        ) -> Dict[str, "pk.SerialChain"]:
            from embodichain.lab.sim.utility.solver_utils import create_pk_serial_chain

            chain = create_pk_serial_chain(
                urdf_path=self._pk_urdf_path, device=device,
                end_link_name="link6", root_link_name="base_link",
            )
            return {"arm": chain}


    if __name__ == "__main__":
        from embodichain.lab.sim import SimulationManager, SimulationManagerCfg

        sim = SimulationManager(SimulationManagerCfg(headless=True, num_envs=1))
        cfg = MyRobotCfg.from_dict({})
        robot = sim.add_robot(cfg=cfg)
        print("MyRobot added:", cfg.to_dict()["uid"])

Path B — package robot with variants
-------------------------------------

When a robot has variants (versions, arm kinds, hand brands), use a package
directory ``embodichain/lab/sim/robots/my_robot/``:

- ``types.py`` — enums (``MyRobotVersion``, ``MyRobotArmKind``) with a proper
  ``__all__``.
- ``cfg.py`` — ``MyRobotCfg`` with a variant-aware ``_build_defaults`` that reads
  variant fields from ``init_dict``::

      def _build_defaults(self, init_dict=None):
          init_dict = init_dict or {}
          self.version = MyRobotVersion(init_dict.get("version", "v1"))
          self.arm_kind = MyRobotArmKind(init_dict.get("arm_kind", "default"))
          ...  # urdf_cfg / control_parts / solver_cfg / drive_pros / attrs
          self.solver_cfg = self._build_default_solver(arm_kind=self.arm_kind)

- ``params.py`` / ``utils.py`` — **optional** helpers (kinematic params, URDF
  assembly builders). They are not part of the protocol; factor them out only when
  the cfg file would otherwise be unwieldy.
- ``__init__.py`` — ``from .cfg import MyRobotCfg``.

``_pk_urdf_path`` for a variant robot is a method (the path depends on the variant)::

    def _pk_urdf_path(self) -> str:
        if self.arm_kind == MyRobotArmKind.KIND_A:
            return get_data_path("MyRobot/arm_a.urdf")
        return get_data_path("MyRobot/arm_b.urdf")

Registering the robot
---------------------

In ``embodichain/lab/sim/robots/__init__.py``::

    from .my_robot import MyRobotCfg

    __all__ = ["MyRobotCfg"]   # add to the existing __all__

Testing
-------

Add a ``__main__`` smoke test (round-trip + add to sim) and the DOF drift guard
(see ``tests/sim/objects/test_robot_cfg.py`` for examples)::

    chains = cfg.build_pk_serial_chain()
    for part, chain in chains.items():
        assert len(chain.get_joint_parameter_names()) == len(cfg.control_parts[part])

Use the ``preview-asset`` CLI to visually verify the URDF loads.

Serialization
------------

``to_dict`` / ``to_string`` / ``save_to_file`` are inherited from ``RobotCfg``::

    cfg.save_to_file("my_robot.json")
    cfg2 = MyRobotCfg.from_dict(cfg.to_dict())   # round-trips

Use this to snapshot tuned configs or persist calibrated parameters.

See Also
--------

- :doc:`/guides/add_robot` — Quick-reference checklist
- :doc:`/tutorial/robot` — Using robots in simulation
- :doc:`/overview/sim/solvers/index` — IK solver reference
- :doc:`/resources/robot/index` — Existing robot documentation
