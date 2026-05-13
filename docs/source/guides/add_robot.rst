.. _guide_add_robot:

Adding a New Robot — Quick Reference
=====================================

This guide provides a checklist and key reference for adding a new robot to EmbodiChain using the ``RobotDef`` protocol.

Checklist
---------

1. **Prepare the URDF** — Place your URDF file (and associated meshes) in the robot assets directory.
2. **Create the robot definition** — Implement the ``RobotDef`` protocol with ``@register_robot("MyRobot")``.
3. **Define control parts** — Group joints into logical sets (e.g., ``arm``, ``gripper``).
4. **Configure IK solver** — Choose ``OPWSolverCfg``, ``SRSSolverCfg``, or a generic ``SolverCfg`` per control part.
5. **Set drive properties** — Configure stiffness, damping, and max effort per joint group.
6. **Implement** ``build_pk_serial_chain`` — Required for PyTorch-Kinematics IK support.
7. **Register in** ``embodichain/lab/sim/robots/__init__.py``.
8. **Add documentation** — Create ``docs/source/resources/robot/my_robot.md`` and update ``resources/robot/index.rst``.
9. **Test** — Add a ``__main__`` block or use the ``preview-asset`` CLI to verify.

Approaches
----------

Simple robot (single-file)
~~~~~~~~~~~~~~~~~~~~~~~~~~

For robots without variants, declare all config as class-level fields:

.. code-block:: python

    from embodichain.lab.sim.robots.protocol import RobotDef
    from embodichain.lab.sim.robots.registry import register_robot
    from embodichain.lab.sim.cfg import (
        RobotCfg, URDFCfg, JointDrivePropertiesCfg, RigidBodyAttributesCfg,
    )
    from embodichain.lab.sim.solvers import SolverCfg, OPWSolverCfg
    from embodichain.data import get_data_path
    import numpy as np

    @register_robot("MyRobot")
    class MyRobotDef:
        name: str = "MyRobot"

        urdf_cfg: URDFCfg = URDFCfg(components=[
            {"component_type": "arm",
             "urdf_path": get_data_path("MyRobot/arm.urdf")},
        ])

        control_parts: dict[str, list[str]] = {
            "arm": ["JOINT[1-6]"],
            "gripper": ["FINGER[1-2]"],
        }

        solver_cfg: dict[str, SolverCfg] = {
            "arm": OPWSolverCfg(
                end_link_name="link6",
                root_link_name="base_link",
            ),
        }

        drive_pros: JointDrivePropertiesCfg = JointDrivePropertiesCfg(
            stiffness={"JOINT[1-6]": 1e4, "FINGER[1-2]": 1e2},
            damping={"JOINT[1-6]": 1e3, "FINGER[1-2]": 1e1},
        )

        attrs: RigidBodyAttributesCfg = RigidBodyAttributesCfg()

        def build_pk_serial_chain(self, device):
            return {}

        def build_cfg(self, **overrides):
            return RobotDef.build_cfg(self, **overrides)

Usage::

    from embodichain.lab.sim.robots import build_robot_cfg
    cfg = build_robot_cfg("MyRobot", overrides={"uid": "my_robot"})
    robot = sim.add_robot(cfg=cfg)

Complex robot (package)
~~~~~~~~~~~~~~~~~~~~~~~

For robots with variants (arm types, hand brands, versions), use ``@property`` methods and variant parameters:

.. code-block:: python

    @register_robot("MyComplexRobot")
    class MyComplexRobotDef:
        name: str = "MyComplexRobot"

        # Variant parameters
        arm_kind: str = "standard"

        def __post_init__(self):
            # Fill defaults based on arm_kind
            ...

        @property
        def urdf_cfg(self) -> URDFCfg:
            return _build_urdf(self.arm_kind)

        @property
        def control_parts(self) -> dict[str, list[str]]:
            return _build_control_parts(self.arm_kind)

        @property
        def solver_cfg(self) -> dict[str, SolverCfg]:
            return _build_solver_cfg(self.arm_kind)

        @property
        def drive_pros(self) -> JointDrivePropertiesCfg:
            return _build_drive_pros(self.arm_kind)

        @property
        def attrs(self) -> RigidBodyAttributesCfg:
            return RigidBodyAttributesCfg()

        def build_pk_serial_chain(self, device):
            return {}

        def build_cfg(self, **overrides):
            return RobotDef.build_cfg(self, **overrides)

Variant usage::

    cfg = build_robot_cfg("MyComplexRobot", arm_kind="extended",
                          overrides={"uid": "my_robot"})

Key Parameters
--------------

+---------------------+----------------------------+----------------------------------+
| Parameter           | Type                       | Description                      |
+=====================+============================+==================================+
| ``name``            | str                        | Unique robot identifier          |
+---------------------+----------------------------+----------------------------------+
| ``urdf_cfg``        | URDFCfg                    | URDF file and components         |
+---------------------+----------------------------+----------------------------------+
| ``control_parts``   | Dict[str, List[str]]       | Joint groups for control         |
+---------------------+----------------------------+----------------------------------+
| ``solver_cfg``      | Dict[str, SolverCfg]       | IK solver configurations         |
+---------------------+----------------------------+----------------------------------+
| ``drive_pros``      | JointDrivePropertiesCfg    | Joint stiffness, damping, force  |
+---------------------+----------------------------+----------------------------------+
| ``attrs``           | RigidBodyAttributesCfg     | Physical attributes              |
+---------------------+----------------------------+----------------------------------+

Backward Compatibility
----------------------

The old ``*Cfg.from_dict()`` API continues to work. ``CobotMagicCfg`` and ``DexforceW1Cfg`` are thin wrappers that delegate to the new ``RobotDef`` implementations.

.. tip::

   See :doc:`/tutorial/robot` for using robots in simulation and
   :doc:`/resources/robot/index` for existing robot documentation.

See Also
--------

- :doc:`/tutorial/robot` — Using robots in simulation
- :doc:`/overview/sim/solvers/index` — IK solver reference
- :doc:`/resources/robot/index` — Existing robot documentation
