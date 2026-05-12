.. _guide_add_robot:

Adding a New Robot — Quick Reference
=====================================

This guide provides a checklist and key reference for adding a new robot to EmbodiChain. For the full step-by-step walkthrough with code examples, see :doc:`/tutorial/add_robot`.

Checklist
---------

1. **Prepare the URDF** — Place your URDF file (and associated meshes) in the robot assets directory.
2. **Create the config class** — Inherit from ``RobotCfg``, implement ``from_dict`` and ``_build_default_cfgs``.
3. **Define control parts** — Group joints into logical sets (e.g., ``arm``, ``gripper``).
4. **Configure IK solver** — Choose ``OPWSolverCfg``, ``SRSSolverCfg``, or a generic ``SolverCfg``.
5. **Set drive properties** — Configure stiffness, damping, and max effort per joint group.
6. **Implement** ``build_pk_serial_chain`` — Required for PyTorch-Kinematics IK support.
7. **Register in** ``embodichain/lab/sim/robots/__init__.py``.
8. **Add documentation** — Create ``docs/source/resources/robot/my_robot.md`` and update ``resources/robot/index.rst``.
9. **Test** — Add a ``__main__`` block or use the ``preview-asset`` CLI to verify.

Approaches
----------

- **Single-file** (simple robots): One ``my_robot.py`` with everything.
- **Package** (complex robots): Directory with ``types.py``, ``params.py``, ``utils.py``, ``cfg.py``, ``__init__.py``.

Key Parameters
--------------

+---------------------+----------------------------+----------------------------------+
| Parameter           | Type                       | Description                      |
+=====================+============================+==================================+
| ``uid``             | str                        | Unique robot identifier          |
+---------------------+----------------------------+----------------------------------+
| ``urdf_cfg``        | URDFCfg                    | URDF file and components         |
+---------------------+----------------------------+----------------------------------+
| ``control_parts``   | Dict[str, List[str]]       | Joint groups for control         |
+---------------------+----------------------------+----------------------------------+
| ``solver_cfg``      | Dict[str, SolverCfg]       | IK solver configurations         |
+---------------------+----------------------------+----------------------------------+
| ``drive_pros``      | JointDrivePropertiesCfg    | Joint stiffness, damping, force  |
+---------------------+----------------------------+----------------------------------+

.. tip::

   See the :doc:`full tutorial </tutorial/add_robot>` for complete code examples of both approaches.

See Also
--------

- :doc:`/tutorial/add_robot` — Full step-by-step tutorial
- :doc:`/tutorial/robot` — Using robots in simulation
- :doc:`/overview/sim/solvers/index` — IK solver reference
- :doc:`/resources/robot/index` — Existing robot documentation
