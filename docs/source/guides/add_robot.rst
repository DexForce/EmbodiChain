.. _guide_add_robot:

Adding a New Robot — Quick Reference
=====================================

This guide is a checklist + reference for adding a new robot to EmbodiChain. For
the full step-by-step walkthrough with code examples, see :doc:`/tutorial/add_robot`.

The protocol
------------

Every robot config subclasses :class:`~embodichain.lab.sim.cfg.RobotCfg` and
overrides two hooks:

- ``_build_defaults(self, init_dict=None)`` — populate ``urdf_cfg``,
  ``control_parts``, ``solver_cfg``, ``drive_pros`` and ``attrs`` from variant
  fields read out of ``init_dict``.
- ``build_pk_serial_chain(self, device=...)`` — return a
  ``{control_part: pk.SerialChain}`` mapping, reading the PK URDF from a single
  ``_pk_urdf_path`` source (so it cannot drift from the sim URDF).

The ``from_dict`` is a 3-line template (do not reimplement it)::

    cfg = cls()
    cfg._build_defaults(init_dict)
    return merge_robot_cfg(cfg, init_dict)

Serialization (``to_dict`` / ``to_string`` / ``save_to_file``) is inherited from
``RobotCfg`` and round-trips: ``RobotCfg.from_dict(cfg.to_dict())`` reproduces the cfg.

Checklist
---------

1. **Prepare the URDF** — place the URDF (+ meshes) in the assets directory.
2. **Override** ``_build_defaults(self, init_dict=None)`` — set variant fields from
   ``init_dict``, then populate ``urdf_cfg`` / ``control_parts`` / ``solver_cfg`` /
   ``drive_pros`` / ``attrs``.
3. **Define control parts** — group joints into logical sets (e.g. ``arm``, ``gripper``).
4. **Configure the IK solver** — ``OPWSolverCfg`` (6-DOF), ``SRSSolverCfg`` (7-DOF),
   or a generic ``SolverCfg``.
5. **Set drive properties** — stiffness/damping/max_effort per joint group.
6. **Implement** ``build_pk_serial_chain`` reading from ``_pk_urdf_path``.
7. **Keep** ``from_dict`` as the 3-line template — do not reimplement.
8. **Register** in ``embodichain/lab/sim/robots/__init__.py`` (and set ``__all__``).
9. **Add documentation** — ``docs/source/resources/robot/<name>.md`` + update
   ``resources/robot/index.rst``.
10. **Test** — a ``__main__`` smoke test + the DOF drift guard + ``preview-asset`` CLI.

Approaches
----------

- **Single-file** (variant-less robots): one ``my_robot.py`` with everything.
- **Package** (robots with variants — versions/arm-kinds/hand-brands): a directory
  with ``types.py`` (enums), ``cfg.py`` (a variant-aware ``_build_defaults``),
  optional ``params.py`` / ``utils.py`` helpers, and ``__init__.py``.

Key parameters
--------------

+---------------------+----------------------------------+----------------------------------+
| Parameter           | Type                             | Description                      |
+=====================+==================================+==================================+
| ``uid``             | str                              | Unique robot identifier          |
+---------------------+----------------------------------+----------------------------------+
| ``urdf_cfg``        | URDFCfg                          | URDF file and components         |
+---------------------+----------------------------------+----------------------------------+
| ``control_parts``   | Dict[str, List[str]]             | Joint groups for control         |
+---------------------+----------------------------------+----------------------------------+
| ``solver_cfg``      | Dict[str, SolverCfg]             | IK solver configurations         |
+---------------------+----------------------------------+----------------------------------+
| ``drive_pros``      | JointDrivePropertiesCfg          | Joint stiffness, damping, force  |
+---------------------+----------------------------------+----------------------------------+
| ``attrs``           | RigidBodyAttributesCfg           | Rigid-body physics attributes    |
+---------------------+----------------------------------+----------------------------------+
| variant fields      | enum / str / bool                | Optional subclass fields         |
|                     |                                  | (e.g. ``version``, ``arm_kind``) |
+---------------------+----------------------------------+----------------------------------+
| ``_pk_urdf_path``   | property or method → str         | URDF for the FK/IK serial chain  |
+---------------------+----------------------------------+----------------------------------+

Common mistakes
---------------

+-----------------------------------+----------------------------------------------------------+
| Mistake                           | Fix                                                      |
+===================================+==========================================================+
| ``all`` instead of ``__all__``    | Use ``__all__`` — lowercase ``all`` breaks ``import *``. |
+-----------------------------------+----------------------------------------------------------+
| ``solver_cfg`` set twice          | Set it once in ``_build_defaults`` only.                  |
+-----------------------------------+----------------------------------------------------------+
| PK URDF drifts from sim URDF      | Route PK through ``_pk_urdf_path``; keep the DOF guard.  |
+-----------------------------------+----------------------------------------------------------+
| Reimplementing ``from_dict``      | Keep the 3-line template; put logic in ``_build_defaults``.|
+-----------------------------------+----------------------------------------------------------+
| ``root_link_name`` as a tuple     | It must be a ``str``.                                     |
+-----------------------------------+----------------------------------------------------------+
| Calling a nonexistent ``validate``| Don't call methods that don't exist.                      |
+-----------------------------------+----------------------------------------------------------+

See Also
--------

- :doc:`/tutorial/add_robot` — Full step-by-step tutorial
- :doc:`/tutorial/robot` — Using robots in simulation
- :doc:`/overview/sim/solvers/index` — IK solver reference
- :doc:`/resources/robot/index` — Existing robot documentation
