.. _guide_add_robot:

Adding a New Robot
==================

This guide provides a practical checklist and reference for adding a new robot
to EmbodiChain.

Official concrete presets live in ``embodichain_robots``; generic robot runtime
and dual-arm composition remain in the core. Use :doc:`robot_catalog` for
catalog registration, display metadata, and canonical import paths.

The protocol
------------

Every robot config subclasses :class:`~embodichain.lab.sim.cfg.RobotCfg` and
overrides two hooks:

- ``_build_defaults(self, init_dict=None)`` — populate ``urdf_cfg``,
  ``control_parts``, ``solver_cfg``, ``joint_drive_props`` and ``attrs`` from variant
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

1. **Prepare the URDF** — publish versioned URDF/mesh resources and declare their asset class in
   the owning vendor/model package. Register the class through the lightweight
   robot asset catalog; reuse the core download/cache implementation.
2. **Override** ``_build_defaults(self, init_dict=None)`` — set variant fields from
   ``init_dict``, then populate ``urdf_cfg`` / ``control_parts`` / ``solver_cfg`` /
   ``joint_drive_props`` / ``attrs``.
3. **Define control parts** — use ``arm`` / ``eef`` for official single-arm
   defaults and ``left_arm`` / ``right_arm`` / ``left_eef`` / ``right_eef`` for
   dual-arm defaults. Custom names remain valid.
4. **Configure the IK solver** — ``OPWSolverCfg`` (6-DOF), ``SRSSolverCfg`` (7-DOF),
   or a generic ``SolverCfg``.
5. **Set drive properties** — stiffness/damping/max_effort per joint group.
6. **Implement** ``build_pk_serial_chain`` reading from ``_pk_urdf_path``.
7. **Keep** ``from_dict`` as the 3-line template — do not reimplement.
8. **Register** the official preset's lazy export in
   ``embodichain_robots/__init__.py`` and its ID, class import path, defaults,
   and control metadata in ``embodichain_robots/_catalog.py``. External packages
   advertise a catalog through an ``embodichain.robots`` entry point.
9. **Add documentation** — ``docs/source/resources/robot/<name>.md`` + update
   ``resources/robot/index.rst``.
10. **Test** — a ``__main__`` smoke test + the DOF drift guard + ``preview-asset`` CLI.

Refresh the editable install after changing package entry-point metadata, then
run ``embodichain list-robot`` and verify variants, independent DOFs, default
control groups, and mimic followers against the selected URDF/configuration.
Keep catalog modules and package initializers lightweight so discovery does not
load robot implementations or download assets.

Approaches
----------

- **Single-file**: ``<vendor>/<model>.py`` when archives are shared at vendor level.
- **Package**: ``<vendor>/<model>/`` for model-owned assets, revisions or calibrations.
  Keep ``cfg.py``, ``assets.py``, optional specification/type helpers, and a lazy
  ``__init__.py`` together. Asset imports must not load configuration modules.

Asset classes register through ``embodichain_robots/_asset_catalog.py`` and the
``embodichain.assets`` entry point. Each asset ID declares a ``class_path`` and
``category``. Keep source versions, checksums and cache prefixes with the asset;
the core owns reusable download, locking and extraction machinery.

Key parameters
--------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Parameter
     - Type
     - Description
   * - ``uid``
     - str
     - Unique robot identifier
   * - ``urdf_cfg``
     - URDFCfg
     - URDF file and components
   * - ``control_parts``
     - Dict[str, List[str]]
     - Joint groups for control
   * - ``solver_cfg``
     - Dict[str, SolverCfg]
     - IK solver configurations
   * - ``joint_drive_props``
     - JointDrivePropertiesCfg
     - Joint drive, limits, friction
   * - ``attrs``
     - RigidBodyPhysicsCfg
     - Grouped rigid-body physics
   * - variant fields
     - enum / str / bool
     - Optional subclass fields, such as ``version``
   * - ``_pk_urdf_path``
     - property or method → str
     - URDF for the FK/IK serial chain

Common mistakes
---------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Mistake
     - Fix
   * - ``all`` instead of ``__all__``
     - Use ``__all__`` — lowercase ``all`` breaks ``import *``.
   * - ``solver_cfg`` set twice
     - Set it once in ``_build_defaults`` only.
   * - PK URDF drifts from sim URDF
     - Route PK through ``_pk_urdf_path``; keep the DOF guard.
   * - Reimplementing ``from_dict``
     - Keep the 3-line template; put logic in ``_build_defaults``.
   * - ``root_link_name`` as a tuple
     - It must be a ``str``.
   * - Calling a nonexistent ``validate``
     - Don't call methods that don't exist.

See Also
--------

- :doc:`/tutorial/robot` — Using robots in simulation
- :doc:`/overview/sim/motion/solvers/index` — IK solver reference
- :doc:`/resources/robot/index` — Existing robot documentation
