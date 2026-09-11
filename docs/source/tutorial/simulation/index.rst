Simulation and Control
======================

Start here to learn EmbodiChain's simulation objects, sensors, robot control,
motion generation, Atomic Skills, and provider-independent task authoring. The
pages are ordered as a progressive learning path.

.. toctree::
   :maxdepth: 1

   ../create_scene
   ../point_cloud_visualization
   ../create_softbody
   ../create_cloth
   ../rigid_object_group
   ../rigid_constraint
   ../articulation
   ../robot
   ../sensor
   ../solver
   ../motion_gen
   ../trajectory_planning
   ../robot_articulation
   ../atomic_actions
   ../task_program_python
   ../gizmo

Additional runnable scripts
----------------------------

The repository also includes focused scripts that are intentionally kept
standalone rather than expanded into separate narrative pages:

* ``scripts/tutorials/grasp/grasp_generator.py`` demonstrates antipodal grasp
  annotation and trajectory generation.
* ``scripts/tutorials/sim/import_usd.py`` and
  ``scripts/tutorials/sim/export_usd.py`` demonstrate USD scene I/O.
* ``scripts/tutorials/visualization/viser_scene.py`` is a lower-level Viser
  runtime example; its options and expected browser endpoint are documented in
  ``scripts/tutorials/visualization/README.md``.
