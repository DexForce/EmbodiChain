Toolkits
========

The :mod:`embodichain.toolkits` package provides standalone utilities for
preparing robot assets and generating manipulation data. These tools complement
the simulation APIs: they can be used in preprocessing scripts, from the
command line, or as part of an EmbodiChain workflow.

Available Toolkits
------------------

.. list-table::
   :header-rows: 1
   :widths: 24 44 32

   * - Toolkit
     - What it does
     - Typical use
   * - :doc:`URDF Convex Decomposition <convex_decomposition>`
     - Decomposes concave URDF meshes into convex collision hulls, with
       optional model scaling and inertia recomputation.
     - Prepare imported URDF assets for stable physics simulation.
   * - :doc:`URDF Assembly <urdf_assembly>`
     - Combines component URDFs, attaches sensors, normalizes names, and writes
       a unified robot description.
     - Build modular or multi-part robots such as arm-and-gripper systems.
   * - :doc:`Parallel-Gripper Grasp Generation <grasp_generator>`
     - Samples antipodal contacts, constructs grasp poses, and filters
       collisions for parallel-jaw grippers.
     - Annotate graspable regions and generate candidate grasps for manipulation.

Choosing a Toolkit
------------------

Use convex decomposition when a source URDF already represents the complete
robot but its collision meshes are unsuitable for simulation. Use URDF assembly
when the robot is distributed across multiple component files. Use grasp
generation after the target object's mesh is ready and a manipulation workflow
needs feasible end-effector poses.

The asset tools can be chained: preprocess component collision meshes, assemble
the components into one URDF, and then load the resulting robot in a grasping
task.

.. toctree::
   :maxdepth: 1

   URDF Convex Decomposition <convex_decomposition>
   URDF Assembly <urdf_assembly>
   Parallel-Gripper Grasp Generation <grasp_generator>
