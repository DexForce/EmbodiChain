embodichain.lab.sim.utility
===================================

.. automodule:: embodichain.lab.sim.utility

Overview
--------

This package contains helper utilities for simulation state conversion,
mesh/geometry handling, configuration transforms, keyboard interaction, and
action/solver adaptation.

.. rubric:: Submodules

.. autosummary::

   action_utils
   atom_action_utils
   cfg_utils
   grasp_geometry
   import_utils
   io_utils
   keyboard_utils
   sim_utils
   mesh_utils
   solver_utils
   tensor

.. currentmodule:: embodichain.lab.sim.utility

Action Utilities
~~~~~~~~~~~~~~~~

.. automodule:: embodichain.lab.sim.utility.action_utils
   :members:

Atomic Action Utilities
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain.lab.sim.utility.atom_action_utils
   :members:

Configuration Utilities
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain.lab.sim.utility.cfg_utils
   :members:

Sampled Hand Geometry
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: embodichain.lab.sim.utility.grasp_geometry

These adapters own named robot FK and explicit hand/TCP calibration. A hand
capture reads the first measured posture without stepping or modifying the
robot. Hand commands must cover the selected moving joints, including mimic
followers, and must leave the declared motion TCP fixed.

The paired provider captures once on its first paired query and belongs to one
Robot/environment adapter. It may be retained across bridge, engine and episode
resets while that model, calibration and hand commands remain unchanged;
single-arm calls do not trigger capture. Concrete sampling and contact fitting
remain data-only GraspKit services.

``RobotCoordinatedGraspProvider.clear_cache()`` releases asset-query caches
without reading the robot or discarding captured hand geometry. It is also safe
before the first query. Changes to the model, commands or TCP calibration still
require a new provider.

.. autosummary::
   :nosignatures:

   RobotHandGeometryCfg
   ordered_pad_line_world
   build_parallel_jaw_hand_geometry
   RobotHandGeometryProvider
   RobotCoordinatedGraspProvider

.. autoclass:: RobotHandGeometryCfg
   :members:

.. autofunction:: ordered_pad_line_world

.. autofunction:: build_parallel_jaw_hand_geometry

.. autoclass:: RobotHandGeometryProvider
   :members:

.. autoclass:: RobotCoordinatedGraspProvider
   :members:


Import Utilities
~~~~~~~~~~~~~~~~

.. automodule:: embodichain.lab.sim.utility.import_utils
   :members:

I/O Utilities
~~~~~~~~~~~~~

.. automodule:: embodichain.lab.sim.utility.io_utils
   :members:

Keyboard Utilities
~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain.lab.sim.utility.keyboard_utils
   :members:

Simulation Utils
~~~~~~~~~~~~~~~~

.. automodule:: embodichain.lab.sim.utility.sim_utils
   :members:

Mesh Utils
~~~~~~~~~~

.. automodule:: embodichain.lab.sim.utility.mesh_utils
   :members:

Solver Utilities
~~~~~~~~~~~~~~~~

.. automodule:: embodichain.lab.sim.utility.solver_utils
   :members:

Tensor Utilities
~~~~~~~~~~~~~~~~

.. automodule:: embodichain.lab.sim.utility.tensor
   :members:

Authored URDF collision inputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The reader returns the existing Spawn ``LinkMesh`` values for explicit local
mesh/box colliders in selected link order. Collision scale and origin precede
body scaling, applied once. Arrays are independently owned and read-only.
``vertices_only`` is an explicit vertex envelope; it cannot establish surface or
native-cooked geometry. Visuals, unresolved URIs, implicit link selection and
shape substitution are unsupported. Reading constructs no Robot or simulator.

.. currentmodule:: embodichain.lab.sim.utility.urdf_collision

.. autofunction:: load_urdf_collision_inputs

Explicit hand collision and preshape adapters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These opt-in Robot adapters consume the existing explicit hand/TCP calibration
and full B=1 joint observation without moving the robot. Model-owned finger
names and commands remain in the robot configuration. Authored collider vertices,
Robot link meshes and sampled GraspKit surfaces retain their separate provenance.

``HandCollisionTipCalibration`` retains the original twelve-step closed-side
bisection and complete finger roster. The native link-mesh tip reader preserves
its original closed command and mesh source. ``capture_preshape_hand_clearance``
keeps the source seven-state sweep, point order, aperture and retreat reserve;
the caller still owns candidate selection and physical effect verification.

.. currentmodule:: embodichain.lab.sim.utility.hand_tip

.. autosummary::

   HandCollisionTipCalibration
   closed_native_hand_tip_offset

.. autoclass:: HandCollisionTipCalibration
   :members:

.. autofunction:: closed_native_hand_tip_offset

.. currentmodule:: embodichain.lab.sim.utility.preshape_geometry

.. autofunction:: capture_preshape_hand_clearance
