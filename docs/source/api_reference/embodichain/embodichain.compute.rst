embodichain.compute
===================

Shared array and tensor computations
------------------------------------

``compute`` owns numerical algorithms that operate on arrays, tensors, and
kinematic parameters without importing the simulation runtime. It does not
own robots, scenes, or environment lifecycle. Importing the root package does
not load Torch or Warp. Domain packages load their required dependencies.

- ``kinematics/_warp`` implements analytical OPW, SRS, and UR computations.
  Stateful solver interfaces remain in ``embodichain.lab.sim.motion.solvers``.
- ``trajectory`` provides the public tensor interfaces below. Its private
  ``_warp`` implementation supports path resampling and trajectory warping.
- ``geometry/_warp/convex_query.py`` evaluates maximum halfspace values for
  convex hulls. These values classify containment; outside values are not
  generally exact Euclidean distances to the hull.
- ``image/_warp/tiling.py`` converts tiled images into image batches.

Private ``_warp`` modules are implementation details, not a backend registry.
Simulation-specific contact scattering belongs to
``embodichain.lab.sim.sensors._warp.contact``.

Trajectory API
--------------

``interpolate_with_distance`` retains every required keyframe.
``resample_with_distance`` samples uniformly along cumulative path distance
and may omit interior input samples. ``warp_trajectory_qpos`` changes a joint
trajectory using interpolated offsets at supplied keyframes; warping here
means trajectory deformation.

Use ``resample_with_distance(..., backend="torch")`` for dtype-preserving
Torch resampling without entering Warp. The default ``auto`` path retains its
float32 Warp implementation and Torch fallback. Both operate on batched paths;
stationary paths and repeated endpoints remain valid inputs.

.. code-block:: python

   import torch
   from embodichain.compute.trajectory import interpolate_with_distance

   keyframes = torch.tensor([[[0.0], [1.0], [3.0]]])
   result = interpolate_with_distance(keyframes, interp_num=5, device="cpu")
   # result: [[[0.0], [0.5], [1.0], [2.0], [3.0]]]

.. currentmodule:: embodichain.compute.trajectory

.. autosummary::

   retime_joint_paths
   differentiate_positions
   resample_in_time
   interpolate_with_distance
   interpolate_minimum_jerk
   interpolate_with_nums
   resample_with_distance
   sort_and_padding_key_frame
   warp_trajectory_qpos

.. automodule:: embodichain.compute.trajectory
   :members:
   :imported-members:

Timed trajectories use ``dt`` arrival intervals. ``differentiate_positions``
uses nonuniform central differences and one-sided endpoints, accepting only
unchanged positions at repeated timestamps. It does not impose rest boundaries
or motion limits. ``resample_in_time`` preserves endpoints and total duration
while sampling the original time profile; callers must recompute derivatives
after changing samples or timing.

``retime_joint_paths`` fits and times a joint path with explicit shared velocity
and acceleration budgets. Its plain tensor tuple retains gradients
and explicit validity/length tensors. Duration is the sum of arrival intervals;
solver success does not certify geometric task success.

Implementation modules
----------------------

.. automodule:: embodichain.compute.trajectory.toppra
   :members:


.. automodule:: embodichain.compute.trajectory.interpolation
   :members:

.. automodule:: embodichain.compute.trajectory.resampling
   :members:

.. automodule:: embodichain.compute.trajectory.timing
   :members:

.. automodule:: embodichain.compute.trajectory.warping
   :members:

Kinematics API
--------------

``yoshikawa_manipulability`` returns ``sqrt(det(J @ J^T))`` per batched
Jacobian; ``condition_number`` returns the ratio of largest to smallest
singular value. ``select_jacobian_rows`` extracts task-specific row subsets
(for example the translational rows) before scoring. All operate on batched
Torch tensors of shape ``(N, R, DOF)`` and preserve input dtype and device.
Workspace analysis and IK candidate ranking share this single numerical
implementation; aggregation stays in the consuming layer.

.. code-block:: python

   import torch
   from embodichain.compute.kinematics import (
       select_jacobian_rows,
       yoshikawa_manipulability,
   )

   jacobians = torch.randn(8, 6, 7)
   scores = yoshikawa_manipulability(jacobians)
   translational = yoshikawa_manipulability(
       select_jacobian_rows(jacobians, "translational")
   )

.. currentmodule:: embodichain.compute.kinematics

.. autosummary::

   condition_number
   select_jacobian_rows
   yoshikawa_manipulability

.. automodule:: embodichain.compute.kinematics.manipulability
   :members:

Migration
---------

Existing imports from ``embodichain.utils.warp`` and its submodules continue
to resolve to the relocated Warp kernel and struct objects. They do not
register duplicate implementations. The legacy contact-kernel alias loads
the simulation sensor package on demand.

Pure trajectory functions formerly defined in
``embodichain.lab.sim.utility.action_utils`` are re-exported there for
compatibility. New consumers should import from ``compute.trajectory``.
``get_trajectory_object_offset_qpos`` remains in the simulation utility module
because it calls a stateful solver's FK and IK interfaces.

Geometry tensor kernels
-----------------------

.. toctree::
   :maxdepth: 1

   embodichain.compute.scene_geometry

Differentiable kinematics
-------------------------

The fixed-step DLS implementation preserves target and seed gradients through
both FK and the geometric Jacobian. ``SerialChainKinematics`` adapts a copied
PK chain without loading simulation. Its ``forward_kinematics`` method computes
only TCP poses, while ``evaluate`` also constructs differentiable Jacobians in
the same FK pass. ``DifferentiableIKCfg`` configures the
compute function and Torch layer; failed solves retain their last iterate.

.. currentmodule:: embodichain.compute.kinematics

.. autosummary::

   damped_least_squares_step
   solve_dls_ik
   pose_error
   SerialChainKinematics
   DifferentiableIKCfg
   DifferentiableIKResult

.. automodule:: embodichain.compute.kinematics.dls
   :members:

.. automodule:: embodichain.compute.kinematics.pose_error
   :members:

.. automodule:: embodichain.compute.kinematics.serial_chain
   :members:

.. automodule:: embodichain.compute.kinematics.types
   :members:
