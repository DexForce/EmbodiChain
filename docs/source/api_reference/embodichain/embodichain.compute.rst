embodichain.compute
===================

Shared array and tensor computations
------------------------------------

``compute`` owns numerical algorithms that operate on arrays, tensors, and
kinematic parameters without importing the simulation runtime. It does not
own robots, scenes, or environment lifecycle. Importing the root package does
not load Torch or Warp. Domain packages load their required dependencies.

- ``kinematics/_warp`` implements analytical OPW, SRS, and UR computations.
  Stateful solver interfaces remain in ``embodichain.lab.sim.solvers``.
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

.. code-block:: python

   import torch
   from embodichain.compute.trajectory import interpolate_with_distance

   keyframes = torch.tensor([[[0.0], [1.0], [3.0]]])
   result = interpolate_with_distance(keyframes, interp_num=5, device="cpu")
   # result: [[[0.0], [0.5], [1.0], [2.0], [3.0]]]

.. currentmodule:: embodichain.compute.trajectory

.. autosummary::

   differentiate_positions
   resample_in_time
   interpolate_with_distance
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

Implementation modules
----------------------

.. automodule:: embodichain.compute.trajectory.interpolation
   :members:

.. automodule:: embodichain.compute.trajectory.resampling
   :members:

.. automodule:: embodichain.compute.trajectory.timing
   :members:

.. automodule:: embodichain.compute.trajectory.warping
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
