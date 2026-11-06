Shared source collision queries
===============================

Scene captures own authored robot collision inputs and borrow native robot and
support instances. Motion consumes those captures for discrete command-frame
envelope checks. GraspKit measures explicit contact point sets without reading
robot or simulator state.

These services preserve the existing source-envelope behavior. They do not
replace cuRobo trajectory validation or the Newton convex collision queries.
Support-box overlap is a conservative rejection, not proof of mesh penetration;
a supported clearance result covers only the supplied frames and geometry.

Robot and support captures
--------------------------

.. currentmodule:: embodichain.lab.sim.scene.robot_collision

.. autosummary::

   RobotCollisionInputs
   SupportCollisionInputs

.. autoclass:: RobotCollisionInputs
   :members:

.. autoclass:: SupportCollisionInputs
   :members:

Captures require one Default environment and exact native owners. The robot
scope includes the complete configured solver-root collision subtree. A support
may be captured before its native body is available; reading incomplete bounds
remains unavailable. Source, native-owner and solver-frame changes reject rather
than replacing the original baseline. Closing a capture releases owned buffers
and references without destroying borrowed physical objects.

Discrete path checks
--------------------

.. currentmodule:: embodichain.lab.sim.motion.collision_query

.. autosummary::

   RobotPathQuery
   SupportPathReport

.. autoclass:: RobotPathQuery
   :members:

.. autoclass:: SupportPathReport
   :members:

``RobotPathQuery`` binds an explicit observed root and ``NativeQueryScope`` to
robot geometry. It requires neither hand calibration nor an articulated target.
``inspect_support_path`` checks caller-selected full-robot command frames against
the support's collider-union AABB, subtracting both native contact skins. Robot
and support captures must belong to the same simulation. Optional required links
are explicit, so the service infers no pad names or action phases.

Reports distinguish unavailable inputs from a supported accepted/rejected
screen. The original revolute API composes this query and retains its public
``RevolutePathReport`` type and metadata. The host owns observation timing and
must close a query before advancing the observation. Closing a query leaves its
borrowed captures reusable.

Contact point measurements
--------------------------

.. currentmodule:: embodichain.toolkits.graspkit.pg_grasp.contact_metrics

.. autosummary::

   sample_contact_vertices
   pad_proximity_metrics

.. autofunction:: sample_contact_vertices

.. autofunction:: pad_proximity_metrics

The caller supplies the sample cap, parent-near threshold, frames and qualified
point sets. Sampling preserves ordered source indices. Proximity measurements
return the near-parent vertex fraction and the largest individual pad's nearest
target distance. They select no candidate ranking and establish no physical
contact. Lab retains FK, frame conversion and joint-displacement measurements;
the specialized revolute query retains its existing caps and thresholds.
