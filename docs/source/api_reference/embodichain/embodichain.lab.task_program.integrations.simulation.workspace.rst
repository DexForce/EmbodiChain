embodichain.lab.task_program.integrations.simulation.workspace
================================================================

This adapter turns valid robot workspace TCP samples into object-pose proposals
using an explicit object-local grasp transform, and checks manipulation targets
with actual pose IK. It requires one environment row and a selected control part.

.. currentmodule:: embodichain.lab.task_program.integrations.simulation.workspace

.. autosummary::
   :nosignatures:

   RobotSceneWorkspace
   SceneWorkspaceCandidate

Object and manipulation frames
-------------------------------

Let ``T_object_tcp`` map TCP coordinates into the object frame. The
``object_grasp_pose`` parameter uses this convention. Sampling derives:

.. code-block:: text

   T_arena_object = T_arena_tcp @ inverse(T_object_tcp)
   T_arena_tcp = T_arena_object @ T_object_tcp

All resulting poses use the local Z-up arena, excluding replication offsets.
``Robot.sample_reachable_pose`` already uses FK with the current base; callers
must not apply that base transform or a Scene Engine coordinate conversion
again.

Sampling and pose checks
------------------------

``sample_object_poses`` uses a device-local seeded generator, preserves the
workspace validity mask, and returns only valid cache entries. Optional bounds
apply to object origins after the grasp transform is inverted. Exhausting the
budget returns fewer proposals. Cache/configuration errors propagate rather
than being hidden as ordinary rejections.

Each ``SceneWorkspaceCandidate`` retains its immutable pose proposal, joint
seed, cache index, and optional score. Scores are available to the caller;
this method does not rank or certify the proposals. Sampling does not establish
support, stable placement, or collision-free motion.

``check_object_pose`` composes the actual object and local grasp transforms,
then calls the selected control part's pose IK. It also works for poses absent
from the sparse cache. The returned mandatory ``workspace.pose_ik`` check applies
to this manipulation target only. Changing an object's height, orientation, or
settled position requires another check, and grasp/pre-grasp/lift/place/retreat
targets require their own checks. Motion between those poses and measured task
success remain separate validation stages.

Producing candidates for the shared source
------------------------------------------

The grasp transform is supplied from the application's object affordance:

.. code-block:: python

   import torch

   from embodichain.lab.sim.scene_expansion import (
       FixedSceneCandidateSource,
       SceneExpansionCfg,
   )
   from embodichain.lab.task_program.integrations.simulation.workspace import (
       RobotSceneWorkspace,
   )

   def propose_cube_layouts(
       workspace: RobotSceneWorkspace,
       object_grasp_pose: torch.Tensor,
   ) -> FixedSceneCandidateSource:
       candidates = workspace.sample_object_poses(
           "cube", object_grasp_pose, num_samples=3, seed=7, max_attempts=64
       )
       return FixedSceneCandidateSource(
           SceneExpansionCfg(seed=7, max_candidates=3, movable_entity_ids=("cube",)),
           parent_scene_id="pick-place-scene:revision-1",
           candidates=((candidate.pose_change,) for candidate in candidates),
           metadata=(("source", "robot-workspace"),),
       )

Construct the adapter as ``RobotSceneWorkspace(env.robot, control_part="arm")``
using the environment's actual configured control-part name. After physical
preparation, use the settled object's pose in ``check_object_pose`` and combine
that result with task-specific initial-state checks. Evaluate each proposal
through :mod:`embodichain.lab.task_program.integrations.scene_expansion` and the
concrete scene host; never replace measured acceptance with workspace success.

Public interfaces
-----------------

.. autoclass:: RobotSceneWorkspace
   :members:

.. autoclass:: SceneWorkspaceCandidate
   :members:
