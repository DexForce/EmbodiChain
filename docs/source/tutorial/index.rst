Tutorials
=========

These tutorials walk you through EmbodiChain step by step, from creating your first simulation scene to training RL agents. Each tutorial includes a complete runnable script and a line-by-line explanation.

Suggested Learning Path
~~~~~~~~~~~~~~~~~~~~~~~

Follow the tutorials in this order for the best learning experience:

**Phase 1: Simulation Basics**

1. :doc:`create_scene` — Set up a simulation, add objects, run the render loop,
   and inspect the scene through Viser. **Start here.**
2. :doc:`point_cloud_visualization` — Inspect native point placement and color data.
3. :doc:`create_softbody` and :doc:`create_cloth` — Add deformable bodies to your scenes.
4. :doc:`rigid_object_group` — Manage collections of rigid objects efficiently.
5. :doc:`rigid_constraint` — Attach and detach two rigid objects via a fixed constraint.
6. :doc:`articulation` — Load a passive articulated asset and inspect its joint drives.
7. :doc:`robot` — Load and control a robot in simulation.
8. :doc:`sensor` — Add cameras and capture RGB/depth/segmentation data.
9. :doc:`solver` — Configure IK solvers for end-effector control.
10. :doc:`motion_gen` — Generate smooth trajectories with motion planners.
11. :doc:`robot_articulation` — Plan contact-rich motion to open a passive drawer and push it halfway back.
12. :doc:`atomic_actions` — Use built-in action primitives (move, move joints, pick, move held object, place).
13. :doc:`semantic_skills` — Configure robot-independent calls through Expert Program.
14. :doc:`gizmo` — Interactively control robots with on-screen gizmos.

**Phase 2: Environments**

14. :doc:`basic_env` — Create a simple Gymnasium environment with ``BaseEnv``. Prerequisite: Phase 1 basics.
15. :doc:`modular_env` — Build a config-driven environment with ``EmbodiedEnv``, managers, and randomization. Prerequisite: :doc:`basic_env`.
16. :doc:`data_generation` — Generate expert demonstration datasets for imitation learning. Prerequisite: :doc:`modular_env`.
17. :doc:`rl` — Train RL agents with PPO or GRPO. Prerequisite: :doc:`basic_env`.

**Phase 3: Extending the Framework**

18. :doc:`/guides/add_robot` — Add a new robot model to EmbodiChain.

.. toctree::
   :maxdepth: 1
   :hidden:

   create_scene
   point_cloud_visualization
   create_softbody
   create_cloth
   rigid_object_group
   rigid_constraint
   articulation
   robot
   solver
   sensor
   motion_gen
   robot_articulation
   atomic_actions
   semantic_skills
   gizmo
   basic_env
   modular_env
   data_generation
   rl
