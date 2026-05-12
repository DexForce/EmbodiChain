Tutorials
=========

These tutorials walk you through EmbodiChain step by step, from creating your first simulation scene to training RL agents. Each tutorial includes a complete runnable script and a line-by-line explanation.

Suggested Learning Path
~~~~~~~~~~~~~~~~~~~~~~~

Follow the tutorials in this order for the best learning experience:

**Phase 1: Simulation Basics**

1. :doc:`create_scene` — Set up a simulation, add objects, and run the render loop. **Start here.**
2. :doc:`create_softbody` and :doc:`create_cloth` — Add deformable bodies to your scenes.
3. :doc:`rigid_object_group` — Manage collections of rigid objects efficiently.
4. :doc:`robot` — Load and control a robot in simulation.
5. :doc:`sensor` — Add cameras and capture RGB/depth/segmentation data.
6. :doc:`solver` — Configure IK solvers for end-effector control.
7. :doc:`motion_gen` — Generate smooth trajectories with motion planners.
8. :doc:`atomic_actions` — Use built-in action primitives (pick, place, move).
9. :doc:`gizmo` — Interactively control robots with on-screen gizmos.

**Phase 2: Environments**

10. :doc:`basic_env` — Create a simple Gymnasium environment with ``BaseEnv``. Prerequisite: Phase 1 basics.
11. :doc:`modular_env` — Build a config-driven environment with ``EmbodiedEnv``, managers, and randomization. Prerequisite: :doc:`basic_env`.
12. :doc:`data_generation` — Generate expert demonstration datasets for imitation learning. Prerequisite: :doc:`modular_env`.
13. :doc:`rl` — Train RL agents with PPO or GRPO. Prerequisite: :doc:`basic_env`.

**Phase 3: Extending the Framework**

14. :doc:`add_robot` — Add a new robot model to EmbodiChain.

.. toctree::
   :maxdepth: 1
   :hidden:

   create_scene
   create_softbody
   create_cloth
   rigid_object_group
   robot
   add_robot
   solver
   sensor
   motion_gen
   atomic_actions
   gizmo
   basic_env
   modular_env
   data_generation
   rl
