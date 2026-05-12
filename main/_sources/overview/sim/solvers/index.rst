Solvers
=================================

This section documents the solvers provided by the project with a focus on
robotic kinematics: forward kinematics (FK), inverse kinematics (IK),
differential (velocity) kinematics, constraint handling and practical
considerations such as singularities and performance tuning.

The repository contains several solver implementations — each has a dedicated
page with implementation details and examples. Use the links at the bottom of
this page to jump to a specific solver.

.. contents:: Table of contents
    :local:
    :depth: 2

Overview
--------

Robotic kinematics solvers translate between joint-space and task-space.

- Forward kinematics (FK) maps joint values q to an end-effector pose.
- Inverse kinematics (IK) finds joint values q that achieve a desired end-effector
   pose.


Forward kinematics
-------------------

Forward kinematics composes joint transforms according to the robot's
kinematic tree to produce the end-effector transform. Practical builders compute these transforms efficiently using the robot's
URDF or internal kinematic model. FK solvers in `embodichain` are
optimized for batch evaluation and for returning both pose and link frames.

Inverse kinematics
-------------------

Inverse kinematics is the core topic for robotics. There are two common
approaches implemented in the repository:

- Analytical IK (closed-form): when the robot geometry admits a closed-form
   solution (e.g., many 6-DOF industrial arms), these solvers return exact
   solutions quickly and deterministically.
- Numerical IK: general-purpose methods based on the Jacobian or optimization
   that work for arbitrary kinematic chains but may be slower and require
   a good initial guess.

Analytical IK
~~~~~~~~~~~~~

Analytical solvers (see the OPW) exploit kinematic
structure to derive algebraic inverse mappings. Benefits include:

- very fast runtime
- exact solutions when they exist

Limitations:

- only available for specific robot families and joint arrangements

Numerical IK (Jacobian-based and optimization)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Numerical IK methods iteratively update joint values q to reduce pose error.
Jacobian-based updates use the task Jacobian J(q) to relate changes in joint
space to end-effector motion.


Multi-chain and closed-loop kinematics
-------------------------------------

Solvers can handle serial chains, branched kinematic trees and some closed-loop
mechanisms. Closed-loop systems commonly require constraint solvers and may
embed loop-closure constraints in the solver as equality constraints.


Choosing a solver
-----------------

- Use analytic solvers (OPW for 6-DOF arms or SRS for 7-DOF arms) when available for speed and
   determinism.
- Use numerical solvers (PyTorch/optimization, Differential) when you need
   flexibility..

See also
--------

.. toctree::
    :maxdepth: 1

    pytorch_solver.md
    differential_solver.md
    pink_solver.md
    pinocchio_solver.md
    opw_solver.md
    srs_solver.md
