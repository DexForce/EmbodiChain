Planners
=================================

This section documents the planners provided by the project with a focus on 
planners for robotic motion: path planning, trajectory generation,
collision avoidance, and practical considerations such as smoothness and
dynamic feasibility.

The repository contains several planner implementations — each has a dedicated
page with implementation details and examples. Use the links at the bottom of
this page to jump to a specific planner.

.. contents:: Table of contents
    :local:
    :depth: 2

Overview
--------

The `embodichain` project provides a unified interface for robot trajectory planning, supporting both joint space and Cartesian space interpolation. The main planners include:

- **MotionGenerator**: A unified trajectory planning interface that supports joint/Cartesian interpolation, automatic constraint handling, flexible planner selection, and is easily extensible for collision checking and additional planners.
- **ToppraPlanner**: A time-optimal trajectory planner based on the TOPPRA library, supporting joint trajectory generation under velocity and acceleration constraints.
- **TrajectorySampleMethod**: An enumeration for trajectory sampling strategies, supporting sampling by time, quantity, or distance.

These tools can be used to generate smooth and dynamically feasible robot trajectories, and are extensible for future collision checking and various sampling requirements.

See also
--------

.. toctree::
    :maxdepth: 1

    motion_generator.md
    toppra_planner.md
    trajectory_sample_method.md
