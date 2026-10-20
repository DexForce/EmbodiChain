Robot Workspace
===============

The ``embodichain.lab.sim.motion.workspace`` package provides offline workspace
analysis, persistent cache management, visualization, and runtime sampling for
task-environment randomization.

For an end-to-end example, see the
:doc:`red/green scene initialization showcase </tutorial/robot_workspace>`:
visualize fixed-grasp reachability on a round table, reject a red start, and
execute Task Program pick-and-place trials from green starts.

.. toctree::
   :maxdepth: 2

   Overview and Usage <workspace_analyzer.md>
   Runtime Sampling <runtime.md>
   CLI <cli.md>
   Configuration System <configs.md>
   Sampling Strategies <samplers.md>
   Visualization Options <visualizers.md>
   Caching System <caches.md>
   Constraint Handling <constraints.md>
   Metric Computation <metrics.md>
