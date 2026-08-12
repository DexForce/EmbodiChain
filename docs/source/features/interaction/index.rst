Interactive Simulation
======================

Interactive Simulation covers the frontends and controls used to inspect or
manipulate a running simulation. Task-specific commands such as asset preview,
workspace visualization, and grasp annotation remain in their own guides or
feature sections.

Interaction Frontends
---------------------

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Frontend
     - Use it for
     - Documentation
   * - Native DexSim window
     - Local camera navigation, selection, keyboard and mouse input, viewer
       recording, and custom window events.
     - :doc:`Native window interaction <window>`
   * - Viser browser
     - Headless or remote scene inspection, camera previews, environment
       visibility, overlays, and trusted browser controls.
     - :doc:`Viser browser visualization </overview/sim/viser_visualization>`

The native window and Viser are mutually exclusive visualization frontends.
Both can expose the same registered Gizmo targets.

Interaction Controls
--------------------

- :doc:`Interactive Gizmos <gizmo>` provide cross-frontend transform controls
  for robots, rigid objects, and cameras.
- :ref:`Interactive replay <run-env-interactive-replay>` lets users scrub a
  recorded trajectory from the terminal or the Viser frame slider.

Related Workflows
-----------------

- :doc:`Preview an asset </guides/preview_asset>` to inspect rigid objects and
  articulations without creating a Gym environment.
- :doc:`Visualize a robot workspace </features/workspace_analyzer/visualizers>`
  through its dedicated analyzer backends.
- :doc:`Annotate grasp regions </features/toolkits/grasp_generator>` with the
  grasp-generation toolkit's browser interface.

.. toctree::
   :maxdepth: 2

   Native window interaction <window.md>
   Interactive Gizmos <gizmo.md>
