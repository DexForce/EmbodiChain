.. _tutorial_writing_demo_scripts:

Writing Demo Scripts
====================

Simulation demo scripts (under ``scripts/tutorials/`` and ``examples/sim/demo/``) share a
lot of boilerplate: argument parsing, simulation setup, window/recording management,
trajectory replay, and cleanup. EmbodiChain ships a small **demo utility layer** that
factors this boilerplate out so new demos can focus on their core *setup* and *run* logic.

The layer has two parts:

- :mod:`embodichain.lab.sim.utility.demo_utils` - standalone helper functions and the
  ``DemoRecording`` context manager. Usable from any script, flat or structured.
- :mod:`embodichain.lab.sim.demo_base` - an optional ``DemoBase`` lifecycle base class
  for demos with a clear setup / run / cleanup structure.

.. note::
    These helpers build on top of
    :func:`~embodichain.lab.gym.utils.gym_utils.add_env_launcher_args_to_parser`; they
    add demo-specific flags (``--auto_play``, ``--record_steps``, ``--record_fps``,
    ``--record_save_path``, ``--no_vis_eef_axis``) rather than replacing it. Common
    multi-word options accept both underscore and hyphen spellings, for example
    ``--record_steps`` and ``--record-steps``.

Command-line Arguments
----------------------

:func:`~embodichain.lab.sim.utility.demo_utils.add_demo_args` extends a parser with the
standard launcher arguments plus the demo flags above:

.. code-block:: python

    import argparse

    from embodichain.lab.sim.utility.demo_utils import add_demo_args

    parser = argparse.ArgumentParser(description="My demo.")
    parser = add_demo_args(parser)
    args = parser.parse_args()

``--auto_play`` skips every interactive ``input()`` prompt (see
:ref:`the helpers <demo-window-helpers>`) so the demo can run end-to-end in CI or
headless. For continuous demos it also selects a finite default run length.
``--record_steps`` enables video recording and gives continuous demos an explicit
simulation-update limit (see :ref:`recording <demo-recording>`).

Creating and Tearing Down the Simulation
-----------------------------------------

:func:`~embodichain.lab.sim.utility.demo_utils.create_default_sim` builds a
:class:`~embodichain.lab.sim.SimulationManager` from the parsed ``args``
(``headless``, ``device``, ``renderer``, ``gpu_id``) with sensible defaults and
optionally a main light. Pass ``num_envs`` for parallel-environment demos, and
``add_default_light=False`` if you want to set up your own lighting.

:func:`~embodichain.lab.sim.utility.demo_utils.shutdown_sim` finishes any active
recording and calls ``sim.destroy()``.
**Always** destroy the simulation before the process exits, otherwise the interpreter
segfaults (exit 139) during teardown.

.. _demo-window-helpers:

Window, User and GPU Helpers
----------------------------

These read the parsed ``args`` and become no-ops when the relevant flag is set, so the
same code path works for interactive and ``--auto_play`` runs:

- :func:`~embodichain.lab.sim.utility.demo_utils.maybe_open_window` - opens the viewer
  unless ``--headless``.
- :func:`~embodichain.lab.sim.utility.demo_utils.maybe_init_gpu_physics` - calls
  ``sim.init_gpu_physics()`` only when the sim is configured for GPU physics.
- :func:`~embodichain.lab.sim.utility.demo_utils.maybe_wait_for_user` - blocks on
  ``input(prompt)`` unless ``--auto_play``.
- :func:`~embodichain.lab.sim.utility.demo_utils.maybe_pause_for_inspection` - the same,
  with an end-of-demo prompt.
- :func:`~embodichain.lab.sim.utility.demo_utils.resolve_demo_steps` - resolves an
  explicit recording limit or a finite ``--auto_play`` limit while leaving interactive
  runs unbounded.
- :func:`~embodichain.lab.sim.utility.demo_utils.run_simulation_loop` - runs the common
  update/FPS loop with optional sleep, step callback, and maximum update count.

.. _demo-recording:

Recording
---------

:class:`~embodichain.lab.sim.utility.demo_utils.DemoRecording` is a context manager that
starts window recording when ``args.record_steps`` is set, generates a timestamped file
name under ``--record_save_path`` (default ``./recordings``), and stops + flushes the
video on exit. A path ending in ``.mp4`` is used as the exact output path; other paths are
treated as output directories. Headless runs use a default fixed camera when the caller
does not provide ``look_at``. If recording fails to start it warns and continues instead
of aborting the demo.

.. code-block:: python

    with DemoRecording(sim, args, prefix="my_demo"):
        # ... run the demo; frames are captured here ...
        replay_trajectory(sim, robot, traj)

Replaying a Trajectory
----------------------

:func:`~embodichain.lab.sim.utility.demo_utils.replay_trajectory` steps a joint-space
trajectory through the simulator, applying each waypoint with ``robot.set_qpos`` and
``sim.update(step=...)``, then holds the final configuration for ``post_steps``. It
accepts 1-D ``(num_joints,)``, 2-D ``(num_steps, num_joints)`` or 3-D
``(batch, num_steps, num_joints)`` tensors.

The ``DemoBase`` Lifecycle Class
--------------------------------

For demos with a clear structure, subclass
:class:`~embodichain.lab.sim.demo_base.DemoBase` and implement :meth:`setup` and
:meth:`run`. The base class stores ``args``, guarantees :meth:`cleanup` (``sim.destroy()``)
runs in a ``finally`` block, and exposes :meth:`main` to drive the lifecycle:

.. code-block:: python

    import argparse

    import torch

    from embodichain.lab.sim.demo_base import DemoBase
    from embodichain.lab.sim.utility.demo_utils import (
        DemoRecording,
        add_demo_args,
        create_default_sim,
        maybe_open_window,
        maybe_pause_for_inspection,
        maybe_wait_for_user,
        replay_trajectory,
        setup_print_options,
    )


    class MyDemo(DemoBase):
        def setup(self) -> None:
            self.sim = create_default_sim(self.args)
            maybe_open_window(self.sim, self.args)
            self.robot = self.sim.add_robot(cfg=...)
            # ... build the rest of the scene ...

        def run(self) -> None:
            maybe_wait_for_user(self.args, "Press Enter to plan...")
            traj = ...  # plan the trajectory
            with DemoRecording(self.sim, self.args, prefix="my_demo"):
                replay_trajectory(self.sim, self.robot, traj)
            maybe_pause_for_inspection(self.args)


    def main() -> None:
        setup_print_options()
        parser = add_demo_args(argparse.ArgumentParser(description="My demo."))
        args = parser.parse_args()
        MyDemo(args).main()


    if __name__ == "__main__":
        main()

Because :meth:`~embodichain.lab.sim.demo_base.DemoBase.main` wraps both :meth:`setup`
and :meth:`run` in ``try / finally``, the simulation is always destroyed - even if setup
fails after allocating the simulation, :meth:`run` raises, or the user interrupts with
``Ctrl+C``.

.. tip::
    Demos that keep the viewer open until ``Ctrl+C`` should use
    ``run_simulation_loop(..., max_steps=resolve_demo_steps(args))`` so interactive runs
    stay open while ``--auto_play`` and ``--record_steps`` runs terminate automatically.

Reference Implementations
-------------------------

The migrated demo scripts are good references:

- ``scripts/tutorials/atomic_action/`` - the atomic-action tutorials use the shared
  argument, recording, and cleanup conventions; the single-arm demos use ``DemoBase``.
- ``examples/sim/demo/`` - ``press_softbody``, ``pick_up_cloth``, ``grasp_cup_to_caffe``
  and ``scoop_ice`` use ``DemoBase`` for lifecycle and the shared helpers for args,
  recording and cleanup.
- ``scripts/tutorials/sim/`` and the focused examples under ``examples/sim/`` show the
  flat-function style for short scene, sensor, solver, gizmo, and planner demos.
