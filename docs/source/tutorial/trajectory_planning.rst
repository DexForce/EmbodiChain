Trajectory planning diagnostics
===============================

The motion-planning tutorial scripts provide small, reproducible examples for
the trapezoidal and Double-S time laws added to EmbodiChain. They complement
the :doc:`motion_gen` tutorial: the diagnostic scripts exercise the planner
directly, while ``motion_generator.py`` shows planner integration with a robot.

Scalar profile
--------------

``trapezoidal_profile.py`` does not start a simulator. It plots position,
velocity, acceleration, and sampled jerk for one scalar move:

.. literalinclude:: ../../../scripts/tutorials/sim/planner/trapezoidal_profile.py
   :language: python
   :linenos:

Run it from the repository root (``--no-show-plot`` is useful on headless
machines):

.. code-block:: bash

   python scripts/tutorials/sim/planner/trapezoidal_profile.py \
       --profile acceleration_trapezoidal \
       --distance 0.1 \
       --samples 501 \
       --no-show-plot

The accepted profile names are ``velocity_trapezoidal`` and
``acceleration_trapezoidal``. The latter is the jerk-limited Double-S profile.

Planner and Cartesian path
--------------------------

``trapezoidal_planner.py`` runs the same time laws through
``MotionGenerator``. Its ``--path`` option selects ``joint``, ``cartesian``,
or ``both``; ``--profile`` selects a profile (or ``both``). It uses the
standard environment launcher options, including ``--headless``,
``--num-envs``, and ``--device``.

.. code-block:: bash

   python scripts/tutorials/sim/planner/trapezoidal_planner.py \
       --path both \
       --profile acceleration_trapezoidal \
       --headless \
       --no-show-plot

Use ``--plot-output outputs/trajectory.png`` to save the diagnostic figure.
The script's complete option list is kept in the executable source:

.. literalinclude:: ../../../scripts/tutorials/sim/planner/trapezoidal_planner.py
   :language: python
   :start-at: def parse_args
   :end-before: def build_demo_waypoints

Regression checks and plots
---------------------------

The lightweight checks do not create a simulator. Run one scenario at a time;
valid choices are ``bezier``, ``trapezoidal``, ``double-s``, ``blend``,
``minimum-duration``, ``batch``, ``se3``, and ``backend``:

.. code-block:: bash

   python scripts/tutorials/sim/planner/trajectory_pr_checks.py backend

``trajectory_pr_plots.py`` writes a PNG for one of ``bezier``, ``trapezoidal``,
``double-s``, ``blend``, ``minimum-duration``, or ``se3``. Add ``--show`` only
when a graphical display is available:

.. code-block:: bash

   python scripts/tutorials/sim/planner/trajectory_pr_plots.py \
       double-s \
       --output outputs/trajectory_plots/double-s.png

.. literalinclude:: ../../../scripts/tutorials/sim/planner/trajectory_pr_checks.py
   :language: python
   :start-at: SCENARIOS:
   :end-at: if __name__ == "__main__":
