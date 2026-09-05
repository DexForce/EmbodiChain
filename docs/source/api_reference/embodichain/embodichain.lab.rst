embodichain.lab
=====================

.. automodule:: embodichain.lab

Overview
--------

The ``lab`` package is EmbodiChain's robotics laboratory. It owns the
provider-independent Task Program language (including its Semantic Call
contracts), the simulation core (``sim``), the Gymnasium-compatible
environment framework (``gym``), real-device controllers (``devices``), and
browser visualization (``visualization``).

.. rubric:: Submodules

.. autosummary::

   devices
   task_program
   gym
   sim
   visualization

Browser Visualization
---------------------

.. toctree::
   :maxdepth: 1

   embodichain.lab.visualization

Task Program
------------

.. toctree::
   :maxdepth: 1

   embodichain.lab.task_program
   embodichain.lab.task_program.language
   embodichain.lab.task_program.semantics
   embodichain.lab.task_program.compiler
   embodichain.lab.task_program.runtime
   embodichain.lab.task_program.integrations
   embodichain.lab.task_program.integrations.simulation

Device Management
-----------------

Real-device controllers that mirror the simulation ``Robot`` API for deploying
policies on physical hardware.

.. automodule:: embodichain.lab.devices
   :members:
   :undoc-members:
   :show-inheritance:
