embodichain.lab
=====================

.. automodule:: embodichain.lab

Overview
--------

The ``lab`` package is EmbodiChain's simulation laboratory. It bundles the
simulation core (``sim``), the Gymnasium-compatible environment framework
(``gym``), real-device controllers (``devices``), and the browser
visualization stack (``visualization``). Most user code interacts with ``lab``
through the environment classes and the simulation manager.

.. rubric:: Submodules

.. autosummary::

   devices
   gym
   sim
   visualization

Browser Visualization
---------------------

.. toctree::
   :maxdepth: 1

   embodichain.lab.visualization

Device Management
-----------------

Real-device controllers that mirror the simulation ``Robot`` API for deploying
policies on physical hardware.

.. automodule:: embodichain.lab.devices
   :members:
   :undoc-members:
   :show-inheritance:
