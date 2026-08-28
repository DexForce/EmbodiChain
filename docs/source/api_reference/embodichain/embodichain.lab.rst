embodichain.lab
=====================

.. automodule:: embodichain.lab

Overview
--------

The ``lab`` package is EmbodiChain's robotics laboratory. It owns declarative
semantic skill contracts (``semantic_skills``), provider-independent Expert
Programs (``expert_program``), the simulation core (``sim``), the
Gymnasium-compatible environment framework (``gym``), real-device controllers
(``devices``), and browser visualization (``visualization``).

.. rubric:: Submodules

.. autosummary::

   devices
   expert_program
   gym
   semantic_skills
   sim
   visualization

Browser Visualization
---------------------

.. toctree::
   :maxdepth: 1

   embodichain.lab.visualization

Expert Programs
---------------

.. toctree::
   :maxdepth: 1

   embodichain.lab.expert_program

Semantic Skills
---------------

.. toctree::
   :maxdepth: 1

   embodichain.lab.semantic_skills

Device Management
-----------------

Real-device controllers that mirror the simulation ``Robot`` API for deploying
policies on physical hardware.

.. automodule:: embodichain.lab.devices
   :members:
   :undoc-members:
   :show-inheritance:
