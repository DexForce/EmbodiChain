EmbodiChain Documentation
=========================

EmbodiChain is a GPU-accelerated robotics simulation framework for embodied AI
research. It provides tools for building, generating, and processing simulation
assets and scenes; inspecting headless or remote simulations in a browser;
creating robot learning environments; generating expert demonstration data;
training policies with imitation learning and reinforcement learning; and
deploying models into the real world.

The framework is built on top of `DexSim <https://github.com/DexForce>`_, a high-performance physics and rendering engine, designed for Embodied AI research and production use.

Start Here
----------

* :doc:`Install EmbodiChain <quick_start/install>` and
  :ref:`create your first scene <first-run-scene>`.
* :ref:`Preview an official task <first-run-task>` or browse the
  :doc:`task catalog <resources/task/index>` for another deployment.
* :doc:`Find a guide <guides/index>` for building environments, preparing assets,
  or collecting and training on demonstration data.

Table of Contents
-----------------

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   Introduction <introduction>
   quick_start/install
   quick_start/first_run
   tutorial/index

.. toctree::
   :maxdepth: 1
   :caption: How-to Guides

   guides/tasks_and_environments
   guides/robots_and_assets
   guides/data_and_training
   guides/integrations

.. toctree::
   :maxdepth: 1
   :caption: Overview
   :glob:

   overview/architecture/index
   overview/sim/index
   overview/gym/index
   overview/task_program/index
   overview/learning/index

.. toctree::
   :maxdepth: 1
   :caption: Features
   :glob:

   features/online_data.md
   features/generative_sim/index*
   features/workspace_analyzer/index*
   features/toolkits/index*
   features/mcp/index

.. toctree::
   :maxdepth: 1
   :caption: Resources
   :glob:

   resources/assets.md
   resources/robot/index*
   resources/task/index*
   resources/models/index
   resources/roadmap.md
   resources/publications/README.md

.. toctree::
   :maxdepth: 2
   :caption: Reference
   :titlesonly:

   guides/cli
   Python API <api_reference/index>

.. toctree::
   :maxdepth: 1
   :caption: Contributing

   Contribution Guide <https://github.com/DexForce/EmbodiChain/blob/main/CONTRIBUTING.md>
   quick_start/docs.md
   guides/agent_skills
