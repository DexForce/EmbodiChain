embodichain.gen_sim.action_engine
==================================

Action Engine turns validated task semantics into coordinate-free action
graphs and reproducible execution bundles. This page documents the planning
and generation surface; live grounding and execution are added in the
dependent runtime layers.

Core protocol and capabilities
------------------------------

.. automodule:: embodichain.gen_sim.action_engine
   :members:

.. automodule:: embodichain.gen_sim.action_engine.protocol
   :members:

.. automodule:: embodichain.gen_sim.action_engine.capabilities
   :members:

.. automodule:: embodichain.gen_sim.action_engine.capabilities.atomic
   :members:

.. automodule:: embodichain.gen_sim.action_engine.capabilities.builtins
   :members:

.. automodule:: embodichain.gen_sim.action_engine.capabilities.held_hand_over
   :members:

.. automodule:: embodichain.gen_sim.action_engine.capabilities.registry
   :members:

Domain contracts
----------------

.. automodule:: embodichain.gen_sim.action_engine.domain
   :members:

.. automodule:: embodichain.gen_sim.action_engine.domain.motion
   :members:

.. automodule:: embodichain.gen_sim.action_engine.domain.programs
   :members:

.. automodule:: embodichain.gen_sim.action_engine.domain.task_contracts
   :members:

.. automodule:: embodichain.gen_sim.action_engine.domain.v2
   :members:

.. automodule:: embodichain.gen_sim.action_engine.domain.visual_contracts
   :members:

Compilation and planning
------------------------

.. automodule:: embodichain.gen_sim.action_engine.compiler
   :members:

.. automodule:: embodichain.gen_sim.action_engine.compiler.core
   :members:

.. automodule:: embodichain.gen_sim.action_engine.compiler.v2
   :members:

.. automodule:: embodichain.gen_sim.action_engine.planning
   :members:

.. automodule:: embodichain.gen_sim.action_engine.planning.dual
   :members:

.. automodule:: embodichain.gen_sim.action_engine.planning.linker
   :members:

.. automodule:: embodichain.gen_sim.action_engine.planning.online
   :members:

.. automodule:: embodichain.gen_sim.action_engine.planning.planner
   :members:

.. automodule:: embodichain.gen_sim.action_engine.planning.selection
   :members:

.. automodule:: embodichain.gen_sim.action_engine.planning.task_planner_prompt
   :members:

.. automodule:: embodichain.gen_sim.action_engine.planning.vision
   :members:

Task assembly
-------------

.. automodule:: embodichain.gen_sim.action_engine.tasks
   :members:

.. automodule:: embodichain.gen_sim.action_engine.tasks.assembly
   :members:

.. automodule:: embodichain.gen_sim.action_engine.tasks.grounding
   :members:

.. automodule:: embodichain.gen_sim.action_engine.tasks.interpretation
   :members:

.. automodule:: embodichain.gen_sim.action_engine.tasks.recipes
   :members:

.. automodule:: embodichain.gen_sim.action_engine.tasks.scene
   :members:

Bundle generation
-----------------

.. automodule:: embodichain.gen_sim.action_engine.config
   :members:

.. automodule:: embodichain.gen_sim.action_engine.config.runtime_policy
   :members:

.. automodule:: embodichain.gen_sim.action_engine.generation
   :members:

.. automodule:: embodichain.gen_sim.action_engine.generation.artifacts
   :members:

.. automodule:: embodichain.gen_sim.action_engine.generation.assets
   :members:

.. automodule:: embodichain.gen_sim.action_engine.generation.config_builder
   :members:

.. automodule:: embodichain.gen_sim.action_engine.generation.generator
   :members:

.. automodule:: embodichain.gen_sim.action_engine.generation.models
   :members:

.. automodule:: embodichain.gen_sim.action_engine.generation.source_scene
   :members:

.. automodule:: embodichain.gen_sim.action_engine.cli.generate_action_agent_config
   :members:

Supporting planning utilities
-----------------------------

.. automodule:: embodichain.gen_sim.action_engine.graph_visualization
   :members:

.. automodule:: embodichain.gen_sim.action_engine.gripper_profiles
   :members:

.. automodule:: embodichain.gen_sim.action_engine.orientation
   :members:

.. automodule:: embodichain.gen_sim.action_engine.solver_profiles
   :members:

.. automodule:: embodichain.gen_sim.action_engine.runtime
   :members:

.. automodule:: embodichain.gen_sim.action_engine.runtime.actions
   :members:

.. automodule:: embodichain.gen_sim.action_engine.runtime.atomic_compat
   :members:

.. automodule:: embodichain.gen_sim.action_engine.runtime.body_grasp
   :members:

.. automodule:: embodichain.gen_sim.action_engine.runtime.coordinated_safety
   :members:

.. automodule:: embodichain.gen_sim.action_engine.runtime.frames
   :members:

.. automodule:: embodichain.gen_sim.action_engine.runtime.geometry_axes
   :members:

.. automodule:: embodichain.gen_sim.action_engine.runtime.grounding
   :members:

.. automodule:: embodichain.gen_sim.action_engine.runtime.grasp_diagnostics
   :members:

.. automodule:: embodichain.gen_sim.action_engine.runtime.loader
   :members:

.. automodule:: embodichain.gen_sim.action_engine.runtime.models
   :members:

.. automodule:: embodichain.gen_sim.action_engine.runtime.motion_policy
   :members:

.. automodule:: embodichain.gen_sim.action_engine.runtime.predicates
   :members:

.. automodule:: embodichain.gen_sim.action_engine.runtime.robot_parts
   :members:

.. automodule:: embodichain.gen_sim.action_engine.runtime.solver_compat
   :members:

.. automodule:: embodichain.gen_sim.action_engine.runtime.state
   :members:

Execution and agent runtime
---------------------------

.. automodule:: embodichain.gen_sim.action_engine.agent
   :members:

.. automodule:: embodichain.gen_sim.action_engine.unbound
   :members:

.. automodule:: embodichain.gen_sim.action_engine.environment
   :members:

.. automodule:: embodichain.gen_sim.action_engine.environment.agent_env
   :members:

.. automodule:: embodichain.gen_sim.action_engine.runtime.dynamic
   :members:

.. automodule:: embodichain.gen_sim.action_engine.runtime.executor
   :members:

.. automodule:: embodichain.gen_sim.action_engine.runtime.grasp_debug
   :members:

.. automodule:: embodichain.gen_sim.action_engine.runtime.recording
   :members:

.. automodule:: embodichain.gen_sim.action_engine.runtime.recovery
   :members:

.. automodule:: embodichain.gen_sim.action_engine.runtime.reporting
   :members:

.. automodule:: embodichain.gen_sim.action_engine.cli.run_agent
   :members:

Evaluation
----------

.. automodule:: embodichain.gen_sim.action_engine.evaluation
   :members:

.. automodule:: embodichain.gen_sim.action_engine.evaluation.ab
   :members:

.. automodule:: embodichain.gen_sim.action_engine.evaluation.e1_e2_scene_action
   :members:

.. automodule:: embodichain.gen_sim.video_archive
   :members:
