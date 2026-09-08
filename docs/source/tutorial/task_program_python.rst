.. _tutorial_task_program_python:

Authoring a Task Program in Python
==================================

.. currentmodule:: embodichain.lab.task_program

This tutorial builds, validates, and compiles an Embodied Task Program entirely
in Python. It focuses on the provider-independent language: no simulator, Gym
environment, robot asset, or YAML file is required.

The result is a compiled semantic workflow, not robot commands. Continue with
:doc:`modular_env` and :doc:`task_program` when you are ready to bind the same
kind of program to a simulation environment. Use :doc:`atomic_actions` instead
when application code should directly plan or execute individual Atomic Skills.

What you will build
-------------------

The example declares one scene object and a typed Pick-and-Place workflow:

.. code-block:: text

   Repeat (3 times)
   └── Segment: move_cube
       ├── Pick cube
       ├── Place cube at the next cyclic target
       ├── Wait for cube to become stable
       └── Validate cube is near the selected target

Two target poses are selected cyclically, so the three repetitions use target
indices ``0``, ``1``, and ``0``. Compilation expands those occurrences into
stable segments and call indices without observing live scene state.

The code
--------

The complete runnable example is
``scripts/tutorials/task_program/build_and_compile.py``:

.. dropdown:: Code for build_and_compile.py
   :icon: code

   .. literalinclude:: ../../../scripts/tutorials/task_program/build_and_compile.py
      :language: python
      :linenos:

Run it from the repository root:

.. code-block:: bash

   python scripts/tutorials/task_program/build_and_compile.py

The output shows the deterministic expansion:

.. code-block:: text

   Program: python_pick_and_place
   Segments: 3
   [0] move_cube (repeat 1/3): Pick -> Place; drop_pose[0]
   [1] move_cube (repeat 2/3): Pick -> Place; drop_pose[1]
   [2] move_cube (repeat 3/3): Pick -> Place; drop_pose[0]

Declare provider-independent scene identities
---------------------------------------------

The compiler needs canonical identities and their semantic types, but it does
not need simulator objects or state providers. A
:class:`~embodichain.lab.task_program.semantics.SceneManifest` is the static
catalog for that boundary:

.. literalinclude:: ../../../scripts/tutorials/task_program/build_and_compile.py
   :language: python
   :start-at: def build_scene_manifest() -> SceneManifest:
   :end-before: def build_program() -> TaskProgramCfg:

``SceneObjectRef("cube")`` establishes that ``cube`` is an object rather than
an articulation, link, or affordance. The compiler uses this typed identity to
reject invalid or unknown references before a live environment is created.

Build the typed program tree
----------------------------

Python authoring uses the same declarative schema as JSON or YAML. Constructors
such as :class:`PickCfg`, :class:`PlaceCfg`, :class:`SequenceCfg`, and
:class:`RepeatCfg` validate their own fields, so discriminator strings such as
``kind: pick`` do not need to be written manually:

.. literalinclude:: ../../../scripts/tutorials/task_program/build_and_compile.py
   :language: python
   :start-at: def build_program() -> TaskProgramCfg:
   :end-before: def print_compiled_program(compiled: CompiledTaskProgram) -> None:

The three values in :class:`TaskProgramIntegrationCfg` are exact identifiers
for the robot profile, scene registry, and runtime preset that a trusted
deployment will later provide. Provider-independent compilation preserves
those selections but does not construct or contact their live providers.

The program remains executable-free even though it is authored in Python.
Do not put callbacks, simulator objects, planners, or controller commands in a
Task Program. Registered extensions also accept only declarative arguments;
their executable lowerers belong to the trusted integration.

Compile and inspect the program
-------------------------------

:class:`TaskProgramCompiler` resolves the scene references, expands the bounded
repeat, selects each cyclic target, and assigns stable segment and call indices:

.. literalinclude:: ../../../scripts/tutorials/task_program/build_and_compile.py
   :language: python
   :start-at: def print_compiled_program(compiled: CompiledTaskProgram) -> None:

The returned :class:`CompiledTaskProgram` is immutable and can be iterated more
than once with the same result. Segment post-policies and validators are also
resolved during compilation, but they run only after an environment adapter
binds the compiled program to live services.

Understand the boundary
-----------------------

This tutorial stops at the correct provider-independent boundary. Compilation
does not:

- observe object poses or robot state;
- select planners or generate controller commands;
- verify that the named robot profile and runtime preset are installed; or
- call ``env.step()``.

Those responsibilities belong to the trusted integration, Atomic Skills, and
the Gym execution bridge. Keeping them out of this example makes the Task
Program language independently testable and keeps task intent reusable across
compatible embodiments.

Diagnose an invalid reference
-----------------------------

Compiler failures include a stable error code and an exact source path. For
example, compiling the same program against an empty scene manifest fails
before any provider can be touched:

.. code-block:: python

   from embodichain.lab.task_program import (
       TaskProgramCompileError,
       TaskProgramCompiler,
       render_config_path,
   )
   from embodichain.lab.task_program.semantics import SceneManifest

   try:
       TaskProgramCompiler(SceneManifest()).compile(build_program())
   except TaskProgramCompileError as error:
       print(error.code, render_config_path(error.path))

Use the reported path to fix the source declaration. Do not catch a compile
error and continue into execution with a partially resolved program.

Next steps
----------

- :doc:`modular_env` — learn how ``EmbodiedEnv`` assembles a modular runtime.
- :doc:`task_program` — deploy and execute a Task Program through typed task,
  embodiment, scene, and execution-policy components.
- :doc:`/overview/task_program/index` — study the full language, semantic, and
  integration architecture.
- :doc:`atomic_actions` — directly construct and execute typed Atomic Skills.
