Configured assembly components
==============================

Peg insertion, gear meshing and nut threading register the common
``EmbodiedEnv``. Its optional ``scene_preparation`` declaration receives an
owned environment configuration before World creation; ``resource_export``
receives the live environment, frozen configuration and checkpoint directory.
Explicit configuration takes precedence over registration defaults.

Preparation expands pinned geometry into standard entities and binds native
event, action, observation, reward and termination components. A startup event
initializes shared state. Success conditions advance state at most once per
new control transition; observations and rewards reuse its measurements.
Selected-row reset does not advance physics or another row's history.
Mating and grasp contact sensors sample physical substeps. Task acceptance and
timeout handling follow the common TerminationManager contract.

The seven arm policy inputs retain encoder-relative position semantics and
the configured warmup mask. The exclusive assembly action also owns encoder
gravity feedforward and the held gripper target; additional action terms are
rejected to avoid conflicting robot commands.

Preparation and saved resources
------------------------------

.. currentmodule:: embodichain_tasks.manipulation.assembly.preparation

.. autofunction:: prepare_assembly_scene

.. currentmodule:: embodichain_tasks.manipulation.assembly.resources

.. autofunction:: export_assembly_resources

Termination conditions
----------------------

.. currentmodule:: embodichain.lab.gym.envs.managers.assembly

.. autofunction:: assembly_success

.. autofunction:: assembly_failure
