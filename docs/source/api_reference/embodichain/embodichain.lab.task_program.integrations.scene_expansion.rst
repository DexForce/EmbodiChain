embodichain.lab.task_program.integrations.scene_expansion
=========================================================

This integration evaluates one
:class:`~embodichain.lab.sim.scene_expansion.SceneVariant` through the existing
Task Program and Gym demonstration execution path. An injected host prepares
the physical scene; a separate validator supplies measured task evidence.
The API currently requires a single environment row and an explicit or
environment-configured Task Program.

.. currentmodule:: embodichain.lab.task_program.integrations.scene_expansion

.. autosummary::
   :nosignatures:

   ScenePreparationResult
   SceneVariantEpisodeResult
   execute_scene_variant

Preparation and execution boundary
----------------------------------

The preparation callback applies or restores the candidate, settles physics,
checks the actual initial state, saves its initial-state artifact, and refreshes
recording observations and affected bindings. It returns a
``ScenePreparationResult`` with the actual initial-state ID and nonempty
``ValidationResult`` checks. Rejecting that initial state prevents execution.
Physical preparation and workspace screening are supplied by the caller at
this layer; the fixed-candidate API does not instantiate a simulation host.

For valid initial states, ``execute_scene_variant`` creates a fresh bridge using
the canonical ``EmbodiedEnv.create_demo_segments(task_program=...)`` path and
passes the resulting segments through ``execute_demo_episode``. Handwritten
segment-planner overrides do not replace the selected Task Program. The normal
Gym executor owns ``env.step()`` and suspends automatic reset during execution.

Measured acceptance and persistence
-----------------------------------

``SceneVariantEpisodeResult`` preserves initial checks, program execution, and
measured task checks independently. Acceptance requires all three to pass.
Empty checks, unavailable evidence, a failed program, or failed measured goals
cannot accept the scene. A completed program with projected effects still
needs physical evidence from the measured validator.

The attempt's proposal, actual initial-state identity, and validation evidence
are published under ``scene_expansion`` in the existing episode metadata.
Preparation provenance is owned and recursively immutable; ``to_metadata()``
returns JSON-compatible values. Failed or interrupted attempts remain
ineligible for reset-time dataset, camera, and trajectory persistence, including
explicit commit-row requests and failed-episode recording.

The integration performs one attempt, propagates configuration and host errors,
and creates no retry scheduler or persistence session. The caller closes the
normal transaction with ``env.reset(options={"save_data": result.accepted})``.
An accepted execution is not confirmation that a dataset write succeeded.

Injecting a host and measured validator
---------------------------------------

The following helper accepts the fixed source from the scene proposal example.
Both callbacks are required application implementations, so the example does
not substitute a constant passing check for physical evidence:

.. code-block:: python

   from collections.abc import Callable

   from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
   from embodichain.lab.sim.motion.expansion import ValidationResult
   from embodichain.lab.sim.scene_expansion import (
       FixedSceneCandidateSource,
       SceneVariant,
   )
   from embodichain.lab.task_program.integrations.scene_expansion import (
       ScenePreparationResult,
       SceneVariantEpisodeResult,
       execute_scene_variant,
   )

   def collect_fixed_variants(
       env: EmbodiedEnv,
       source: FixedSceneCandidateSource,
       *,
       prepare_scene: Callable[[SceneVariant], ScenePreparationResult],
       measured_validator: Callable[[], ValidationResult],
   ) -> list[SceneVariantEpisodeResult]:
       results = []
       for variant in source:
           result = execute_scene_variant(
               env,
               variant,
               prepare_scene=prepare_scene,
               measured_validator=measured_validator,
               episode_index=variant.ordinal,
               attempt_id=variant.ordinal,
           )
           results.append(result)
           env.reset(options={"save_data": result.accepted})
       return results

The environment must already have its Task Program integration configured.
``prepare_scene`` begins each candidate from a clean recording boundary using
reset with ``save_data=False`` when needed. It captures state and reseeds
observations only after settling and refreshing bindings. The measured callback
reads terminal state and any required intermediate-event evidence; it returns
``ValidationResult`` rather than an arbitrary truthy value. Applications retain
their own bounded retry policy and exceptional-attempt cleanup.

Public interfaces
-----------------

.. autoclass:: ScenePreparationResult
   :members:

.. autoclass:: SceneVariantEpisodeResult
   :members:

.. autofunction:: execute_scene_variant
