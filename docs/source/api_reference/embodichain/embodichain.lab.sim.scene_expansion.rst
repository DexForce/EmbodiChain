embodichain.lab.sim.scene_expansion
===================================

Scene expansion describes changes to a base scene before a new task execution.
The initial API provides immutable existing-entity pose proposals and a bounded
source of authored candidates. Task Program and GenSim hosts can share these
contracts without introducing their execution or generation dependencies into
the proposal package.

The core owns no physical restoration, task semantics, simulator stepping,
episode acceptance, or dataset writes. Public imports use the normal
``embodichain.lab.sim`` initialization and require the simulation dependencies;
constructing these values creates no simulation world.

.. currentmodule:: embodichain.lab.sim.scene_expansion

.. autosummary::
   :nosignatures:

   SceneExpansionCfg
   ScenePoseChange
   SceneVariant
   FixedSceneCandidateSource

Frames and proposal identity
----------------------------

``ScenePoseChange.arena_pose`` is a proper homogeneous 4x4 transform from the
entity frame into its local Z-up arena. It excludes replication offsets and
does not use a robot chain-root frame. Nested numeric input is copied into
immutable tuples; nonfinite, reflected, nonorthonormal, or malformed transforms
are rejected.

``SceneVariant.variant_id`` includes the original scene identity, seed,
ordinal, proposed geometry, and provenance. Entity and provenance ordering is
canonicalized. A proposal ID is separate from the actual initial-state identity
captured after physical settling, and neither certifies task completion.
The seed reproduces a proposal schedule; it does not promise identical physics
across engines or versions.

Bounded fixed candidates
------------------------

``FixedSceneCandidateSource`` snapshots at most ``max_candidates`` entries from
its input iterable. It leaves later entries untouched, permits changes only to
declared movable entity UIDs, and returns the same immutable candidates on each
iteration. An empty pose-change tuple is an explicit nominal candidate; an
empty source generates no candidates.

.. code-block:: python

   from embodichain.lab.sim.scene_expansion import (
       FixedSceneCandidateSource,
       SceneExpansionCfg,
       ScenePoseChange,
   )

   source = FixedSceneCandidateSource(
       SceneExpansionCfg(seed=7, max_candidates=1, movable_entity_ids=("cube",)),
       parent_scene_id="pick-place-scene:revision-1",
       candidates=[
           (
               ScenePoseChange(
                   entity_uid="cube",
                   arena_pose=(
                       (1, 0, 0, 0.35),
                       (0, 1, 0, 0.10),
                       (0, 0, 1, 0.76),
                       (0, 0, 0, 1),
                   ),
               ),
           ),
       ],
       metadata=(("source", "authored-layout"),),
   )
   variant = source.variants[0]

This only constructs a proposal. Support, manipulation reachability, collision
checks, and task semantics remain explicit host responsibilities. To evaluate
one proposal through the shared Task Program executor, use
:mod:`embodichain.lab.task_program.integrations.scene_expansion`.

Public contracts
----------------

.. autoclass:: SceneExpansionCfg
   :members:
   :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: ScenePoseChange
   :members:

.. autoclass:: SceneVariant
   :members:

.. autoclass:: FixedSceneCandidateSource
   :members:

Implementation module exports
-----------------------------

.. currentmodule:: embodichain.lab.sim.scene_expansion.cfg

.. autosummary::

   SceneExpansionCfg

.. currentmodule:: embodichain.lab.sim.scene_expansion.contracts

.. autosummary::

   ScenePoseChange
   SceneVariant

.. currentmodule:: embodichain.lab.sim.scene_expansion.source

.. autosummary::

   FixedSceneCandidateSource
