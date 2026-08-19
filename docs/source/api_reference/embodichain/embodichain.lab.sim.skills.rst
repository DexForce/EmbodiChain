embodichain.lab.sim.skills
==========================

.. automodule:: embodichain.lab.sim.skills

   .. rubric:: Scene integration contracts

   .. autosummary::

      SceneRegistry
      RegistrySceneProvider
      SceneEntityRegistration
      SceneEntityRef
      SceneObjectRef
      SceneArticulationRef
      SceneLinkRef
      SceneAffordanceRef
      SceneEntityStateProvider
      SceneGeometryProvider
      SceneDynamics
      SceneCollisionRole
      SceneCollisionWorldMode

   .. rubric:: Robot skill profiles

   .. autosummary::

      RobotSkillProfile
      BoundRobotSkillProfile
      RobotResource
      ResourceEndpoint
      ResourceEndpointAdapter
      EndpointResolution
      ControlPartEndpoint
      ControlPartEndpointAdapter
      ResourceBinding
      ResolvedResourceEndpoint
      ResolvedRobotResource
      ResolvedSkillBinding
      ResourceClaim
      SkillPolicyPreset
      ProfileValidationError
      UnsupportedSkillError
      AmbiguousSkillBindingError

.. currentmodule:: embodichain.lab.sim.skills

Robot resources and profiles
----------------------------

.. autoclass:: RobotSkillProfile
   :members:

.. autoclass:: BoundRobotSkillProfile
   :members:

.. autoclass:: RobotResource
   :members:

.. autoclass:: ResourceEndpoint
   :members:

.. autoclass:: ResourceEndpointAdapter
   :members:

.. autoclass:: EndpointResolution
   :members:

.. autoclass:: ControlPartEndpoint
   :members:

.. autoclass:: ControlPartEndpointAdapter
   :members:

.. autoclass:: ResourceBinding
   :members:

.. autoclass:: ResolvedResourceEndpoint
   :members:

.. autoclass:: ResolvedRobotResource
   :members:

.. autoclass:: ResolvedSkillBinding
   :members:

.. autoclass:: ResourceClaim
   :members:

.. autoclass:: SkillPolicyPreset
   :members:

Profile errors
--------------

.. autoclass:: ProfileValidationError

.. autoclass:: UnsupportedSkillError

.. autoclass:: AmbiguousSkillBindingError

Registry and provider
---------------------

.. autoclass:: SceneRegistry
   :members:

.. autoclass:: RegistrySceneProvider
   :members:

Registration contracts
----------------------

.. autoclass:: SceneEntityRegistration
   :members:

.. autoclass:: SceneEntityStateProvider
   :members:

.. autoclass:: SceneGeometryProvider
   :members:

References and enums
--------------------

.. autoclass:: SceneEntityRef
   :members:

.. autoclass:: SceneObjectRef
   :members:

.. autoclass:: SceneArticulationRef
   :members:

.. autoclass:: SceneLinkRef
   :members:

.. autoclass:: SceneAffordanceRef
   :members:

.. autoclass:: SceneDynamics
   :members:

.. autoclass:: SceneCollisionRole
   :members:

.. autoclass:: SceneCollisionWorldMode
   :members:
