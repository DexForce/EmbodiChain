embodichain.lab.sim.motion.motion_generator
===========================================

.. automodule:: embodichain.lab.sim.motion.motion_generator

Motion generation coordinates planner backends, IK interpolation, collision-world
options and normalized trajectory results. Import its configuration and strategy
options from this module; backend contracts remain under ``motion.planners``.

.. currentmodule:: embodichain.lab.sim.motion.motion_generator

.. autosummary::
   :nosignatures:

   MotionGenerator
   MotionGenCfg
   MotionGenOptions

Motion Generator
----------------

.. autoclass:: MotionGenCfg
    :members:
    :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: MotionGenerator
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: MotionGenOptions
    :members:
    :exclude-members: __init__, copy, replace, to_dict, validate
