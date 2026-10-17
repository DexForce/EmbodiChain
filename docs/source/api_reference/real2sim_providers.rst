:orphan:

Registered real2sim source providers
===================================

These APIs admit content-pinned, non-executable source bundles and bind a nominal
source to one expanded Task Program call. Sources provide data and complete plans;
the existing host retains command execution, effect checks and persistence.
Review states describe input prerequisites and never imply physical success.

Source bundles and gripper conventions
-------------------------------------

.. automodule:: embodichain.data_pipeline.real2sim.bundle
   :members: write_real2sim_bundle, load_real2sim_bundle

.. automodule:: embodichain.data_pipeline.real2sim.gripper
   :members:

.. autoclass:: embodichain.data_pipeline.real2sim.GripperPositionCalibration
   :no-index:

.. autofunction:: embodichain.data_pipeline.real2sim.gripper_release_duration
   :no-index:

``gripper_release_duration`` measures trailing release evidence from host-supplied
measured-open and bilateral-contact masks using elapsed timestamps. The host
retains calibration, object containment/stability and task acceptance decisions.

Registered nominal plans
-----------------------

.. automodule:: embodichain.lab.task_program.integrations.expert_sources
   :members: ExpertSourceProvider, ExpertSourceProviderFactory, ExpertSourceSelection, ExpertSourceRegistry

.. automodule:: embodichain.lab.task_program.integrations.real2sim
   :members: Real2SimExpertSourceProviderFactory, Real2SimExpertSourceProvider

The initial registered source supports B=1, sequential, fixed-qpos Pick replay.
PickUp owns semantic segment and effect assembly. Replay-only sources reject
replanning/retry policies and configured combined expansion. Unverified source
bundles remain usable for offline diagnostics but cannot enter formal admission.
Provider instances must support weak references. Runtime identity tracking does
not retain completed providers or their recording arrays; active callbacks retain
their own provider. Both sampled velocity references and per-command-interval
speeds must satisfy the live robot limits, including at rapid reversals.
