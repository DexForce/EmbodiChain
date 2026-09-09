Agent Lab
=========

Agent Lab runs open-ended local coding experiments independently of the Task
Program interpreter. Simulation methods are optional; execution evidence and
task-success claims remain separate.

Final Delivery
--------------

.. autofunction:: embodichain.gen_sim.agent_lab.delivery.finalize_run

The delivery schema is ``agent-lab-delivery/v1``. The fixed ``final`` directory
link exposes a consistent result manifest, a Markdown report derived from it,
and a validated MP4 when available. Finalization does not execute the task or
promote agent claims to physical success. Previous publications and original
attempts are retained, and unchanged inputs reuse their publication.

Resource Accounting
-------------------

.. autofunction:: embodichain.gen_sim.agent_lab.usage.collect_usage

.. autofunction:: embodichain.gen_sim.agent_lab.usage.refresh_usage

.. autoclass:: embodichain.gen_sim.agent_lab.usage.UsageMeter
   :members:

``usage.json`` and ``result.json.resources`` use ``agent-lab-usage/v1``.
Host interval unions avoid double-counting tool waits and exclude gaps between
launches. Linked Codex usage counters are deduplicated by invocation and turn;
cached input and reasoning output are subsets. Missing or interrupted records
are marked unavailable or partial rather than estimated from text. These are
usage observations, not billing or account-quota measurements.

Execution
---------

.. autofunction:: embodichain.gen_sim.agent_lab.session.create_run

.. autofunction:: embodichain.gen_sim.agent_lab.session.execute_script

.. autofunction:: embodichain.gen_sim.agent_lab.session.solve

.. autofunction:: embodichain.gen_sim.agent_lab.worker.main

Optional Runtime
----------------

.. autoclass:: embodichain.gen_sim.agent_lab.runtime.LabCfg
   :members:

.. autoclass:: embodichain.gen_sim.agent_lab.runtime.Lab
   :members:

Recorded Evidence
-----------------

These inspection functions summarize actual state and decodable video, not
task-specific acceptance.

.. autofunction:: embodichain.gen_sim.agent_lab.report.inspect_attempt

.. autofunction:: embodichain.gen_sim.agent_lab.report.motion_summary

Task Inputs
-----------

.. autoclass:: embodichain.gen_sim.agent_lab.catalog.Task
   :members:

.. autofunction:: embodichain.gen_sim.agent_lab.catalog.read_catalog

.. autofunction:: embodichain.gen_sim.agent_lab.catalog.write_json
