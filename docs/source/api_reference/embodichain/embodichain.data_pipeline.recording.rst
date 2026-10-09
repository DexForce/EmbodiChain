embodichain.data_pipeline.recording
===================================

Overview
--------

Recording provenance and commit evidence link training episodes, natural
segment fragments, and replay trajectories through stable identities. The
journal records progress around the LeRobot commit boundary; it does not make
Parquet, RGB/depth video, and JSONL one atomic transaction.

Use ``inspect_recording`` after stopping a writer to diagnose incomplete
commits. ``recover_recording(..., repair=True)`` can restore a missing sidecar
from its journal only when committed frame and depth artifacts verify. It
never replays actions or rewrites an unfinished SDK write.

.. currentmodule:: embodichain.data_pipeline.recording

.. autosummary::

   RecordingJournal
   inspect_recording
   recover_recording
   build_recording_provenance
   stable_config_hash

.. automodule:: embodichain.data_pipeline.recording
   :members:
   :imported-members:

Commit journal
--------------

.. automodule:: embodichain.data_pipeline.recording.journal
   :members:

Configuration provenance
------------------------

.. automodule:: embodichain.data_pipeline.recording.provenance
   :members:

Segment frame mapping
---------------------

.. automodule:: embodichain.data_pipeline.recording.segments
   :members:
