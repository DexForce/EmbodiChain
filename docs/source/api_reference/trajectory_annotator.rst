:orphan:

Trajectory image annotation
===========================

The annotation application consumes indexed RGB images through a common frame
source. Video and HDF5 adapters own their input resources; the application owns
one worker thread, a bounded JPEG preview cache and revision-checked JSON output.
Frame indices are zero-based and action intervals include both endpoints.

Frame input contract
--------------------

.. currentmodule:: embodichain.toolkits.trajectory_annotator.frames

.. autosummary::

   ImageFrame
   FrameSourceInfo
   FrameSource
   ArrayFrameSource

.. autoclass:: ImageFrame
   :members:

.. autoclass:: FrameSourceInfo
   :members:

.. autoclass:: FrameSource
   :members:

.. autoclass:: ArrayFrameSource
   :members:

File adapters
-------------

Video uses presentation timestamps and a full presentation-order index. HDF5
reads a selected camera dataset on demand; its display time is derived from an
explicit or recorded FPS. Neither adapter preloads the full RGB sequence.

.. currentmodule:: embodichain.toolkits.trajectory_annotator.hdf5

.. autosummary::

   Hdf5FrameSource

.. autoclass:: Hdf5FrameSource
   :members:

.. currentmodule:: embodichain.toolkits.trajectory_annotator.video

.. autosummary::

   VideoFrameSource

.. autoclass:: VideoFrameSource
   :members:

Application entry points
------------------------

Pass a factory returning a fresh source. Construction, preparation, reads and
cleanup run on the same worker thread. Caller-owned image arrays must remain
unchanged for the application's lifetime.

.. currentmodule:: embodichain.toolkits.trajectory_annotator.app

.. autosummary::

   create_annotation_app
   launch_annotation

.. autofunction:: create_annotation_app

.. autofunction:: launch_annotation

For example::

   from embodichain.toolkits.trajectory_annotator.app import launch_annotation
   from embodichain.toolkits.trajectory_annotator.hdf5 import Hdf5FrameSource

   launch_annotation(
       lambda: Hdf5FrameSource("episode.hdf5"),
       "episode.cam_high.segments.json",
   )
