embodichain.data_pipeline.depth_video
=====================================

.. automodule:: embodichain.data_pipeline.depth_video

Overview
--------

Compressed depth-sidecar storage for LeRobot datasets on Python 3.10/3.11.
The package stores camera depth as ``gray12le``/HEVC sidecar videos alongside
a LeRobot dataset, without modifying the installed LeRobot package and without
requiring Python 3.12. Depth quantization math is vendored from lerobot v0.6.0
so that sidecar videos stay binary-compatible with the official reader once the
project upgrades to Python 3.12.

The main entry points are :class:`DepthVideoWriter` and
:class:`DepthSidecarManager` for writing, :class:`DepthVideoReader` for
reading, and :func:`load_depth_dataset` / :func:`load_depth_meta` for loading
depth data alongside a LeRobot dataset. Behavior is configured through
:class:`DepthVideoCfg`; codec selection goes through :func:`detect_depth_encoder`
and :func:`resolve_depth_vcodec`.

.. rubric:: Classes

.. autosummary::

   DepthVideoCfg
   DepthVideoWriter
   DepthSidecarManager
   DepthVideoReader
   DepthVideoLibrary
   DepthCodecError

.. rubric:: Functions

.. autosummary::

   detect_depth_encoder
   resolve_depth_vcodec
   quantize_depth
   dequantize_depth
   load_depth_dataset
   load_depth_meta

Configuration
-------------

.. automodule:: embodichain.data_pipeline.depth_video.cfg
   :members:
   :undoc-members:
   :show-inheritance:

Codec & Quantization
--------------------

.. automodule:: embodichain.data_pipeline.depth_video.codec
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: embodichain.data_pipeline.depth_video.depth_utils
   :members:
   :undoc-members:
   :show-inheritance:

Writer
------

.. automodule:: embodichain.data_pipeline.depth_video.writer
   :members:
   :undoc-members:
   :show-inheritance:

Reader
------

.. automodule:: embodichain.data_pipeline.depth_video.reader
   :members:
   :undoc-members:
   :show-inheritance:
