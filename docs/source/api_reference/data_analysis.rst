:orphan:

Data diversity analysis
=======================

The offline analysis package stores versioned episode metadata in SQLite,
computes distributions, and replays portable recordings through Viser.
Simulation is imported only by the existing-task collection worker. The Gradio
workbench is optional: install the ``analysis`` extra and run
``embodichain analyze-data serve --catalog /path/to/catalog.sqlite``.

Collection currently qualifies the existing Franka repeated-pick-place task.
Configured variation, physical observations and unknown measurements remain
distinct. Geometric lift/final-placement checks do not certify every intermediate
contact; the workbench does not yet perform model attribution or trajectory clustering.

embodichain.data_analysis
-------------------------

.. currentmodule:: embodichain.data_analysis

.. autosummary::

   DIMENSION_KEYS
   IDENTITY_FIELDS
   MEASUREMENT_SOURCES
   SCHEMA_VERSION
   STATUSES
   Catalog
   RecordValidationError
   ReplayViewer
   coverage
   distribution
   import_lerobot_metadata
   joint_distribution
   unknown_measurement
   validate_record

.. autodata:: DIMENSION_KEYS

.. autodata:: IDENTITY_FIELDS

.. autodata:: MEASUREMENT_SOURCES

.. autodata:: SCHEMA_VERSION

.. autodata:: STATUSES

.. autoclass:: Catalog
   :members:

.. autoclass:: RecordValidationError
   :members:

.. autoclass:: ReplayViewer
   :members:

.. autofunction:: coverage

.. autofunction:: distribution

.. autofunction:: import_lerobot_metadata

.. autofunction:: joint_distribution

.. autofunction:: unknown_measurement

.. autofunction:: validate_record

embodichain.data_analysis.catalog
---------------------------------

.. currentmodule:: embodichain.data_analysis.catalog

.. autosummary::

   Catalog

.. autoclass:: Catalog
   :members:

embodichain.data_analysis.cli
-----------------------------

.. currentmodule:: embodichain.data_analysis.cli

.. autosummary::

   main

.. autofunction:: main

embodichain.data_analysis.collection
------------------------------------

.. currentmodule:: embodichain.data_analysis.collection

.. autosummary::

   candidate_parameters
   StepObserver
   collect_preview
   collect_worker

.. autofunction:: candidate_parameters

.. autoclass:: StepObserver
   :members:

.. autofunction:: collect_preview

.. autofunction:: collect_worker

embodichain.data_analysis.importers
-----------------------------------

.. currentmodule:: embodichain.data_analysis.importers

.. autosummary::

   import_lerobot_metadata

.. autofunction:: import_lerobot_metadata

embodichain.data_analysis.recording
-----------------------------------

.. currentmodule:: embodichain.data_analysis.recording

.. autosummary::

   write_recording
   load_trajectory
   physical_metrics
   json_value

.. autofunction:: write_recording

.. autofunction:: load_trajectory

.. autofunction:: physical_metrics

.. autofunction:: json_value

embodichain.data_analysis.replay
--------------------------------

.. currentmodule:: embodichain.data_analysis.replay

.. autosummary::

   ReplayViewer

.. autoclass:: ReplayViewer
   :members:

embodichain.data_analysis.schema
--------------------------------

.. currentmodule:: embodichain.data_analysis.schema

.. autosummary::

   DIMENSION_KEYS
   IDENTITY_FIELDS
   MEASUREMENT_SOURCES
   REQUIRED_FIELDS
   SCHEMA_VERSION
   STATUSES
   RecordValidationError
   unknown_measurement
   validate_record

.. autodata:: DIMENSION_KEYS

.. autodata:: IDENTITY_FIELDS

.. autodata:: MEASUREMENT_SOURCES

.. autodata:: REQUIRED_FIELDS

.. autodata:: SCHEMA_VERSION

.. autodata:: STATUSES

.. autoclass:: RecordValidationError
   :members:

.. autofunction:: unknown_measurement

.. autofunction:: validate_record

embodichain.data_analysis.statistics
------------------------------------

.. currentmodule:: embodichain.data_analysis.statistics

.. autosummary::

   coverage
   distribution
   joint_distribution

.. autofunction:: coverage

.. autofunction:: distribution

.. autofunction:: joint_distribution

embodichain.data_analysis.ui
----------------------------

.. currentmodule:: embodichain.data_analysis.ui

.. autosummary::

   query_view
   export_slice
   compare_snapshots
   build_app

.. autofunction:: query_view

.. autofunction:: export_slice

.. autofunction:: compare_snapshots

.. autofunction:: build_app

Six-family definitions and slices
---------------------------------

The definition registry groups scalar fields into six feature families. Missing,
explicitly unknown and incompatible type/unit values are counted independently.
Joint cells retain exact episode membership; numeric conditions use canonical
identities, and exported slices retain the applied population and bin edges.

.. currentmodule:: embodichain.data_analysis.dimensions

.. autosummary::

   DIMENSIONS
   build_filters
   dimension_summary
   joint_cells
   select_cell
   match_dimension

.. autodata:: DIMENSIONS

.. autofunction:: build_filters

.. autofunction:: dimension_summary

.. autofunction:: joint_cells

.. autofunction:: select_cell

.. autofunction:: match_dimension

Phase-aligned trajectory analysis
---------------------------------

Phase keys identify a task-segment name and its occurrence. Curves share a
normalized display axis, while speed and duration use original timestamps.
Geometric distance compares world-coordinate paths resampled by arc length;
it is a descriptive distance, not an automatic geometry-family classifier.
Joint plots use recorded column indices because portable recordings currently
lack joint-name/unit metadata. Cross-robot or unequal-width comparisons fail
explicitly rather than silently aligning different joint vectors.

.. currentmodule:: embodichain.data_analysis.trajectory

.. autosummary::

   phase_options
   trajectory_comparison
   trajectory_plot

.. autofunction:: phase_options

.. autofunction:: trajectory_comparison

.. autofunction:: trajectory_plot
