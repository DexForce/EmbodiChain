# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Simulator-independent benchmark execution, measurement and artifacts."""

from __future__ import annotations

from .artifacts import (
    ArtifactStore,
    JsonlLedger,
    append_jsonl,
    create_experiment_directory,
)
from .contracts import (
    AggregateMetric,
    Budget,
    ExperimentDefinition,
    MetricDefinition,
    RawObservation,
    RunRecord,
)
from .measurement import (
    AttemptMeasurement,
    AttemptOutcome,
    Measurement,
    ResourceSample,
    StageEvent,
    StageMeasurement,
    measure_attempts,
    measure_loop,
    measure_stages,
    sample_resource,
)
from .execution import execute_worker, repeat_schedule, run_experiment
from .planning import (
    BudgetExceeded,
    BudgetLedger,
    MatrixCase,
    RunPlan,
    build_run_plan,
    expand_parameter_matrix,
)
from .provenance import (
    gpu_snapshot,
    host_snapshot,
    package_version,
    process_memory_snapshot,
    software_snapshot,
)
from .records import RunSpec, SCHEMA_VERSION, validate_run_result

__all__ = [
    "AggregateMetric",
    "ArtifactStore",
    "AttemptMeasurement",
    "AttemptOutcome",
    "Budget",
    "BudgetExceeded",
    "BudgetLedger",
    "ExperimentDefinition",
    "SCHEMA_VERSION",
    "JsonlLedger",
    "MatrixCase",
    "Measurement",
    "MetricDefinition",
    "RawObservation",
    "ResourceSample",
    "RunPlan",
    "RunRecord",
    "RunSpec",
    "StageEvent",
    "StageMeasurement",
    "append_jsonl",
    "build_run_plan",
    "create_experiment_directory",
    "expand_parameter_matrix",
    "execute_worker",
    "gpu_snapshot",
    "host_snapshot",
    "measure_attempts",
    "measure_loop",
    "measure_stages",
    "package_version",
    "process_memory_snapshot",
    "sample_resource",
    "software_snapshot",
    "repeat_schedule",
    "run_experiment",
    "validate_run_result",
]
