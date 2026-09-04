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

"""Application-level effective-drive calibration and qualification."""

from __future__ import annotations

from .asset_audit import audit_assets, audits_admit_calibration
from .evaluator import CandidateEvaluation, EvaluationError, run_candidate
from .metrics import (
    QualificationGate,
    QualificationResult,
    TrackingMetrics,
    compute_tracking_metrics,
    qualify,
)
from .overlay import build_drive_overlay, load_overlay, write_overlay
from .report import (
    build_calibration_report,
    calibration_report_to_markdown,
    write_calibration_reports,
)
from .schema import (
    CalibrationConfig,
    ControlSchedule,
    DriveParameterSpec,
    EvaluatorConfig,
    QualificationThresholds,
    load_calibration_config,
    resolve_control_schedule,
)
from .tuning import TuningResult, TuningTrial, tune_drive

__all__ = [
    "CalibrationConfig",
    "CandidateEvaluation",
    "ControlSchedule",
    "DriveParameterSpec",
    "EvaluationError",
    "EvaluatorConfig",
    "QualificationGate",
    "QualificationResult",
    "QualificationThresholds",
    "TrackingMetrics",
    "TuningResult",
    "TuningTrial",
    "audit_assets",
    "audits_admit_calibration",
    "build_drive_overlay",
    "build_calibration_report",
    "calibration_report_to_markdown",
    "compute_tracking_metrics",
    "load_overlay",
    "load_calibration_config",
    "qualify",
    "resolve_control_schedule",
    "run_candidate",
    "tune_drive",
    "write_overlay",
    "write_calibration_reports",
]
