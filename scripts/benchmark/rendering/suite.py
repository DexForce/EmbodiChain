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

"""Contracts and matrices for the pure-rendering R-series experiments.

The suite deliberately uses one procedural scene and explicit capture
boundaries. Each R-series item changes one workload axis so its results can be
compared with a matching Isaac Lab process without mixing scene, modality,
resolution or delivery semantics.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from scripts.benchmark.core.artifacts import stable_hash
from scripts.benchmark.core.contracts import MetricDefinition
from scripts.benchmark.core.measurement import measure_loop
from scripts.benchmark.core.planning import MatrixCase, expand_parameter_matrix

__all__ = [
    "CapturePacket",
    "EXPERIMENT_IDS",
    "RenderCaseCfg",
    "RenderExperiment",
    "case_config",
    "config_hash",
    "expand_experiment_cases",
    "experiment_catalog",
    "measure_capture",
    "metric_definitions",
    "scene_spec",
    "validate_capture_packet",
]

EXPERIMENT_IDS = ("R-03", "R-04", "R-05", "R-06", "R-09")


@dataclass
class RenderCaseCfg:
    """One resolved pure-rendering workload cell.

    ``num_envs * cameras_per_env`` is the number of camera exposures produced
    by one callback. Isaac Lab materializes this as a tiled camera batch;
    EmbodiChain materializes one camera group per logical camera over its
    arenas. ``delivery`` makes the observation boundary explicit.
    ``RenderCaseCfg`` is a resolved benchmark value record. It intentionally
    uses only standard-library dataclass machinery so listing and rebuilding a
    report never imports the simulator-oriented project configuration package.
    """

    experiment_id: str = "R-04"
    scene_version: str = "three_boxes_v1"
    width: int = 256
    height: int = 256
    num_envs: int = 1
    cameras_per_env: int = 1
    modalities: tuple[str, ...] = ("rgb",)
    delivery: str = "host_readback"
    temporal_mode: str = "static"
    warmup_frames: int = 10
    measured_frames: int = 50
    physics_steps_in_measurement: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.experiment_id, str) or not self.experiment_id.strip():
            raise ValueError("experiment_id must be a nonempty string")
        if self.scene_version != "three_boxes_v1":
            raise ValueError("only three_boxes_v1 is supported by the R-series pilot")
        for name in ("width", "height", "num_envs", "cameras_per_env"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.warmup_frames) is not int or self.warmup_frames < 0:
            raise ValueError("warmup_frames must be a nonnegative integer")
        if type(self.measured_frames) is not int or self.measured_frames < 1:
            raise ValueError("measured_frames must be a positive integer")
        if self.physics_steps_in_measurement != 0:
            raise ValueError("pure rendering measurements cannot step physics")
        modalities = tuple(self.modalities)
        if not modalities or any(
            modality not in {"rgb", "depth", "normals"} for modality in modalities
        ):
            raise ValueError("modalities must contain only rgb, depth or normals")
        if len(set(modalities)) != len(modalities):
            raise ValueError("modalities must not contain duplicates")
        if self.delivery not in {"render_only", "host_readback", "duplicate_readback"}:
            raise ValueError(
                "delivery must be render_only, host_readback or duplicate_readback"
            )
        if self.temporal_mode not in {"static", "moving"}:
            raise ValueError("temporal_mode must be static or moving")
        self.modalities = modalities

    @property
    def exposure_count(self) -> int:
        """Return logical camera exposures produced per observation."""
        return self.num_envs * self.cameras_per_env

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible workload cell."""
        return {
            "experiment_id": self.experiment_id,
            "scene_version": self.scene_version,
            "width": self.width,
            "height": self.height,
            "num_envs": self.num_envs,
            "cameras_per_env": self.cameras_per_env,
            "modalities": list(self.modalities),
            "delivery": self.delivery,
            "temporal_mode": self.temporal_mode,
            "warmup_frames": self.warmup_frames,
            "measured_frames": self.measured_frames,
            "physics_steps_in_measurement": self.physics_steps_in_measurement,
        }


@dataclass(frozen=True)
class RenderExperiment:
    """Public definition of one R-series question and parameter matrix."""

    experiment_id: str
    title: str
    question: str
    parameter_matrix: Mapping[str, tuple[object, ...]]

    def to_dict(self) -> dict[str, object]:
        """Return the frozen experiment definition used in reports."""
        return {
            "experiment_id": self.experiment_id,
            "title": self.title,
            "question": self.question,
            "parameter_matrix": {
                key: list(values) for key, values in self.parameter_matrix.items()
            },
        }


@dataclass(frozen=True)
class CapturePacket:
    """One rendered observation and its delivery accounting evidence."""

    arrays: Mapping[str, object]
    render_calls: int
    readback_calls: int
    gpu_sync_calls: int
    host_bytes: int
    exposure_count: int
    delivery: str

    def __post_init__(self) -> None:
        if type(self.render_calls) is not int or self.render_calls < 1:
            raise ValueError("render_calls must be positive")
        for name in (
            "readback_calls",
            "gpu_sync_calls",
            "host_bytes",
            "exposure_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not self.arrays:
            raise ValueError("a capture packet must contain at least one modality")
        if self.delivery not in {"render_only", "host_readback", "duplicate_readback"}:
            raise ValueError(f"unsupported delivery: {self.delivery!r}")

    def to_metadata(self) -> dict[str, object]:
        """Return accounting fields without serializing device tensors."""
        return {
            "render_calls": self.render_calls,
            "readback_calls": self.readback_calls,
            "gpu_sync_calls": self.gpu_sync_calls,
            "host_bytes": self.host_bytes,
            "exposure_count": self.exposure_count,
            "delivery": self.delivery,
        }


def scene_spec() -> dict[str, object]:
    """Return the shared procedural scene used by all pure-rendering cells."""
    from .workload import scene_spec as pilot_scene_spec

    return pilot_scene_spec()


def experiment_catalog() -> dict[str, RenderExperiment]:
    """Return the issue #679 pure-rendering experiment catalog."""
    return {
        "R-03": RenderExperiment(
            "R-03",
            "Resolution and render-quality scaling",
            "How does a usable RGB exposure scale with image resolution?",
            {"resolution": ("128x128", "256x256", "512x512")},
        ),
        "R-04": RenderExperiment(
            "R-04",
            "Batched camera rendering",
            "How does camera exposure throughput scale with batch size?",
            {"num_envs": (1, 4, 16)},
        ),
        "R-05": RenderExperiment(
            "R-05",
            "Observation modality cost",
            "What is the incremental cost of depth and normal observations?",
            {"modalities": ("rgb", "rgb_depth", "rgb_normals")},
        ),
        "R-06": RenderExperiment(
            "R-06",
            "Temporal camera rendering",
            "What changes when a camera produces a moving temporal sequence?",
            {"temporal_mode": ("static", "moving")},
        ),
        "R-09": RenderExperiment(
            "R-09",
            "Usable observation delivery",
            "What does rendering plus device synchronization and host delivery cost?",
            {"delivery": ("render_only", "host_readback", "duplicate_readback")},
        ),
    }


def expand_experiment_cases(experiment_id: str) -> tuple[MatrixCase, ...]:
    """Expand one catalog entry into deterministic case identities."""
    try:
        experiment = experiment_catalog()[experiment_id]
    except KeyError as exc:
        raise ValueError(f"unknown rendering experiment: {experiment_id}") from exc
    return expand_parameter_matrix(experiment.parameter_matrix)


def _resolution(value: object) -> tuple[int, int]:
    """Parse a catalog resolution value."""
    if not isinstance(value, str) or "x" not in value:
        raise ValueError(f"invalid resolution value: {value!r}")
    width_text, height_text = value.split("x", 1)
    width, height = int(width_text), int(height_text)
    if width < 1 or height < 1:
        raise ValueError("resolution dimensions must be positive")
    return width, height


def case_config(
    experiment_id: str,
    parameters: Mapping[str, object],
    *,
    warmup_frames: int = 10,
    measured_frames: int = 50,
) -> RenderCaseCfg:
    """Resolve one catalog case into a backend-neutral workload config."""
    if experiment_id not in experiment_catalog():
        raise ValueError(f"unknown rendering experiment: {experiment_id}")
    width, height = 256, 256
    if "resolution" in parameters:
        width, height = _resolution(parameters["resolution"])
    modalities_value = parameters.get("modalities", "rgb")
    modality_aliases = {
        "rgb": ("rgb",),
        "rgb_depth": ("rgb", "depth"),
        "rgb_normals": ("rgb", "normals"),
    }
    try:
        modalities = modality_aliases[modalities_value]
    except KeyError as exc:
        raise ValueError(f"unknown modality bundle: {modalities_value!r}") from exc
    return RenderCaseCfg(
        experiment_id=experiment_id,
        width=width,
        height=height,
        num_envs=int(parameters.get("num_envs", 1)),
        cameras_per_env=int(parameters.get("cameras_per_env", 1)),
        modalities=modalities,
        delivery=str(parameters.get("delivery", "host_readback")),
        temporal_mode=str(parameters.get("temporal_mode", "static")),
        warmup_frames=warmup_frames,
        measured_frames=measured_frames,
    )


def config_hash(cfg: RenderCaseCfg) -> str:
    """Hash the common scene and workload semantics, excluding the backend."""
    return stable_hash(
        {
            "config": cfg.to_dict(),
            "scene": scene_spec(),
            "boundary": "render_to_observation_delivery",
        }
    )


def _array_shape(value: object) -> tuple[int, ...] | None:
    """Return an array-like shape without importing a device array library."""
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(int(item) for item in shape)


def validate_capture_packet(packet: CapturePacket, cfg: RenderCaseCfg) -> None:
    """Validate modality shape, batch size and host/device delivery evidence."""
    if not isinstance(packet, CapturePacket):
        raise ValueError("capture must return a CapturePacket")
    if packet.delivery != cfg.delivery:
        raise ValueError(
            f"capture delivery {packet.delivery!r} does not match {cfg.delivery!r}"
        )
    expected_batch = cfg.exposure_count
    for modality in cfg.modalities:
        if modality not in packet.arrays:
            raise ValueError(f"capture is missing modality {modality!r}")
        shape = _array_shape(packet.arrays[modality])
        if modality == "rgb":
            expected = (expected_batch, cfg.height, cfg.width, 3)
        elif modality == "depth":
            expected = (expected_batch, cfg.height, cfg.width, 1)
        else:
            expected = (expected_batch, cfg.height, cfg.width, 3)
        if shape != expected:
            raise ValueError(f"capture batch/shape {shape} does not match {expected}")
    if packet.exposure_count != expected_batch:
        raise ValueError(
            f"capture exposure count {packet.exposure_count} does not match batch {expected_batch}"
        )
    if cfg.delivery == "render_only" and packet.readback_calls != 0:
        raise ValueError("render_only capture must not perform host readback")
    if cfg.delivery == "host_readback" and packet.readback_calls != 1:
        raise ValueError("host_readback capture must perform one host readback")
    if cfg.delivery == "duplicate_readback" and packet.readback_calls != 2:
        raise ValueError("duplicate_readback capture must perform two host readbacks")


def measure_capture(
    cfg: RenderCaseCfg,
    capture: Any,
    *,
    clock: Any,
) -> dict[str, object]:
    """Measure one suite cell and retain render/readback accounting."""
    timing = measure_loop(
        capture,
        warmup=cfg.warmup_frames,
        iterations=cfg.measured_frames,
        validate=lambda packet: validate_capture_packet(packet, cfg),
        clock=clock,
    )
    # The last packet is supplied by the adapter through this closure.
    last_packet = getattr(capture, "last_packet", None)
    if last_packet is None:
        raise ValueError("capture callback must expose last_packet after measurement")
    observations_per_s = timing.operations_per_s
    frames_per_s = observations_per_s * cfg.exposure_count
    return {
        "boundary": "render_to_observation_delivery",
        "completion": (
            "gpu_synchronized_device"
            if cfg.delivery == "render_only"
            else "blocking_host_readback"
        ),
        "num_envs": cfg.num_envs,
        "cameras_per_env": cfg.cameras_per_env,
        "exposure_count": cfg.exposure_count,
        "warmup_frames": cfg.warmup_frames,
        "measured_frames": cfg.measured_frames,
        "window_s": timing.window_s,
        "observations_per_s": observations_per_s,
        "camera_frames_per_s": frames_per_s,
        "capture_latency_s": list(timing.latencies_s),
        "latency_p50_s": timing.p50_s,
        "latency_p95_s": timing.p95_s,
        "host_readback_bytes_per_s": last_packet.host_bytes * observations_per_s,
        "host_bytes_per_observation": last_packet.host_bytes,
        "render_calls_per_observation": last_packet.render_calls,
        "readback_calls_per_observation": last_packet.readback_calls,
        "gpu_sync_calls_per_observation": last_packet.gpu_sync_calls,
    }


def metric_definitions() -> tuple[MetricDefinition, ...]:
    """Return stable metric definitions shared by all R-series reports."""
    return (
        MetricDefinition("observations_per_s", "1/s", "observation", "1.0"),
        MetricDefinition("camera_frames_per_s", "1/s", "camera_exposure", "1.0"),
        MetricDefinition("latency_p50_s", "s", "observation", "1.0"),
        MetricDefinition("latency_p95_s", "s", "observation", "1.0"),
        MetricDefinition("host_readback_bytes_per_s", "byte/s", "host_transfer", "1.0"),
        MetricDefinition("host_bytes_per_observation", "byte", "observation", "1.0"),
        MetricDefinition("render_calls_per_observation", "count", "observation", "1.0"),
        MetricDefinition(
            "readback_calls_per_observation", "count", "observation", "1.0"
        ),
        MetricDefinition(
            "gpu_sync_calls_per_observation", "count", "observation", "1.0"
        ),
    )
