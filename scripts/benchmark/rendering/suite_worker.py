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

"""Isolated worker for one R-series rendering workload cell."""

from __future__ import annotations

import argparse
from pathlib import Path
import resource
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark.core.artifacts import write_json
from scripts.benchmark.core.provenance import (
    gpu_snapshot,
    git_revision,
    package_version,
    software_snapshot,
)

__all__ = ["main"]


class _CaptureCall:
    """Expose the last packet to the metric reducer after measure_loop."""

    def __init__(self, adapter: object, delivery: str) -> None:
        self.adapter = adapter
        self.delivery = delivery
        self.last_packet = None

    def __call__(self):
        self.last_packet = self.adapter.capture_packet(self.delivery)
        return self.last_packet


def _sample_array(packet: object) -> object:
    """Select the first batch item for validation and evidence."""
    arrays = packet.arrays
    if "rgb" in arrays:
        return arrays["rgb"][0]
    if "depth" in arrays:
        return arrays["depth"][0]
    return arrays["normals"][0]


def _mean_abs_delta(first: object, second: object) -> float:
    """Calculate a finite pixel-space change without requiring a fixed dtype."""
    import numpy as np

    lhs = np.asarray(first, dtype=np.float32)
    rhs = np.asarray(second, dtype=np.float32)
    return float(np.abs(lhs - rhs).mean())


def _save_sample(output: Path, packet: object) -> bool:
    """Save RGB PNG or a modality array and return a nonempty check."""
    import numpy as np

    arrays = packet.arrays
    if "rgb" in arrays:
        from PIL import Image

        sample = arrays["rgb"][0]
        Image.fromarray(np.asarray(sample, dtype=np.uint8)).save(output / "sample.png")
        return bool(np.asarray(sample).std() > 1.0)
    for name, value in arrays.items():
        np.save(output / f"sample_{name}.npy", np.asarray(value[0]))
    sample = np.asarray(_sample_array(packet), dtype=np.float32)
    return bool(np.isfinite(sample).any() and float(np.nanstd(sample)) > 0.0)


def main() -> None:
    """Run one suite cell and write a complete result record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("embodichain", "isaaclab"), required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from scripts.benchmark.rendering.suite import (
        RenderCaseCfg,
        config_hash,
        measure_capture,
        scene_spec,
    )

    config_data = __import__("json").loads(args.config.read_text(encoding="utf-8"))
    try:
        cfg = RenderCaseCfg(**config_data["cases"][args.case_id])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid suite case {args.case_id!r}") from exc
    result = {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "case_id": args.case_id,
        "backend": args.backend,
        "status": "failed",
        "quality_status": "not_qualified",
        "config": cfg.to_dict(),
        "config_sha256": config_hash(cfg),
        "scene": scene_spec(),
        "software": software_snapshot(
            ROOT,
            [
                ROOT / "scripts/benchmark/__main__.py",
                *[
                    path
                    for name in ("core", "reporting", "rendering")
                    for path in (ROOT / "scripts/benchmark" / name).rglob("*.py")
                ],
            ],
        ),
    }
    adapter = None
    try:
        import torch

        result["software"].update(
            torch=torch.__version__,
            cuda_runtime=torch.version.cuda,
            dexsim_engine=package_version("dexsim-engine"),
            isaacsim=package_version("isaacsim"),
            isaaclab_distribution=package_version("isaaclab"),
        )
        started = time.perf_counter()
        if args.backend == "embodichain":
            from scripts.benchmark.rendering.backends.embodichain import (
                EmbodiChainCamera,
            )

            adapter = EmbodiChainCamera(cfg)
        else:
            from scripts.benchmark.rendering.backends.isaaclab import IsaacLabCamera

            adapter = IsaacLabCamera(cfg)
            import isaaclab

            lab_root = Path(isaaclab.__file__).resolve().parents[3]
            result["software"]["isaaclab_commit"] = git_revision(lab_root)
            result["software"]["isaaclab_root"] = str(lab_root)
        result["setup_s"] = time.perf_counter() - started
        result["renderer"] = adapter.metadata

        for _ in range(5):
            adapter.capture_host()
        original = adapter.capture_host()
        stable = adapter.capture_host()
        adapter.set_probe_offset(0.4)
        moved = adapter.capture_host()
        stable_delta = _mean_abs_delta(_sample_array(original), _sample_array(stable))
        moved_delta = _mean_abs_delta(_sample_array(original), _sample_array(moved))
        result["validation"] = {
            "probe": "camera_eye_x_plus_0.4m_then_restore",
            "stationary_mean_abs_delta": stable_delta,
            "moved_mean_abs_delta": moved_delta,
            "freshness_probe_passed": moved_delta > max(1.0, 2 * stable_delta),
        }
        if not result["validation"]["freshness_probe_passed"]:
            raise RuntimeError(
                "camera pose-change probe did not produce a fresh observation"
            )
        adapter.set_probe_offset(0.0)
        result["gpu_before_measurement"] = gpu_snapshot()
        devices = result["gpu_before_measurement"].get("devices", [])
        result["hardware_id"] = devices[0]["uuid"] if len(devices) == 1 else None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        capture = _CaptureCall(adapter, cfg.delivery)
        result["metrics"] = measure_capture(
            cfg,
            capture,
            clock=time.perf_counter,
        )
        result["metrics"]["cpu_peak_rss_bytes"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
        result["metrics"][
            "torch_peak_allocated_bytes"
        ] = torch.cuda.max_memory_allocated()
        result["gpu_after_measurement"] = gpu_snapshot()
        sample = adapter.capture_host()
        result["validation"]["sample_nonempty"] = _save_sample(args.output, sample)
        if not result["validation"]["sample_nonempty"]:
            raise RuntimeError("rendered sample is empty or has no finite variation")
        result["status"] = "completed"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        write_json(args.output / "result.json", result)
        if adapter is not None:
            adapter.close()
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
