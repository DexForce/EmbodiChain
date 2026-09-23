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

"""One isolated camera worker. Run via the camera-pilot benchmark command."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import resource
import sys
import time
import traceback

# The installed Isaac Lab wrapper runs this file outside this repository.
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


def main() -> None:
    """Capture actual images and persist raw timing plus provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("embodichain", "isaaclab"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.argv = sys.argv[:1]
    args.output.mkdir(parents=True, exist_ok=True)
    from scripts.benchmark.rendering.workload import (
        PilotCfg,
        config_hash,
        measure,
        scene_spec,
    )
    import numpy as np
    import torch
    from PIL import Image

    cfg = PilotCfg.load(args.config)
    result = {
        "schema_version": 1,
        "backend": args.backend,
        "status": "failed",
        "quality_status": "not_qualified",
        "config": asdict(cfg),
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
    result["software"].update(
        torch=torch.__version__,
        cuda_runtime=torch.version.cuda,
        dexsim_engine=package_version("dexsim-engine"),
        isaacsim=package_version("isaacsim"),
        isaaclab_distribution=package_version("isaaclab"),
    )
    adapter = None
    try:
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
        # Unscored probe: changing the view must change the delivered pixels.
        for _ in range(10):
            adapter.capture()
        original = adapter.capture()
        stable = adapter.capture()
        adapter.set_probe_offset(0.4)
        for _ in range(4):
            moved = adapter.capture()
        stable_delta = float(np.abs(original.astype(float) - stable).mean())
        moved_delta = float(np.abs(original.astype(float) - moved).mean())
        result["validation"] = {
            "probe": "camera_eye_x_plus_0.4m_then_restore",
            "stationary_mean_abs_delta": stable_delta,
            "moved_mean_abs_delta": moved_delta,
            "freshness_probe_passed": moved_delta > max(1.0, 4 * stable_delta),
        }
        Image.fromarray(original).save(args.output / "probe_original.png")
        Image.fromarray(moved).save(args.output / "probe_moved.png")
        if not result["validation"]["freshness_probe_passed"]:
            raise RuntimeError(
                "Camera pose-change probe did not demonstrate fresh image delivery"
            )
        adapter.set_probe_offset(0.0)
        result["gpu_before_measurement"] = gpu_snapshot()
        devices = result["gpu_before_measurement"].get("devices", [])
        # A single physical NVIDIA device is unambiguous in this pilot.
        result["hardware_id"] = devices[0]["uuid"] if len(devices) == 1 else None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        result["metrics"] = measure(cfg, adapter.capture)
        result["metrics"]["cpu_peak_rss_bytes"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
        result["metrics"][
            "torch_peak_allocated_bytes"
        ] = torch.cuda.max_memory_allocated()
        result["gpu_after_measurement"] = gpu_snapshot()
        sample = adapter.capture()
        Image.fromarray(sample).save(args.output / "sample.png")
        result["validation"]["sample_nonempty"] = bool(sample.std() > 1.0)
        if not result["validation"]["sample_nonempty"]:
            raise RuntimeError("Rendered sample is empty or essentially uniform")
        result["status"] = "completed"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        # Preserve evidence even if native teardown fails or exits abruptly.
        write_json(args.output / "result.json", result)
        if adapter is not None:
            adapter.close()
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
