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

"""One fresh-process L-02 measurement; entry-to-ready includes engine imports."""

from __future__ import annotations
import time

ENTRY_NS = time.perf_counter_ns()
ENTRY_EPOCH = time.time()
import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys
import traceback


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    out = Path(cfg["output"])
    meta = dict(
        experiment="L-02",
        platform=cfg["platform"],
        shape=cfg["shape"],
        object_count=cfg["object_count"],
        asset_mode=cfg["asset_mode"],
        cache_mode=cfg["cache_mode"],
        purpose=cfg["purpose"],
        repeat=cfg["repeat"],
        entry_perf_ns=ENTRY_NS,
        entry_epoch=ENTRY_EPOCH,
        status="running",
        stage_times_s={},
    )
    (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")

    def mark(stage: str) -> None:
        meta["stage_times_s"][stage] = (time.perf_counter_ns() - ENTRY_NS) / 1e9
        print("L02_STAGE", stage, meta["stage_times_s"][stage], flush=True)

    app = None
    simulation = None
    try:
        if cfg["platform"] == "isaaclab":
            from isaaclab.app import AppLauncher

            app = AppLauncher(headless=True, visualizer="none", enable_cameras=True).app
        module = importlib.import_module(
            "scripts.benchmark.tech_reports.loading.platforms."
            + cfg["platform"]
            + ".l02_loading"
        )
        simulation = module.Simulation(cfg, mark)
        import numpy as np

        state = simulation.step()
        assert state.shape == (cfg["object_count"], 13), state.shape
        assert np.isfinite(state).all(), "Non-finite first state"
        mark("ready")
        first_frames = []
        for attempt in range(100):
            rgb = simulation.render()
            # Benchmark assets have saturated colors. Background has none.
            chroma = rgb.astype("int16").max(-1) - rgb.astype("int16").min(-1)
            count = int((chroma > 60).sum())
            first_frames.append(
                dict(
                    attempt=attempt,
                    elapsed_s=(time.perf_counter_ns() - ENTRY_NS) / 1e9,
                    colored_pixels=count,
                )
            )
            if rgb.shape == (480, 640, 3) and count >= 2:
                break
        else:
            raise RuntimeError("No scene-colored pixels in 100 camera acquisitions")
        mark("frame")
        meta["first_frame_attempts"] = first_frames
        from PIL import Image

        Image.fromarray(rgb).save(out / "first_frame.png")
        np.save(out / "first_state.npy", state)
        arrays, audit = simulation.audit()
        np.savez(out / "model.npz", **arrays)
        (out / "model-audit.json").write_text(
            json.dumps(audit, indent=2, default=str) + "\n"
        )
        expected = np.asarray(cfg["positions"])
        meta["position_error_max_m"] = float(
            np.linalg.norm(state[:, :3] - expected, axis=1).max()
        )
        meta["max_speed_mps"] = float(np.linalg.norm(state[:, 7:10], axis=1).max())
        meta["mass_error_max_kg"] = float(
            np.max(np.abs(arrays["mass"].astype(float) - 0.1))
        )
        meta["mesh_triangle_counts"] = sorted(
            set(x["triangles"] for x in audit["meshes"])
        )
        assert (
            len(audit["meshes"]) == cfg["object_count"]
        ), "Visual object count mismatch"
        assert meta["mesh_triangle_counts"] == [
            cfg["mesh_triangles"][cfg["shape"]]
        ], meta["mesh_triangle_counts"]
        assert len(audit["materials"]) == cfg["object_count"], "Missing material"
        colors = np.asarray([m["color_linear"][:3] for m in audit["materials"]])
        expected_colors = np.asarray([o["color"] for o in cfg["objects"]])
        meta["material_color_error_max"] = float(
            np.max(np.abs(colors - expected_colors))
        )
        meta["unique_color_readback_count"] = len(np.unique(colors, axis=0))
        if cfg["platform"] == "embodichain":
            types = audit["native"]["shape_type"]
            assert (
                types.count(1) == 1
                and types.count(7 if cfg["shape"] == "cube" else 3)
                == cfg["object_count"]
            ), types
            meta["native_material_id_count"] = len(
                {m["id"] for m in audit["materials"]}
            )
        else:
            types = [x["type"] for x in audit["collisions"]]
            assert (
                types.count("Plane") == 1
                and types.count("Cube" if cfg["shape"] == "cube" else "Sphere")
                == cfg["object_count"]
            ), types
        meta["module_origins"] = {
            name: getattr(sys.modules.get(name), "__file__", None)
            for name in [
                "embodichain",
                "embodichain_robots",
                "embodichain_tasks",
                "dexsim",
                "isaaclab",
                "isaacsim",
                "newton",
                "warp",
                "torch",
            ]
        }
        meta["module_resolution"] = {}
        for name in [
            "embodichain",
            "embodichain_robots",
            "embodichain_tasks",
            "dexsim",
            "isaaclab",
        ]:
            spec = importlib.util.find_spec(name)
            meta["module_resolution"][name] = spec.origin if spec is not None else None
        meta["package_versions"] = {}
        for name in [
            "embodichain",
            "dexsim-engine",
            "isaaclab",
            "isaacsim",
            "newton",
            "warp-lang",
            "torch",
            "numpy",
        ]:
            try:
                meta["package_versions"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        if cfg.get("record"):
            frames = out / "native_camera"
            frames.mkdir()
            timestamps = []
            states = []
            for i in range(60):
                for _ in range(10):
                    s = simulation.step()
                frame = simulation.render()
                Image.fromarray(frame).save(frames / f"frame_{i:04d}.png")
                states.append(s)
                timestamps.append(0.005 + (i + 1) * 0.05)
            np.save(out / "recording_states.npy", np.asarray(states))
            (frames / "capture.json").write_text(
                json.dumps(
                    dict(
                        physical_times_s=timestamps,
                        fps=20,
                        speed=1,
                        camera=audit["camera_world_opengl"],
                        intrinsics=audit["intrinsics"],
                        scope="3 s actual native simulation after initial load; loading latency comes from raw timestamps, not video",
                    ),
                    indent=2,
                )
                + "\n"
            )
        meta["status"] = "completed"
    except BaseException as e:
        meta["status"] = "failed"
        meta["failure_reason"] = repr(e)
        traceback.print_exc()
        raise
    finally:
        meta["end_epoch"] = time.time()
        (out / "run.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
        print("L02_RESULT", meta["status"], flush=True)
        # Process owns the native world; shutdown errors stay visible in exit status.
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
