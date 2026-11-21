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
        num_envs=cfg["num_envs"],
        objects_per_env=1,
        object_count=cfg["num_envs"],
        protocol_version=3,
        asset_mode=cfg["asset_mode"],
        cache_mode=cfg["cache_mode"],
        purpose=cfg["purpose"],
        repeat=cfg["repeat"],
        rigid_enable_sleep=cfg.get("rigid_enable_sleep", True),
        entry_perf_ns=ENTRY_NS,
        entry_epoch=ENTRY_EPOCH,
        status="running",
        stage_times_s={},
    )
    (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")

    def mark(stage: str) -> None:
        meta["stage_times_s"][stage] = (time.perf_counter_ns() - ENTRY_NS) / 1e9
        print("L02_STAGE", stage, meta["stage_times_s"][stage], flush=True)
        (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")

    app = None
    simulation = None
    try:
        if cfg["platform"] == "isaaclab":
            from isaaclab.app import AppLauncher

            app = AppLauncher(
                headless=True, visualizer="none", enable_cameras=cfg["record"]
            ).app
        module = importlib.import_module(
            "scripts.benchmark.tech_reports.loading.platforms."
            + cfg["platform"]
            + ".l02_multi_loading"
        )
        simulation = module.Simulation(cfg, mark)
        import numpy as np

        state = simulation.step()
        assert state.shape == (cfg["num_envs"], 1, 13), state.shape
        assert np.isfinite(state).all(), "Non-finite first state"
        mark("ready")
        np.save(out / "first_state.npy", state)
        arrays, audit = simulation.audit()
        np.savez(out / "model.npz", **arrays)
        (out / "model-audit.json").write_text(
            json.dumps(audit, indent=2, default=str) + "\n"
        )
        assert audit["num_envs"] == cfg["num_envs"] == audit["instance_count"]
        assert audit["objects_per_env"] == 1
        assert len(set(audit["arena_names"])) == cfg["num_envs"]
        assert arrays["mass"].size == cfg["num_envs"]
        assert len(np.unique(arrays["env_origins"], axis=0)) == cfg["num_envs"]
        if cfg["platform"] == "embodichain":
            assert len(np.unique(arrays["body_world"])) == cfg["num_envs"]
            assert len(arrays["body_world"]) == cfg["num_envs"]
            assert (
                int(
                    np.sum(arrays["shape_type"] == (7 if cfg["shape"] == "cube" else 3))
                )
                == cfg["num_envs"]
            )
        else:
            assert len(audit["colliders"]) == cfg["num_envs"]
            assert {x["type"] for x in audit["colliders"]} == {
                "Cube" if cfg["shape"] == "cube" else "Sphere"
            }
        expected = np.asarray(cfg["positions"])
        meta["position_error_max_m"] = float(
            np.linalg.norm(state[..., :3] - expected, axis=-1).max()
        )
        meta["max_speed_mps"] = float(np.linalg.norm(state[..., 7:10], axis=-1).max())
        meta["mass_error_max_kg"] = float(
            np.max(np.abs(arrays["mass"].astype(float) - cfg["mass_kg"]))
        )
        meta["audited_instances"] = int(arrays["mass"].size)
        if cfg["purpose"] == "pilot" and cfg["num_envs"] > 1:
            probe = simulation.isolation_probe()
            np.savez(out / "isolation-probe.npz", **probe)
            meta["isolation_max_drift_m"] = float(
                np.max(
                    np.linalg.norm(
                        probe["states"][..., :3] - probe["after_write"][..., :3],
                        axis=-1,
                    )
                )
            )
            meta["isolation_max_speed_mps"] = float(
                np.max(np.linalg.norm(probe["states"][..., 7:10], axis=-1))
            )
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
            from PIL import Image

            attempts = []
            for attempt in range(100):
                rgb = simulation.render()
                chroma = rgb.astype("int16").max(-1) - rgb.astype("int16").min(-1)
                colored = int((chroma > 60).sum())
                attempts.append({"attempt": attempt, "colored_pixels": colored})
                if colored >= 2:
                    break
            else:
                raise RuntimeError("No scene-colored pixels in native camera")
            mark("frame")
            meta["first_frame_attempts"] = attempts
            Image.fromarray(rgb).save(out / "first_frame.png")
            audit.update(simulation.camera_audit())
            (out / "model-audit.json").write_text(
                json.dumps(audit, indent=2, default=str) + "\n"
            )
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
                        num_envs=cfg["num_envs"],
                        camera_instances=audit["camera_instances"],
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
