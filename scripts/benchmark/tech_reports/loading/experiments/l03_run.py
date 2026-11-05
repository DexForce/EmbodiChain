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

"""L-03 fresh-process loading, all-arena audit and separate native recording.

Run: python -m scripts.benchmark.tech_reports.loading.experiments.l03_run --config RUN_CONFIG
"""

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

__all__ = ["main"]


def main() -> None:
    """Include runtime imports, initialization, one step and full state readback."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    out = Path(cfg["output"])
    meta = {
        k: cfg[k]
        for k in ("platform", "robot", "num_envs", "purpose", "cache_mode", "repeat")
    }
    meta.update(
        experiment="L-03",
        entry_perf_ns=ENTRY_NS,
        entry_epoch=ENTRY_EPOCH,
        status="running",
        stage_times_s={},
        quaternion_order="xyzw",
        performance_eligibility_record="exit.json; decided by parent after resource and exit checks",
        rendering=cfg["record"],
        timing_scope="entry through first physics step, synchronization and all-env joint/root/link state CPU readback; input hash verification is performed by the parent before launch",
    )

    def mark(stage: str) -> None:
        meta["stage_times_s"][stage] = (time.perf_counter_ns() - ENTRY_NS) / 1e9
        (out / "run.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
        print("L03_STAGE", stage, meta["stage_times_s"][stage], flush=True)

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
            + ".l03_loading"
        )
        simulation = module.Simulation(cfg, mark)
        first = simulation.step()
        mark("ready")
        import numpy as np

        np.savez(
            out / "states.npz",
            **{"initial_" + k: v for k, v in simulation.initial.items()},
            **{"first_" + k: v for k, v in first.items()},
        )
        arrays, audit = simulation.audit()
        np.savez(out / "model.npz", **arrays)
        (out / "model-audit.json").write_text(
            json.dumps(audit, indent=2, default=str) + "\n"
        )
        n = cfg["num_envs"]
        assert audit["instance_count"] == audit["num_envs"] == n
        assert len(set(audit["arena_names"])) == n
        assert sorted(audit["joint_names"]) == cfg["expected_joint_names"]
        assert set(audit["link_names"]) == set(cfg["source_body_names"])
        assert arrays["mass"].shape == (n, len(audit["link_names"]))
        assert first["qpos"].shape == (n, len(audit["joint_names"]))
        assert first["link_pose"].shape == (n, len(audit["link_names"]), 7)
        assert len(np.unique(arrays["env_origins"], axis=0)) == n
        assert all(np.isfinite(a).all() for a in first.values())
        if cfg["platform"] == "embodichain":
            shape_counts = np.bincount(
                arrays["native_shape_world"][arrays["native_shape_world"] >= 0],
                minlength=n,
            )
            body_counts = np.bincount(
                arrays["native_body_world"][arrays["native_body_world"] >= 0],
                minlength=n,
            )
            assert np.array_equal(shape_counts, np.full(n, cfg["colliders_per_env"]))
            assert np.array_equal(body_counts, np.full(n, len(audit["link_names"])))
        else:
            assert audit["native_shapes_per_articulation"] == cfg["colliders_per_env"]
        meta.update(
            joint_names=audit["joint_names"],
            link_names=audit["link_names"],
            audited_instances=n,
            first_state_shapes={k: list(a.shape) for k, a in first.items()},
            root_initial_error_max_m=float(
                np.linalg.norm(
                    simulation.initial["root_pose"][:, :3]
                    - np.asarray(cfg["initial_position"]),
                    axis=-1,
                ).max()
            ),
            max_first_joint_change=float(
                np.abs(first["qpos"] - simulation.initial["qpos"]).max()
            ),
            max_first_link_translation_change_m=float(
                np.linalg.norm(
                    first["link_pose"][..., :3]
                    - simulation.initial["link_pose"][..., :3],
                    axis=-1,
                ).max()
            ),
            mass_per_env_kg=arrays["mass"].sum(axis=1).tolist(),
        )
        if cfg["record"]:
            from PIL import Image

            # Render several times without physics advancement to allow first-frame compilation.
            for _ in range(8):
                rgb = simulation.render()
            Image.fromarray(rgb).save(out / "first_frame.png")
            camera = simulation.camera_audit()
            audit.update(camera)
            (out / "model-audit.json").write_text(
                json.dumps(audit, indent=2, default=str) + "\n"
            )
            frames = out / "native_camera"
            frames.mkdir()
            timestamps, states = [], []
            for index in range(60):
                for _ in range(10):
                    state = simulation.step()
                Image.fromarray(simulation.render()).save(
                    frames / f"frame_{index:04d}.png"
                )
                timestamps.append(0.005 + (index + 1) * 0.05)
                states.append(state)
            np.savez(
                out / "recording-states.npz",
                **{k: np.stack([state[k] for state in states]) for k in states[0]},
            )
            (frames / "capture.json").write_text(
                json.dumps(
                    dict(
                        physical_times_s=timestamps,
                        fps=20,
                        speed=1,
                        num_envs=n,
                        camera=camera,
                        scope="3 s native simulation after loading; zero gravity and passive joints, not a dynamic control benchmark",
                    ),
                    indent=2,
                )
                + "\n"
            )
        meta["status"] = "completed"
    except BaseException:
        meta["status"] = "failed"
        meta["failure"] = traceback.format_exc()
        raise
    finally:
        meta["module_origins"] = {
            n: getattr(sys.modules.get(n), "__file__", None)
            for n in (
                "isaaclab",
                "isaacsim",
                "embodichain",
                "dexsim",
                "newton",
                "warp",
                "torch",
            )
        }
        meta["module_resolution"] = {}
        for name in (
            "embodichain",
            "embodichain_robots",
            "embodichain_tasks",
            "isaaclab",
            "dexsim",
        ):
            spec = importlib.util.find_spec(name)
            meta["module_resolution"][name] = spec.origin if spec else None
        meta["packages"] = {}
        for name in (
            "embodichain",
            "isaacsim",
            "isaaclab",
            "dexsim-engine",
            "newton",
            "warp-lang",
            "torch",
            "numpy",
        ):
            try:
                meta["packages"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        meta["end_epoch"] = time.time()
        (out / "run.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
        print("L03_RESULT", meta["status"], flush=True)
        if simulation is not None:
            simulation.close()
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
    if "embodichain.lab.sim" in sys.modules:
        import gc
        from embodichain.lab.sim import SimulationManager

        gc.collect()
        SimulationManager.flush_cleanup_queue()
        print("L03_CLEANUP_RETURNED", flush=True)
