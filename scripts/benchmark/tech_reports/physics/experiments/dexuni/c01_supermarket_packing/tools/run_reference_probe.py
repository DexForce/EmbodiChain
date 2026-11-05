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
"""Record the unmodified supermarket demo for C01 physics diagnosis only.

Run: <native-python> run_reference_probe.py --out <diagnostic-directory>
No rendering, latency/throughput measurement, or changes to the imported demo.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np

__all__ = ["main"]


def main() -> None:
    """Run the original scene/controller and preserve physical-state evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = Path(
        "/home/oem/code/repos/dexsimforembodychain/dexsim/modules/dexsim/examples/python/physics/multiphys"
    )
    demo_path = source / "demo_dexuni_supermarket_packing.py"
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location("original_supermarket", demo_path)
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    import dexsim
    import newton
    import warp as wp

    output = args.out
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source"
    snapshot.mkdir()
    for name in (demo_path.name, "dexuni_piper_utils.py", "dexuni_asset_utils.py"):
        shutil.copy2(source / name, snapshot / name)
    shutil.copy2(__file__, snapshot / Path(__file__).name)
    config_args = argparse.Namespace(
        assets=Path("/home/oem/.cache/dexsim/resources/mjvbd_v2/supermarket"),
        device="cuda:0",
        substeps=8,
        iterations=24,
        disable_cuda_graph=False,
    )
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = dict(
        purpose="Original demo physical-state diagnosis; no performance measurements",
        status="running",
        executable=sys.executable,
        dexsim_version=dexsim.__version__,
        newton_version=newton.__version__,
        warp_version=wp.__version__,
        reference_path=str(demo_path),
        reference_sha256=sha(demo_path),
        steps=1800,
        dt=1 / 60,
        substeps=8,
        iterations=24,
        original_acceptance_errors=[],
        asset_sha256={
            str(p.relative_to(config_args.assets)): sha(p)
            for p in sorted(config_args.assets.rglob("*"))
            if p.is_file()
        },
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    ik = demo.PiperIK(config_args.assets)
    config = dexsim.WorldConfig()
    config.use_default_physics = False
    config.open_windows = False
    config.backend = dexsim.types.Backend.VULKAN
    config.renderer = dexsim.types.Renderer.OFFLINERT
    config.newton_cfg = demo.solver_config(config_args)
    world = dexsim.World(config)
    world.set_manual_update(True)
    scene = None
    states: dict[str, list[np.ndarray]] = {}
    try:
        scene, control = demo.build_scene(world, config_args, ik)
        world.enable_physics(True)
        checks = demo.PackingChecks(scene, config_args.assets)
        handles = {n: scene.get_particle_set(n) for n in ("bag", "bread", "pouch")}
        robot = scene.get_articulation("piper")
        layout = {
            n.name.rsplit("/", 1)[-1]: n.dof_start for n in robot.joint_dof_layout
        }
        joint_ids = [layout[n] for n in ik.names]
        manifest["joint_names"] = list(ik.names)
        for frame in range(1801):
            if frame:
                position, opening, _ = demo.target(frame / 60)
                target = ik._ik(position, opening)
                control.set_target(target)
                world.update(1 / 60)
            else:
                target = ik._ik(demo.TCP_HOME, demo.OPEN)
            actual_time = float(scene.get_simulation_time())
            assert abs(actual_time - frame / 60) < 1e-7
            values = {n: h.get_particle_positions().numpy() for n, h in handles.items()}
            values.update(
                time=np.asarray(actual_time),
                qpos=robot.get_current_qpos()[joint_ids],
                target_q=np.asarray(target),
            )
            assert all(np.isfinite(v).all() for v in values.values())
            for name, value in values.items():
                states.setdefault(name, []).append(np.array(value, copy=True))
            if frame and (frame % 15 == 0 or frame == 1800):
                try:
                    checks.measure(actual_time, final=frame == 1800)
                except AssertionError as error:
                    manifest["original_acceptance_errors"].append(
                        {"frame": frame, "time_s": actual_time, "error": str(error)}
                    )
            if frame % 120 == 0:
                print(f"reference frame={frame}/1800 t={actual_time:.2f}", flush=True)
        manifest.update(
            status="completed",
            actual_time_s=float(scene.get_simulation_time()),
            effective=dict(scene.get_solver_diagnostics()),
            graph_captured=scene.is_cuda_graph_captured(),
        )
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        if states:
            np.savez_compressed(
                output / "states.npz", **{k: np.stack(v) for k, v in states.items()}
            )
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if scene is not None:
            scene.close()


if __name__ == "__main__":
    main()
