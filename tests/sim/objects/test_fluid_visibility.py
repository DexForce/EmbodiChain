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

"""FastRT fluid primary visibility, independent of physical particle counts."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.subprocess_sim]


@pytest.mark.parametrize("clipping", [False, True])
def test_fluid_surface_changes_record_camera_pixels_in_every_environment(
    tmp_path, clipping
):
    from dexsim.engine.newton_physics import DexUniSolverCfg

    if "sph_cfg" not in getattr(DexUniSolverCfg, "__dataclass_fields__", {}):
        pytest.skip("This DexSim build does not compose SPH through DexUni.")
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            str(tmp_path),
            str(int(clipping)),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    results = json.loads((tmp_path / "visibility.json").read_text())
    assert results
    for row in results:
        assert row["visible_blue_pixels"] > 100
        for hidden, restored in zip(
            row["hidden_blue_pixels"], row["restored_blue_pixels"]
        ):
            assert row["visible_blue_pixels"] > 4 * hidden
            assert restored > 100 and restored > 4 * hidden


def _run_visibility_worker(output: Path, clipping: bool) -> None:
    import numpy as np
    import torch
    from PIL import Image
    from dexsim.engine.newton_physics.solver_runtime import ensure_solver_available
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import (
        FluidObjectCfg,
        FluidSpawnCfg,
        FluidRenderCfg,
        NewtonPhysicsCfg,
        RenderCfg,
    )
    from embodichain.lab.sim.material import VisualMaterialCfg
    from embodichain.lab.sim.sensors import CameraCfg

    solver = ensure_solver_available("sph")
    num_envs = (
        2
        if getattr(solver, "supports_world_isolation", False)
        and getattr(solver, "supports_world_reset", False)
        else 1
    )
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            physics_dt=1 / 1440,
            device="cuda",
            num_envs=num_envs,
            arena_space=1.0,
            render_cfg=RenderCfg(renderer="fast-rt", spp=8),
            physics_cfg=NewtonPhysicsCfg(
                num_substeps=1,
                collision_cfg=None,
                solver_cfg={
                    "solver_type": "dexuni",
                    "sph_cfg": {
                        "sph_method": "dfsph",
                        "volume_map_boundary_enabled": False,
                        "particle_support_radius": 0.00675,
                        "pressure_relaxation": 0.2,
                        "max_velocity": 2.0,
                    },
                },
            ),
        )
    )
    fluid = camera = None
    try:
        fluid = sim.add_fluid_object(
            FluidObjectCfg(
                uid="visible_water",
                particle_spacing=0.003,
                init_pos=(0, 0, 0.15),
                spawn=FluidSpawnCfg(kind="box", size=(0.06,) * 3),
                render=FluidRenderCfg(
                    mode="mesh_surface",
                    render_radius=0.0009,
                    clip_below_z=0.05 if clipping else None,
                    material=VisualMaterialCfg(
                        base_color=[0.05, 0.35, 0.9, 1],
                        roughness=0.1,
                    ),
                    mesh_surface={
                        "gpu_enabled": True,
                        "cell_size": 0.00162,
                        "max_res": 128,
                        "kernel_radius": 0.0045,
                        "iso_density": 0.57,
                        "density_smooth_iters": 2,
                        "min_component_tris": 32,
                    },
                ),
            )
        )
        camera = sim.add_sensor(
            CameraCfg(
                uid="record",
                width=320,
                height=240,
                intrinsics=(350, 350, 160, 120),
                visualization_role="record",
                extrinsics=CameraCfg.ExtrinsicsCfg(
                    eye=(0.18, -0.24, 0.28), target=(0, 0, 0.15), up=(0, 0, 1)
                ),
            )
        )
        sim.prepare()
        sim.update(step=2)

        def capture():
            sim.render_camera_group([camera.group_id])
            camera.update(fetch_only=True)
            return camera.get_data()["color"][..., :3].cpu().numpy().copy()

        visible = capture()
        before = fluid.data.particle_state_w
        hidden_state = before.clone()
        hidden_state[..., 0] += 10
        hidden_frames = []
        restored_frames = []
        # Several publications may occur before the next camera consumes one.
        for state in (before, hidden_state):
            fluid.write_particle_state_to_sim(state)
            sim.sync_render_state()
        hidden_frames.append(capture())
        fluid.write_particle_state_to_sim(before)
        restored_frames.append(capture())
        for _ in range(3):
            fluid.write_particle_state_to_sim(hidden_state)
            hidden_frames.append(capture())
            # Visibility captures do not advance physics.
            fluid.write_particle_state_to_sim(before)
            restored_frames.append(capture())
        stream = torch.cuda.Stream(device=fluid.device)
        stream.wait_stream(torch.cuda.current_stream(fluid.device))
        with torch.cuda.stream(stream):
            fluid.write_particle_state_to_sim(hidden_state)
            # Force the renderer's staging copy to wait for producer work.
            torch.cuda._sleep(20_000_000)
            hidden_frames.append(capture())
            fluid.write_particle_state_to_sim(before)
            restored_frames.append(capture())
        torch.testing.assert_close(fluid.data.particle_state_w, before, rtol=0, atol=0)

        def blue_pixels(frame):
            frame = frame.astype(np.int16)
            return int(
                (
                    (frame[..., 2] > frame[..., 0] + 40)
                    & (frame[..., 2] > frame[..., 1] + 20)
                ).sum()
            )

        results = []
        for env_id in range(num_envs):
            Image.fromarray(visible[env_id]).save(output / f"visible_env{env_id}.png")
            Image.fromarray(hidden_frames[-1][env_id]).save(
                output / f"hidden_env{env_id}.png"
            )
            results.append(
                {
                    "env_id": env_id,
                    "visible_blue_pixels": blue_pixels(visible[env_id]),
                    "hidden_blue_pixels": [
                        blue_pixels(frame[env_id]) for frame in hidden_frames
                    ],
                    "restored_blue_pixels": [
                        blue_pixels(frame[env_id]) for frame in restored_frames
                    ],
                }
            )
        (output / "visibility.json").write_text(json.dumps(results, indent=2) + "\n")
    finally:
        fluid = camera = None
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    _run_visibility_worker(Path(sys.argv[1]), bool(int(sys.argv[2])))
