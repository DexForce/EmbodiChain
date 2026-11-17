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

"""Real cable observations across Gym steps and selective episode resets."""

from __future__ import annotations

import faulthandler
import gc
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.subprocess_sim, pytest.mark.gpu]


def test_cable_recording_keeps_t_plus_one_states_and_partial_reset(
    tmp_path: Path,
) -> None:
    # A fresh process preserves NoRender without the shared test engine's
    # renderer initialization. No downloaded robot or scene assets are needed.
    urdf = tmp_path / "slider.urdf"
    urdf.write_text(
        """<robot name="slider">
<link name="base"><inertial><mass value="1"/><inertia ixx=".01" iyy=".01" izz=".01" ixy="0" ixz="0" iyz="0"/></inertial></link>
<link name="tool"><inertial><mass value=".1"/><inertia ixx=".001" iyy=".001" izz=".001" ixy="0" ixz="0" iyz="0"/></inertial></link>
<joint name="slide" type="prismatic"><parent link="base"/><child link="tool"/><axis xyz="1 0 0"/><limit lower="-.2" upper=".2" effort="100" velocity="1"/></joint>
</robot>""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(urdf)],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _run_recording(urdf: str) -> None:
    import torch

    from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
    from embodichain.lab.gym.envs.managers import ObservationCfg, SceneEntityCfg
    from embodichain.lab.gym.envs.managers.observations import get_entity_state
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import (
        CableObjectCfg,
        NewtonPhysicsCfg,
        RenderCfg,
        RobotCfg,
    )

    cfg = EmbodiedEnvCfg(
        num_envs=2,
        max_episode_steps=4,
        sim_steps_per_control=1,
        ignore_terminations=True,
        record_trajectory=True,
        trajectory_auto_save=False,
        trajectory_uids=["wire", "long_wire"],
        sim_cfg=SimulationManagerCfg(
            headless=True,
            device="cuda:0",
            arena_space=3.0,
            physics_dt=1 / 120,
            enable_entity_gizmo=False,
            startup_summary="off",
            render_cfg=RenderCfg(renderer="no-render"),
            physics_cfg=NewtonPhysicsCfg(
                device="cuda:0",
                num_substeps=2,
                solver_cfg={"solver_type": "dexuni"},
            ),
        ),
        robot=RobotCfg(
            uid="slider",
            fpath=urdf,
            init_pos=(-1.0, 0.0, 1.0),
            asset_physics_mode="overlay",
        ),
        cable_object=[
            CableObjectCfg(
                uid=uid,
                points=[[0.1 * index, 0, 0] for index in range(segments + 1)],
                radius=0.005,
                init_pos=(0.0, y, 1.0),
            )
            for uid, segments, y in (("wire", 2, 0.0), ("long_wire", 3, 0.5))
        ],
        observations={
            "wire": ObservationCfg(
                func=get_entity_state,
                name="wire_alias",
                mode="add",
                consumers=("policy", "recording"),
                params={
                    "entity_cfg": SceneEntityCfg(uid="wire"),
                    "quantity": "segment_state_w",
                },
            )
        },
    )
    env = None
    try:
        env = EmbodiedEnv(cfg)
        assert not env.sim.has_native_renderer
        obs, _ = env.reset(seed=73)
        manager = env.observation_manager
        assert manager is not None and manager.has_history
        cables = {uid: env.sim.get_cable_object(uid) for uid in cfg.trajectory_uids}
        fields = {uid: f"cables/{uid}/segment_state_w" for uid in cables}
        episodes = env.episode_context.episode_ids
        expected = {
            uid: [cable.data.segment_state_w.clone()] for uid, cable in cables.items()
        }
        torch.testing.assert_close(obs["wire_alias"], expected["wire"][0])
        assert manager.field_ids["wire_alias"] == manager.field_ids[fields["wire"]]

        action = torch.zeros((2, 1), device=env.device)
        for _ in range(2):
            env.step(action)
            for uid, cable in cables.items():
                expected[uid].append(cable.data.segment_state_w.clone())
        assert env._traj_steps.tolist() == [2, 2]
        window = manager.read_history_window(
            [0, 0],
            [2, 2],
            fields=list(fields.values()),
            episode_ids=episodes,
            device=env.device,
        )
        for uid, field in fields.items():
            states = torch.stack(expected[uid], dim=1)
            assert states.shape == (2, 3, cables[uid].data.num_segments, 13)
            assert bool((states[:, -1, :, 2] < states[:, 0, :, 2]).all())
            observed = torch.stack(window["fields"][field]["values"])
            torch.testing.assert_close(observed, states, rtol=0, atol=0)
            assert bool(torch.stack(window["fields"][field]["valid"]).all())
            descriptor = manager.schema[field]
            assert descriptor.coordinate_frame == "world"
            assert descriptor.shape == (cables[uid].data.num_segments, 13)
            assert descriptor.representation == "xyz_xyzw_com_lin_vel_ang_vel"
            recorded = env._traj_buffer["states", "cables", uid, "segment_state_w"]
            assert recorded.shape[1] == cfg.max_episode_steps + 1
            torch.testing.assert_close(recorded[:, :3], states, rtol=0, atol=0)

        # Reset row 0 through the public lifecycle. Row 1 must keep both its
        # physical state and all three historical boundaries, including S2.
        env.reset(options={"reset_ids": [0]})
        new_episodes = env.episode_context.episode_ids
        assert new_episodes == (episodes[0] + 1, episodes[1])
        assert env._traj_steps.tolist() == [0, 2]
        reset_state = manager.read_history(
            [0, 2], fields=list(fields.values()), episode_ids=new_episodes
        )
        for uid, field in fields.items():
            native = cables[uid].data.segment_state_w
            torch.testing.assert_close(native[0], expected[uid][0][0], rtol=0, atol=0)
            torch.testing.assert_close(native[1], expected[uid][-1][1], rtol=0, atol=0)
            torch.testing.assert_close(
                reset_state.values[field], native.cpu(), rtol=0, atol=0
            )
        with pytest.raises(ValueError, match="episode"):
            manager.read_history(
                [0], fields=[fields["wire"]], episode_ids=[episodes[0]], env_ids=[0]
            )

        env.step(action)
        assert env._traj_steps.tolist() == [1, 3]
        latest = manager.read_history(
            [1, 3], fields=list(fields.values()), episode_ids=new_episodes
        )
        retained = manager.read_history_window(
            [0],
            [2],
            fields=list(fields.values()),
            episode_ids=[episodes[1]],
            env_ids=[1],
            device=env.device,
        )
        for uid, field in fields.items():
            torch.testing.assert_close(
                latest.values[field],
                cables[uid].data.segment_state_w.cpu(),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                retained["fields"][field]["values"][0],
                torch.stack(expected[uid], dim=1)[1],
                rtol=0,
                atol=0,
            )
        # Commit the nonzero arena through public reset with auto-save enabled,
        # then replay that actual artifact in a fresh single-environment scene.
        from embodichain.lab.gym.envs.wrapper import ReplayWrapper
        from embodichain.lab.gym.utils.gym_utils import load_trajectory

        env.cfg.trajectory_auto_save = True
        env.cfg.trajectory_save_dir = str(Path(urdf).parent)
        source_origin = env.sim.arena_offsets[1].cpu().clone()
        env.reset(options={"reset_ids": [1]})
        paths = list(Path(urdf).parent.glob("traj_env1_*.pt"))
        assert len(paths) == 1
        saved = load_trajectory(str(paths[0]))
        assert saved["meta"]["env_ids"] == [1]
        assert saved["meta"]["arena_origins_w"] == [source_origin.tolist()]
        for uid, field in fields.items():
            assert saved["raw_observations"]["fields"][field]["values"].shape[1] == 4

        env.close(exit_process=False)
        env = None
        manager = None
        cables.clear()
        gc.collect()
        SimulationManager.flush_cleanup_queue()

        cfg.num_envs = 1
        cfg.sim_cfg.num_envs = 1
        cfg.record_trajectory = False
        env = EmbodiedEnv(cfg)
        replay = ReplayWrapper(env, saved)
        replay.reset()
        for _ in range(saved["meta"]["lengths"][0]):
            replay.step(None)
        for uid, field in fields.items():
            expected_final = latest.values[field][[1]].clone()
            expected_final[..., :3] -= source_origin
            observed = env.sim.get_cable_object(uid).data.segment_state_w.cpu()
            observed[..., :3] -= env.sim.arena_offsets.cpu()[:, None, :]
            torch.testing.assert_close(observed, expected_final, atol=2e-6, rtol=1e-5)
    finally:
        if env is not None:
            env.close(exit_process=False)
        env = None
        gc.collect()
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    faulthandler.enable()
    faulthandler.dump_traceback_later(120)
    _run_recording(sys.argv[1])
