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

"""Execute a preflighted deployment through the ordinary Gym demo owner."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

__all__: list[str] = []


class _EpisodeRecording:
    """Record every camera and the native state at each executed control step.

    Frames come from the environment's own sensor observations, so the video
    shows the actual executed motion rather than a re-rendered replay.
    """

    def __init__(self, env: Any, root: Path) -> None:
        self.env = env
        self.root = root
        self.fps = 1.0 / float(env.step_dt)
        self.frames: dict[str, list[Any]] = {}
        self.states: list[dict[str, Any]] = []

    def observe(self, observation: Any) -> None:
        """Append the observation's camera frames and the native object state."""
        import numpy as np

        sensors = observation.get("sensor", {}) if observation is not None else {}
        for uid in list(sensors.keys()):
            data = sensors[uid]
            if "color" not in data.keys():
                continue
            frame = data["color"][0, ..., :3]
            frame = frame.detach().cpu().numpy() if hasattr(frame, "detach") else frame
            self.frames.setdefault(uid, []).append(np.asarray(frame, dtype=np.uint8))
        step = len(self.states)
        robot = observation.get("robot", {}) if observation is not None else {}
        state: dict[str, Any] = {
            "control_step": step,
            "control_time_s": step / self.fps,
            "qpos": (
                robot["qpos"][0].detach().cpu().tolist()
                if "qpos" in robot.keys()
                else None
            ),
            "objects": {},
        }
        for uid in self.env.sim.get_rigid_object_uid_list():
            pose = self.env.sim.get_rigid_object(uid).get_local_pose(to_matrix=True)
            state["objects"][uid] = pose[0, :3, 3].detach().cpu().tolist()
        self.states.append(state)

    def write(self, report: dict[str, Any] | None) -> dict[str, Any]:
        """Encode one H.264 video per camera and the state timeline."""
        import json
        from dexsim.utility import images_to_video

        self.root.mkdir(parents=True, exist_ok=True)
        cameras = {}
        for uid, frames in self.frames.items():
            images_to_video(frames, str(self.root), uid, fps=self.fps)
            height, width = frames[0].shape[:2]
            cameras[uid] = {"frames": len(frames), "width": width, "height": height}
        (self.root / "states.jsonl").write_text(
            "".join(json.dumps(state) + "\n" for state in self.states)
        )
        manifest = {
            "schema_version": "embodichain.gensim-native-recording/v1",
            "fps": self.fps,
            "frames": len(self.states),
            "cameras": cameras,
            "episode_completed": None if report is None else report.get("completed"),
        }
        (self.root / "recording.json").write_text(json.dumps(manifest, indent=2))
        return manifest


def _execute_deployment(
    path: Path,
    *,
    device: str | None,
    headless: bool,
    reset_observer: Callable[[Any], None] | None = None,
    step_observer: Callable[[Any, Any, Any], None] | None = None,
    recording: Path | None = None,
) -> dict[str, Any]:
    from embodichain.lab.gym.envs.demo import execute_demo_episode
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.gym.utils.registration import (
        discover_task_packages,
        execute_init_hooks,
        get_env_spec,
    )
    from embodichain.lab.sim import SimulationManager
    from embodichain.utils.utility import load_config

    discover_task_packages()
    execute_init_hooks()
    data = load_config(path)
    cfg = config_to_cfg(data, source_path=path)
    if device is not None:
        cfg.sim_cfg.device = device
    cfg.sim_cfg.headless = headless
    env = None
    recorder = None
    try:
        env = get_env_spec(data["id"]).make(cfg=cfg)
        reset_result = env.reset(options={"save_data": False})
        if recording is not None:
            recorder = _EpisodeRecording(env, recording)
            recorder.observe(reset_result[0])
        if reset_observer is not None:
            reset_observer(reset_result[0])
        kwargs = {"episode_index": 0}
        if step_observer is not None or recorder is not None:

            def observe(active: Any, action: Any, observation: Any) -> None:
                if recorder is not None:
                    recorder.observe(observation)
                if step_observer is not None:
                    step_observer(active, action, observation)

            kwargs["step_observer"] = observe
        result = None
        try:
            result = execute_demo_episode(env, **kwargs)
        finally:
            if recorder is not None and result is None and recorder.states:
                # An interrupted episode still keeps its recorded frames.
                recorder.write(None)
        report = result.to_metadata()
        if recorder is not None:
            report["recording"] = recorder.write(report)
        report.update(
            status="succeeded" if result.completed and result.all_success else "failed",
            all_success=result.all_success,
            effect_assurance="verified",
        )
        env.reset(options={"save_data": result.completed and result.all_success})
        return report
    finally:
        try:
            if env is not None:
                env.close(exit_process=False)
        finally:
            SimulationManager.flush_cleanup_queue()
