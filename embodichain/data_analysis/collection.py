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

"""Bounded collection of the existing Franka repeated-pick-place deployment.

Simulator imports are restricted to the worker. Each attempt runs in a fresh
process, isolating native renderer teardown and preserving failed attempts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .catalog import Catalog
from .recording import json_value, physical_metrics, write_recording
from .schema import DIMENSION_KEYS, unknown_measurement

__all__ = ["candidate_parameters", "StepObserver", "collect_preview", "collect_worker"]


def candidate_parameters(index: int, seed: int) -> dict[str, Any]:
    """Return a deterministic preview design, not an inferred data distribution."""
    if index < 0:
        raise ValueError("Candidate index must be nonnegative.")
    rng = np.random.default_rng(seed + index)
    return {
        "size": [0.05, 0.045][index % 2],
        "x": float(-0.42 + rng.uniform(-0.018, 0.018)),
        "y": float(-0.08 + rng.uniform(-0.02, 0.02)),
        "material": ["matte_blue", "glossy_orange"][(index // 2) % 2],
        "light": [3.0, 7.0][(index // 4) % 2],
    }


class StepObserver:
    """Delegate an environment step and observe its completed physical state."""

    def __init__(self, env: Any, capture: Callable[[], None]) -> None:
        self.env, self.capture = env, capture

    @property
    def unwrapped(self) -> Any:
        """Underlying environment used by the production demo executor."""
        return self.env.unwrapped

    def step(self, action: Any) -> Any:
        """Return the original step result after capturing its state."""
        result = self.env.step(action)
        self.capture()
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


def _measurement(
    value: Any, source: str, unit: str = "", scope: str = "episode"
) -> dict[str, Any]:
    return dict(
        value=value,
        source=source,
        unit=unit,
        frame="world" if unit in ("m", "rad") else "not_applicable",
        scope=scope,
    )


def collect_preview(
    directory: str | Path, *, count: int = 12, seed: int = 17, timeout: float = 180.0
) -> dict[str, Any]:
    """Collect bounded real-task attempts and preserve failure logs in the catalog."""
    if count < 1 or count > 1000:
        raise ValueError("count must be between 1 and 1000.")
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_id = "pick-place-" + uuid.uuid4().hex[:10]
    with Catalog(root / "catalog.sqlite") as catalog:
        for index in range(count):
            episode_id = f"{run_id}-{index:03d}"
            record = dict(
                schema_version=1,
                episode_id=episode_id,
                run_id=run_id,
                candidate_id=episode_id,
                attempt_id=0,
                task_id="TaskProgramRepeatedPickPlace-Franka-v1",
                robot_id="FrankaPanda",
                status="proposed",
                reason="",
                seed=seed + index,
                parents=[],
                dimensions={
                    k: unknown_measurement("not yet measured") for k in DIMENSION_KEYS
                },
                artifacts={},
                segments=[],
                metrics={},
                provenance={
                    "collector": "repeated_pick_place_preview_v1",
                    "domain": "simulation",
                    "desired": candidate_parameters(index, seed),
                },
            )
            catalog.upsert(record)
            request = root / f"{episode_id}.request.json"
            request.write_text(json.dumps(record), encoding="utf-8")
            log = root / f"{episode_id}.log"
            try:
                with log.open("w") as stream:
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "embodichain.data_analysis.cli",
                            "_worker",
                            str(request),
                        ],
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        timeout=timeout,
                    )
                current = catalog.get(episode_id)
                if current["status"] in ("proposed", "pending_write"):
                    final_status = (
                        "partial_commit"
                        if current["status"] == "pending_write"
                        else "rollout_failed"
                    )
                    current.update(
                        status=final_status,
                        reason=f"Worker exited {result.returncode} before finalizing; see log",
                        artifacts=current["artifacts"] | {"log": str(log)},
                    )
                    catalog.upsert(current)
            except subprocess.TimeoutExpired:
                current = catalog.get(episode_id)
                if current["status"] not in (
                    "committed",
                    "partial_commit",
                    "rejected",
                    "planning_failed",
                    "rollout_failed",
                ):
                    final_status = (
                        "partial_commit"
                        if current["status"] == "pending_write"
                        else "rollout_failed"
                    )
                    current.update(
                        status=final_status,
                        reason=f"Worker exceeded {timeout}s",
                        artifacts=current["artifacts"] | {"log": str(log)},
                    )
                    catalog.upsert(current)
            print(
                json.dumps(
                    {
                        "episode_id": episode_id,
                        "status": catalog.get(episode_id)["status"],
                    }
                ),
                flush=True,
            )
        result = catalog.summary()
        catalog.snapshot(run_id)
        (root / f"{run_id}.summary.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        return result


def collect_worker(request_path: str | Path) -> None:
    """Execute one real task with configured variants and record measured states."""
    import gymnasium
    from embodichain.lab.gym.utils.gym_utils import (
        add_env_launcher_args_to_parser,
        build_env_cfg_from_args,
    )
    from embodichain.lab.gym.utils.registration import (
        discover_task_packages,
        execute_init_hooks,
    )
    from embodichain.lab.gym.envs.demo import execute_demo_episode
    from embodichain.lab.sim.sensors.camera import CameraCfg
    from embodichain.lab.sim.material import VisualMaterialCfg
    from embodichain.lab.visualization.cfg import VisualizationCfg
    from embodichain.lab.visualization.scene_exporter import SceneExporter

    request = Path(request_path).resolve()
    root = request.parent
    record = json.loads(request.read_text())
    desired = record["provenance"]["desired"]
    repo = Path(__file__).resolve().parents[2]
    config = (
        repo
        / "embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.franka.yaml"
    )
    parser = argparse.ArgumentParser()
    add_env_launcher_args_to_parser(parser, require_gym_config=True)
    args = parser.parse_args(
        [
            "--gym_config",
            str(config),
            "--headless",
            "--seed",
            str(record["seed"]),
            "--max_episodes",
            "1",
        ]
    )
    discover_task_packages()
    execute_init_hooks()

    def modify(cfg: dict[str, Any]) -> None:
        cube = cfg["rigid_object"][0]
        cube["shape"]["size"] = [desired["size"]] * 3
        cube["init_pos"] = [desired["x"], desired["y"], desired["size"] / 2]
        cfg["light"]["direct"][0]["intensity"] = desired["light"]

    cfg, gym_config, actions = build_env_cfg_from_args(args, gym_config_modifier=modify)
    env = None
    with Catalog(root / "catalog.sqlite") as catalog:
        try:
            env = gymnasium.make(id=gym_config["id"], cfg=cfg, **actions)
            env.reset(seed=record["seed"])
            target = env.unwrapped
            sim = target.sim
            robot = target.robot
            cube = sim.get_rigid_object("cube")
            mat = desired["material"]
            color = (
                [0.12, 0.35, 0.8, 1.0]
                if mat == "matte_blue"
                else [0.9, 0.35, 0.06, 1.0]
            )
            cube.set_visual_material(
                sim.create_visual_material(
                    VisualMaterialCfg(
                        uid="analysis_cube",
                        base_color=color,
                        roughness=0.85 if mat == "matte_blue" else 0.15,
                    )
                )
            )
            sim.add_sensor(
                CameraCfg(
                    uid="analysis_camera",
                    width=320,
                    height=240,
                    intrinsics=(280, 280, 160, 120),
                    extrinsics=CameraCfg.ExtrinsicsCfg(
                        eye=(-1.35, 1.25, 1.05),
                        target=(-0.3, 0.15, 0.2),
                        up=(0.0, 0.0, 1.0),
                    ),
                    enable_color=True,
                    enable_depth=False,
                )
            )
            exporter = SceneExporter(
                sim, VisualizationCfg(env_ids=[0]), run_id=record["episode_id"]
            )
            manifest = exporter.build_manifest()
            tcp_link = next(
                (n for n in robot.link_names if n.endswith("hand_tcp")), None
            )
            if tcp_link is None:
                raise ValueError("Task embodiment has no hand_tcp link.")
            rows = {
                k: []
                for k in (
                    "timestamps",
                    "qpos",
                    "tcp_position",
                    "object_position",
                    "scene_positions",
                    "scene_wxyz",
                    "scene_visible",
                    "camera_positions",
                    "camera_wxyz",
                )
            }
            cameras = []

            def numpy(value: Any) -> np.ndarray:
                return value.detach().cpu().numpy().copy()

            def capture() -> None:
                step = len(rows["timestamps"])
                timestamp = step * target.step_dt
                frame = exporter.capture(
                    sim_step=step, sim_time=timestamp, capture_dynamic_geometry=False
                ).frame
                rows["timestamps"].append(timestamp)
                rows["qpos"].append(numpy(robot.get_qpos()[0]))
                rows["tcp_position"].append(numpy(robot.get_link_pose(tcp_link)[0, :3]))
                rows["object_position"].append(numpy(cube.get_local_pose()[0, :3]))
                for key, attribute in [
                    ("scene_positions", "positions"),
                    ("scene_wxyz", "wxyz"),
                    ("scene_visible", "visible"),
                    ("camera_positions", "camera_positions"),
                    ("camera_wxyz", "camera_wxyz"),
                ]:
                    rows[key].append(getattr(frame, attribute).copy())
                images = exporter.capture_camera_images(
                    sim_step=step, sim_time=timestamp
                ).frame.images
                if images:
                    cameras.append(images[0].image.copy())

            capture()
            outcome = execute_demo_episode(StepObserver(env, capture))
            metadata = target.get_demo_episode_metadata(0)
            arrays = {k: np.asarray(v) for k, v in rows.items()}
            metrics = physical_metrics(arrays, target_xy=(-0.4, 0.48))
            metrics["program_reported_success"] = outcome.all_success
            metrics["program_completed"] = outcome.completed
            metrics["sample_count"] = len(rows["timestamps"])
            record["metrics"] = metrics
            record["segments"] = json_value(metadata.get("segments", []))
            record["provenance"].update(
                task_config=str(config.relative_to(repo)),
                measurement_version="1",
                execution_policy="trajectory_open_loop; projected semantic effects",
                native_demo_metadata=metadata,
            )
            d = record["dimensions"]
            for key, value, source, unit, scope in [
                (
                    "asset",
                    f"cube_{round(desired['size']*1000)}mm",
                    "configured",
                    "",
                    "episode",
                ),
                ("material", mat, "configured", "", "episode"),
                (
                    "light",
                    desired["light"],
                    "configured",
                    "renderer_intensity",
                    "episode",
                ),
                ("affordance", "antipodal_grasp", "annotated", "", "episode"),
                ("approach", "task_program_pick_up", "annotated", "", "episode"),
                ("duration_s", metrics["duration_s"], "measured", "s", "episode"),
            ]:
                d[key] = _measurement(value, source, unit, scope)
            for i, key in enumerate(("pose_x", "pose_y", "pose_z")):
                d[key] = _measurement(
                    float(arrays["object_position"][0, i]), "measured", "m", "initial"
                )
            d["yaw"] = _measurement(0.0, "configured", "rad", "initial")
            # Keep geometric families unknown until a validated clustering policy exists.
            d["trajectory_family"] = unknown_measurement(
                "No geometry-clustering policy configured; measured path length is available."
            )
            accepted = outcome.completed and metrics["physical_success"]
            if accepted:
                record.update(status="pending_write", reason="")
                catalog.upsert(record)
            try:
                record["artifacts"] = write_recording(
                    root / "episodes" / record["episode_id"],
                    arrays,
                    json_value(manifest),
                    camera=np.asarray(cameras) if cameras else None,
                )
                record.update(
                    status=(
                        "committed"
                        if outcome.completed and metrics["physical_success"]
                        else "rejected"
                    ),
                    reason=(
                        ""
                        if outcome.completed and metrics["physical_success"]
                        else "Task completion or geometric physical check failed"
                    ),
                )
                catalog.upsert(record)
            except Exception as exc:
                record.update(
                    status="partial_commit", reason=f"{type(exc).__name__}: {exc}"
                )
                catalog.upsert(record)
                raise
            env.reset(options={"save_data": False})
            print(
                json.dumps(
                    {
                        "episode": record["episode_id"],
                        "metrics": metrics,
                        "status": record["status"],
                    }
                ),
                flush=True,
            )
        except Exception as exc:
            current = catalog.get(record["episode_id"])
            if current["status"] == "proposed":
                current.update(
                    status="rollout_failed", reason=f"{type(exc).__name__}: {exc}"
                )
                catalog.upsert(current)
            raise
        finally:
            if env is not None:
                env.unwrapped.close(exit_process=False)
