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

"""Native camera collection and closed-loop evaluation for the DexUni Cube."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import time
from typing import Any
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import torch

from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.cfg import DenoisingCfg, RenderCfg

from .cube._env import OGBenchCubeCfg, OGBenchCubeEnv
from ._official_data import HDF5_SHA256
from ._reference import _sha

_NATIVE_RENDER = {
    "renderer": "hybrid",
    "offscreen_denoising": "optix",
    "robot_materials": "compiled-urdf",
    "camera": "front_pixels",
    "width": 224,
    "height": 224,
}
_OFFICIAL_WEIGHTS_SHA256 = (
    "2839a907362f403f9136383016e91774373a295d958ae75121791f22a9fddf89"
)


def prepare_dexuni_reference(reference: Path, output: Path) -> dict[str, Any]:
    """Write a separate reference bundle with authored OBJ visual materials."""
    root = reference.resolve()
    if output.resolve().is_relative_to(root):
        raise ValueError("The native Cube bundle must be outside the reference input")
    source_robot = root / "robot"
    urdf = source_robot / "robot.urdf"
    colors = {}
    for visual in ET.parse(urdf).findall("./link/visual"):
        mesh = visual.find("./geometry/mesh")
        color = visual.find("./material/color")
        if mesh is None or color is None:
            continue
        relative = (source_robot / mesh.attrib["filename"]).resolve().relative_to(root)
        red, green, blue, alpha = color.attrib["rgba"].split()
        rgba = (red, green, blue, alpha)
        if relative in colors and colors[relative] != rgba:
            raise ValueError(
                f"One compiled Cube mesh has conflicting colors: {relative}"
            )
        colors[relative] = rgba
    if not colors:
        raise ValueError("The compiled Cube URDF has no mesh colors")
    shutil.copytree(root, output)
    files = {}
    for name, (red, green, blue, alpha) in colors.items():
        mesh_path = output / name
        material_path = mesh_path.with_suffix(".mtl")
        obj = mesh_path.read_text()
        if any(line.startswith(("mtllib ", "usemtl ")) for line in obj.splitlines()):
            raise ValueError(
                f"Expected a compiled OBJ without material bindings: {name}"
            )
        before, marker, after = obj.partition("\nf ")
        if not marker:
            raise ValueError(f"The compiled OBJ has no faces: {name}")
        material_path.write_text(
            f"newmtl source_color\nKd {red} {green} {blue}\n"
            f"Ka 0 0 0\nKs 0.15 0.15 0.15\nNs 28\nd {alpha}\n"
        )
        mesh_path.write_text(
            f"mtllib {material_path.name}\n{before}\nusemtl source_color{marker}{after}"
        )
        for path in (mesh_path, material_path):
            files[str(path.relative_to(output))] = _sha(path)
    report = {
        "source_urdf_sha256": _sha(urdf),
        "visual_meshes": len(colors),
        "sha256": files,
    }
    (output / "dexuni-render-materials.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def _make_cube(
    reference: Path, num_envs: int, gpu_id: int, *, images: bool = True
) -> OGBenchCubeEnv:
    cfg = OGBenchCubeCfg(
        reference_root=str(reference), num_envs=num_envs, enable_images=images
    )
    cfg.sim_cfg.gpu_id = gpu_id
    cfg.sim_cfg.render_cfg = RenderCfg(
        renderer="hybrid" if images else "no-render",
        denoising=DenoisingCfg(offscreen="optix"),
    )
    cfg.sim_cfg.enable_entity_gizmo = False
    cfg.sim_cfg.robot_ik_gizmo = None
    cfg.sim_cfg.startup_summary = "off"
    return OGBenchCubeEnv(cfg)


def _pixels(obs: Any) -> np.ndarray:
    image = obs["sensor"]["main_cam"]["color"][..., :3].cpu().numpy().copy()
    if image.dtype != np.uint8 or image.shape[1:] != (224, 224, 3):
        raise RuntimeError("Unexpected native Cube camera image")
    return image


def _query_metadata(queries: Path) -> dict[str, Any]:
    metadata = json.loads(queries.with_suffix(".json").read_text())
    if (
        metadata["num_eval"] != 50
        or metadata["seed"] != 42
        or metadata["goal_offset_steps"] != 25
        or metadata["source_sha256"] != HDF5_SHA256
        or metadata["queries_sha256"] != _sha(queries)
    ):
        raise ValueError("Official 50-query contract or checksum mismatch")
    return metadata


def _planner_weights(
    checkpoint: Path, output: Path, metadata: dict[str, Any]
) -> tuple[Path, str]:
    if checkpoint.is_file():
        digest = _sha(checkpoint)
        if digest != _OFFICIAL_WEIGHTS_SHA256:
            raise ValueError("Official Cube checkpoint checksum mismatch")
        return checkpoint, digest

    from embodichain.extensions.models.dynamics.lewm import load_lewm

    model, manifest = load_lewm(checkpoint)
    training = manifest["metadata"]
    if (
        training.get("render") != _NATIVE_RENDER
        or training.get("environment") != "OGBenchCubeDexUni-v0"
        or training.get("physics", {}).get("solver") != "dexuni"
        or not set(metadata["episode_ids"]).issubset(training["eval_episodes_excluded"])
    ):
        raise ValueError(
            "Export must be trained on native Cube frames with held-out queries"
        )
    for key in ("action_mean", "action_std"):
        actual = getattr(model, key)
        torch.testing.assert_close(
            actual, torch.tensor(metadata[key], dtype=actual.dtype).repeat(5)
        )
    state = model.state_dict()
    del state["action_mean"], state["action_std"]
    upstream = output / "upstream-compatible-weights.pt"
    torch.save(state, upstream)
    return upstream, manifest["weights_sha256"]


def _reset_at(
    env: OGBenchCubeEnv, qpos: np.ndarray, qvel: np.ndarray, goal: np.ndarray
) -> tuple[Any, np.ndarray, np.ndarray]:
    from ._official_eval import _native_state

    step_before = env.sim.physics_step_index
    obs, _ = env.reset(seed=42, options={"qpos": qpos, "qvel": qvel, "goal_pos": goal})
    if env.sim.physics_step_index != step_before:
        raise RuntimeError("Restoring a Cube query advanced physics")
    pos, vel = (batch[0] for batch in _native_state(env))
    np.testing.assert_allclose(pos[:17], qpos[:17], atol=2e-6, rtol=0)
    np.testing.assert_allclose(abs(pos[17:] @ qpos[17:]), 1, atol=2e-6, rtol=0)
    np.testing.assert_allclose(vel, qvel, atol=2e-5, rtol=0)
    return obs, pos, vel


@torch.inference_mode()
def evaluate_dexuni_cube(
    *,
    reference: Path,
    queries: Path,
    checkpoint: Path,
    swm_root: Path,
    output: Path,
    device: str,
    gpu_id: int,
    preview_queries: tuple[int, ...] = (),
) -> dict[str, Any]:
    """Run official CEM over live native RGB and native renders of frozen goal poses."""
    from ._official_cube import CUBE_SWM_REV
    from ._official_eval import _native_state, _original_cube_planner

    metadata = _query_metadata(queries)
    count, block_steps, budget = 50, 25, 50
    if any(not 0 <= index < count for index in preview_queries):
        raise ValueError("Preview query indices must be between 0 and 49")
    with h5py.File(queries, "r") as source:
        qpos, qvel, goal_qpos, goal_qvel, goals = (
            source[key][:]
            for key in (
                "qpos",
                "qvel",
                "goal_qpos",
                "goal_qvel",
                "goal_privileged_block_0_pos",
            )
        )
    output.mkdir(parents=True)
    weights, weight_digest = _planner_weights(checkpoint, output, metadata)
    plan = _original_cube_planner(swm_root, weights, metadata, device)
    initial_pixels = np.empty((count, 224, 224, 3), dtype=np.uint8)
    goal_pixels = np.empty_like(initial_pixels)
    replan_pixels = np.zeros_like(initial_pixels)
    replanned = np.zeros(count, dtype=bool)
    first_success = np.full(count, -1, dtype=np.int32)
    positions = np.empty((budget + 1, count, 21), dtype=np.float64)
    velocities = np.empty((budget + 1, count, 20), dtype=np.float64)
    actions = np.full((budget, count, 5), np.nan, dtype=np.float32)
    plans = np.full((2, count, block_steps, 5), np.nan, dtype=np.float32)
    start = time.monotonic()
    env = _make_cube(reference, 1, gpu_id)
    try:
        solver = env.sim.spawn_result.get_solver_diagnostics()
        if not solver or solver["solver_type"] != "dexuni" or solver["empty"]:
            raise RuntimeError("The native Cube did not start a DexUni solver")
        for index in range(count):
            obs, _, _ = _reset_at(env, goal_qpos[index], goal_qvel[index], goals[index])
            goal_pixels[index] = _pixels(obs)[0]
            obs, _, _ = _reset_at(env, qpos[index], qvel[index], goals[index])
            initial_pixels[index] = _pixels(obs)[0]
        plans[0] = plan(initial_pixels, goal_pixels, np.arange(count)).reshape(
            count, block_steps, 5
        )
        print({"phase": "initial_plan", "queries": count}, flush=True)
        for index in range(count):
            obs, positions[0, index], velocities[0, index] = _reset_at(
                env, qpos[index], qvel[index], goals[index]
            )
            preview = [_pixels(obs)[0]] if index in preview_queries else None
            # CEM uses batch size one. Planning for the surviving rows in ascending
            # order consumes its RNG as in the official second planning barrier.
            for step in range(budget):
                block, offset = divmod(step, block_steps)
                if step == block_steps:
                    replan_pixels[index] = _pixels(obs)[0]
                    replanned[index] = True
                    plans[1, index] = plan(
                        replan_pixels[index : index + 1],
                        goal_pixels[index : index + 1],
                        np.array([0]),
                    ).reshape(block_steps, 5)
                command = plans[block, index, offset]
                actions[step, index] = command
                obs, _, _, _, info = env.step(torch.from_numpy(command[None]))
                if preview is not None:
                    preview.append(_pixels(obs)[0])
                pos, vel = (batch[0] for batch in _native_state(env))
                positions[step + 1, index], velocities[step + 1, index] = pos, vel
                if bool(info["success"][0]):
                    first_success[index] = step + 1
                    positions[step + 2 :, index] = pos
                    velocities[step + 2 :, index] = vel
                    break
            if preview is not None:
                np.savez_compressed(
                    output / f"preview-query-{index:02d}.npz",
                    frames=preview,
                    goal=goal_pixels[index],
                )
            progress = {
                "queries_completed": index + 1,
                "success_count": int((first_success > 0).sum()),
                "elapsed_seconds": time.monotonic() - start,
            }
            (output / "progress.json").write_text(json.dumps(progress) + "\n")
            print(progress, flush=True)
    finally:
        env.close()
        SimulationManager.flush_cleanup_queue()

    np.savez_compressed(
        output / "trace.npz",
        qpos=positions,
        qvel=velocities,
        actions=actions,
        plans=plans,
        goals=goals,
    )
    np.savez_compressed(
        output / "observations.npz",
        initial=initial_pixels,
        goal=goal_pixels,
        replan=replan_pixels,
        replanned=replanned,
    )
    successes = first_success > 0
    return {
        "kind": "official_cube_50_query_dexuni_native",
        "environment": "OGBenchCubeDexUni-v0",
        "solver": "dexuni",
        "solver_diagnostics": solver,
        "render": _NATIVE_RENDER,
        "initial_and_goal_source": "EmbodiChain camera at frozen query states",
        "replan_source": "EmbodiChain camera after 25 live control steps",
        "queries_sha256": metadata["queries_sha256"],
        "query_episode_ids": metadata["episode_ids"],
        "query_start_steps": metadata["start_steps"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": weight_digest,
        "stable_worldmodel_revision": CUBE_SWM_REV,
        "cem": {
            "seed": 42,
            "samples": 300,
            "iterations": 30,
            "topk": 30,
            "horizon": 5,
            "action_block": 5,
            "batch_size": 1,
        },
        "num_eval": count,
        "eval_budget": budget,
        "planned_queries_per_phase": [count, int(replanned.sum())],
        "success_count": int(successes.sum()),
        "success_rate": float(successes.mean() * 100),
        "episode_successes": successes.tolist(),
        "first_success_steps": first_success.tolist(),
        "final_goal_distances": np.linalg.norm(
            positions[-1, :, 14:17] - goals, axis=1
        ).tolist(),
        "elapsed_seconds": time.monotonic() - start,
    }
