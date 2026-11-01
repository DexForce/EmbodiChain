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

"""Compare an authored installation with measured joint-search results in 3D.

Run with ``embodichain run-example expansion/scene_robot_feasibility/preview``.
The generated HTML needs neither a simulator nor a network connection to replay.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import copy
from dataclasses import fields, is_dataclass
import json
import os
from pathlib import Path
from typing import Any
import webbrowser

import numpy as np

__all__ = ["main"]


def _json_value(value: Any) -> Any:
    """Convert owned simulation/contract values without pickle or executable code."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {f.name: _json_value(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    return value


def _workflow(output: Path, source: Path, demonstration: bool) -> Path:
    from embodichain.utils.utility import load_config

    config = copy.deepcopy(load_config(source))
    if demonstration:
        # Deliberately poor *installation*, retaining the original physical task.
        config["request"]["entity_poses"]["UR5"][0][3] = 0.75
        installation = next(
            v for v in config["request"]["variables"] if v["name"] == "installation"
        )
        installation["reference_pose"][0][3] = 0.75
        installation["lower"] = [-0.8, -0.08, -0.12]
        installation["upper"] = [0.05, 0.08, 0.12]
        config["search"].update(
            max_wall_time_s=300.0, max_candidates=8, max_regions=1, validation_samples=1
        )
    for key in ("deployment", "program"):
        if key in config["task_program"]:
            resource = (source.parent / config["task_program"][key]).resolve()
            if key == "deployment" and demonstration and not resource.is_file():
                # Wheels keep examples under share/ and official task configs
                # in the bundled extension package, rather than extensions/.
                from importlib.resources import files

                resource = Path(str(files("embodichain_tasks.configs"))) / (
                    "tasks/manipulation/repeated_pick_place/task.ur5.yaml"
                )
            if not resource.is_file():
                raise FileNotFoundError(resource)
            config["task_program"][key] = str(resource)
    target = output / "workflow.json"
    target.write_text(json.dumps(config, indent=2, allow_nan=False))
    return target


def _baseline(config: dict, output: Path) -> dict:
    from embodichain.lab.sim.scene.feasibility import FeasibilityBudget
    from embodichain.lab.task_program.integrations.feasibility import (
        build_feasibility_host,
    )

    prepared, host = build_feasibility_host(
        config, output_directory=output / "baseline"
    )
    # The authored zero offsets have their own exact position in this joint box.
    parameters = []
    for variable in prepared.request.variables:
        for lo, hi in zip(variable.lower, variable.upper):
            if not lo <= 0 <= hi:
                raise ValueError("Preview baseline requires zero offsets inside bounds")
            parameters.append(0.0 if lo == hi else -lo / (hi - lo))
    candidate = prepared.candidate(
        parameters, mode_id=prepared.request.modes[0].mode_id, ordinal=0
    )
    budget = FeasibilityBudget(prepared.cfg)
    try:
        host.prepare(prepared.request, budget)
        budget.charge("candidates")
        evaluation = host.evaluate(candidate, budget)
        return {
            "candidate_id": candidate.candidate_id,
            "request_id": candidate.request_id,
            "accepted": evaluation.accepted,
            "checks": [_json_value(c) for c in evaluation.checks],
            "initial_state_id": evaluation.initial_state_id,
        }
    finally:
        host.close()


def _asset_environment(config: dict) -> Any:
    """Create one read-only asset/FK source; this environment does not replay actions."""
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.gym.utils.registration import REGISTERED_ENVS
    from embodichain.utils.utility import load_config

    adapter = config["task_program"]
    deployment = Path(adapter["deployment"])
    data = load_config(deployment)
    cfg = config_to_cfg(
        data,
        source_path=deployment,
        task_program_path_override=adapter.get("program"),
    )
    cfg.num_envs = cfg.sim_cfg.num_envs = 1
    cfg.sim_cfg.device = "cpu"
    cfg.sim_cfg.headless = True
    cfg.sim_cfg.render_cfg.renderer = "no-render"
    cfg.enable_sensor = False
    cfg.dataset = None
    cfg.init_rollout_buffer = cfg.record_trajectory = cfg.trajectory_auto_save = False
    cfg.light.direct, cfg.light.indirect = [], None
    return REGISTERED_ENVS[data["id"]].make(cfg=cfg)


def _asset_model(env: Any) -> tuple[list[dict], list[str]]:
    from embodichain.lab.visualization.cfg import VisualizationCfg
    from embodichain.lab.visualization.scene_exporter import SceneExporter
    from urllib.parse import unquote

    exporter = SceneExporter(env.sim, VisualizationCfg(env_ids=[0]))
    manifest = exporter.build_manifest()
    geometries = {g.geometry_id: g for g in manifest.geometries}
    meshes, links = [], []
    for node in manifest.nodes:
        target = exporter.resolve_node_target(node.node_id)
        if target is None:
            continue
        uid, kind = target
        if uid == "default_plane":
            continue  # The browser draws a reference floor/grid instead.
        if kind not in {"robot", "rigid"} or node.dynamic_geometry:
            raise ValueError("Preview supports recorded rigid scenes and one robot")
        geometry = geometries[node.geometry_id]
        vertices = np.asarray(geometry.vertices, dtype=np.float32)
        faces = np.asarray(geometry.faces, dtype=np.int32)
        normals = np.zeros_like(vertices)
        a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
        face_normals = np.cross(b - a, c - a)
        for column in range(3):
            np.add.at(normals, faces[:, column], face_normals)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        link = unquote(node.node_id.rsplit("/link:", 1)[1]) if kind == "robot" else None
        if link is not None:
            links.append(link)
        meshes.append(
            {
                "uid": uid,
                "kind": kind,
                "link": link,
                "vertices": vertices.reshape(-1),
                "normals": normals.reshape(-1),
                "faces": faces.reshape(-1),
            }
        )
    if not links or not meshes:
        raise ValueError("Native asset export produced no preview geometry")
    return meshes, links


def _matrices(
    env: Any, meshes: list[dict], links: list[str], q: np.ndarray, entities: dict
) -> np.ndarray:
    import torch

    robot = env.robot
    names = robot.pk_chain.get_joint_parameter_names()
    indices = [robot.joint_names.index(name) for name in names]
    with torch.no_grad():
        fk = (
            robot.compute_fk(
                torch.as_tensor(
                    q[:, indices], dtype=torch.float32, device=robot.device
                ),
                link_names=links,
                qpos_joint_names=names,
            )
            .detach()
            .cpu()
            .numpy()
        )
    lookup = {link: index for index, link in enumerate(links)}
    result = np.empty((len(q), len(meshes), 4, 4), dtype=np.float32)
    for index, mesh in enumerate(meshes):
        if mesh["kind"] == "robot":
            result[:, index] = entities[mesh["uid"]] @ fk[:, lookup[mesh["link"]]]
        else:
            result[:, index] = entities[mesh["uid"]]
    # WebGL consumes column-major matrices. These are FK of measured joints,
    # never the requested controller targets or an interpolated success animation.
    return result.swapaxes(-1, -2).reshape(len(q), len(meshes), 16)


def _preview_data(config: dict, root: Path, atlas: Any, baseline: dict) -> dict:
    from embodichain.lab.sim import SimulationManager

    owned = _asset_environment(config)
    env = getattr(owned, "unwrapped", owned)
    try:
        env.reset(options={"save_data": False})
        if len(env.sim.get_robot_uid_list()) != 1:
            raise ValueError("Preview supports one robot, including composed dual arms")
        meshes, links = _asset_model(env)
        base_entities = {
            uid: np.asarray(pose, dtype=np.float32)[None]
            for uid, pose in atlas.request.entity_poses.items()
        }
        initial_q = env.robot.get_qpos().detach().cpu().numpy()
        baseline["matrices"] = _matrices(env, meshes, links, initial_q, base_entities)[
            0
        ]
        baseline["robot_pose"] = base_entities[env.robot.uid][0]
        baseline["entities"] = {uid: pose[0] for uid, pose in base_entities.items()}
        # Check the FK/mesh convention against native measured links before export.
        native_root = env.robot.get_local_pose(to_matrix=True)[0].detach().cpu().numpy()
        native_entities = dict(base_entities, **{env.robot.uid: native_root[None]})
        check = _matrices(env, meshes, links, initial_q, native_entities)[0]
        for i, mesh in enumerate(meshes):
            if mesh["link"] is not None:
                native = (
                    env.robot.get_link_pose(mesh["link"], to_matrix=True)[0]
                    .detach()
                    .cpu()
                    .numpy()
                )
                if not np.allclose(check[i].reshape(4, 4).T, native, atol=2e-4, rtol=0):
                    raise ValueError("Preview FK does not match native link frames")
        episodes = {}
        for path in sorted((root / "episodes").glob("*.npz")):
            with np.load(path, allow_pickle=False) as saved:
                metadata = json.loads(str(saved["metadata"]))
                candidate_id = metadata["metadata"]["candidate_id"]
                observation = next(
                    (
                        o
                        for o in atlas.observations
                        if o.candidate.candidate_id == candidate_id
                    ),
                    None,
                )
                if (
                    observation is None
                    or not observation.evaluation.accepted
                    or not observation.evaluation.committed
                ):
                    raise ValueError(
                        "Preview episode lacks accepted, confirmed atlas evidence"
                    )
                if metadata["metadata"]["request_id"] != atlas.request.request_id:
                    raise ValueError("Preview episode belongs to a different request")
                if (
                    metadata["identity"]["initial_state_id"]
                    != observation.evaluation.initial_state_id
                ):
                    raise ValueError(
                        "Preview episode does not match the accepted initial state"
                    )
                q = saved["observation/joint_positions"]
                times = saved["timestamps"]
                if len(times) != len(q) or len(q) != len(saved["actions"]) + 1:
                    raise ValueError("Preview requires measured T+1 episode states")
                # Keep actual samples at at most 30 Hz, preserving both endpoints.
                stride = max(
                    1,
                    int(np.ceil(1 / (30 * np.median(np.diff(times))))),
                    int(np.ceil((len(q) - 1) / 240)),
                )
                selected = np.unique(np.r_[np.arange(0, len(q), stride), len(q) - 1])
                entities = {
                    mesh["uid"]: saved["observation/entity/" + mesh["uid"]][
                        selected
                    ].copy()
                    for mesh in meshes
                }
                episodes[candidate_id] = {
                    "times": times[selected].copy(),
                    "matrices": _matrices(env, meshes, links, q[selected], entities),
                    "path": str(path.relative_to(root)),
                    "initial_state_id": metadata["identity"]["initial_state_id"],
                }
        trials = []
        for observation in atlas.observations:
            candidate, evaluation = observation.candidate, observation.evaluation
            entities = dict(base_entities)
            for change in candidate.scene.pose_changes:
                entities[change.entity_uid] = np.asarray(
                    change.arena_pose, dtype=np.float32
                )[None]
            for uid, pose in candidate.deployment.poses.items():
                entities[uid] = np.asarray(pose, dtype=np.float32)[None]
            trials.append(
                {
                    "id": candidate.candidate_id,
                    "accepted": evaluation.accepted,
                    "committed": evaluation.committed,
                    "holdout": observation.region_id is not None,
                    "completed": observation.completed,
                    "checks": [_json_value(c) for c in evaluation.checks],
                    "parameters": candidate.parameters,
                    "robot_pose": entities[env.robot.uid][0],
                    "entities": {uid: pose[0] for uid, pose in entities.items()},
                    "matrices": _matrices(env, meshes, links, initial_q, entities)[0],
                    "episode": episodes.get(candidate.candidate_id),
                }
            )
        return _json_value(
            {
                "schema": "scene-feasibility-preview/v1",
                "task": atlas.request.task_id,
                "request_id": atlas.request.request_id,
                "robot_uid": env.robot.uid,
                "object_uid": next(
                    c["entity"]
                    for c in config["task_program"]["final_checks"]
                    if c["kind"] == "pose_near"
                ),
                "goal": next(
                    c["position"]
                    for c in config["task_program"]["final_checks"]
                    if c["kind"] == "pose_near"
                ),
                "meshes": meshes,
                "baseline": baseline,
                "trials": trials,
                "regions": atlas.regions,
                "metrics": dict(atlas.metrics),
                "variables": atlas.request.variables,
            }
        )
    finally:
        try:
            owned.close()
        finally:
            SimulationManager.flush_cleanup_queue()


def main() -> None:
    """Generate the physical comparison, or reopen a previously generated preview."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/scene_feasibility_preview")
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Existing UR5 workflow; omit for the contrasting installation demo",
    )
    parser.add_argument(
        "--from-run",
        type=Path,
        help="Reuse an existing atlas.json and episodes/ with --config",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Rebuild HTML from output/preview.json without loading simulation",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the standalone HTML in the default browser",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.reuse:
        data = json.loads((output / "preview.json").read_text())
    else:
        # This example creates and tears down several hosts in one process.
        os.environ["EMBODICHAIN_SIM_EXIT_PROCESS"] = "0"
        import torch
        from embodichain.lab.task_program.integrations.feasibility import (
            run_feasibility,
        )
        from embodichain.lab.sim.scene.feasibility import load_feasibility_atlas

        torch.set_num_threads(4)
        if args.from_run is not None and args.config is None:
            parser.error("--from-run requires the original --config")
        if (output / "preview.json").exists():
            parser.error("Preview exists; use --reuse or choose a new output directory")
        source = (
            args.config.resolve()
            if args.config
            else Path(__file__).with_name("search.ur5.yaml")
        )
        workflow = _workflow(output, source, args.config is None)
        config = json.loads(workflow.read_text())
        print(
            "Checking the authored installation with the same task gates…", flush=True
        )
        baseline = _baseline(config, output)
        root = args.from_run.resolve() if args.from_run else output / "search"
        if args.from_run:
            atlas = load_feasibility_atlas(root / "atlas.json")
        else:
            print(
                "Searching permitted layouts and installations; physical budget: 300 s in the default demo…",
                flush=True,
            )
            atlas = run_feasibility(workflow, root)
        if baseline["request_id"] != atlas.request.request_id:
            raise ValueError(
                "Baseline and search must use the same resolved task request"
            )
        data = _preview_data(config, root, atlas, baseline)
        (output / "preview.json").write_text(json.dumps(data, allow_nan=False))
    payload = (
        json.dumps(data, allow_nan=False)
        .replace("<", r"\u003c")
        .replace("&", r"\u0026")
    )
    html = Path(__file__).with_name("preview.html").read_text()
    target = output / "index.html"
    target.write_text(html.replace("__PREVIEW_DATA__", payload), encoding="utf-8")
    print(f"Preview: {target}", flush=True)
    if args.open:
        webbrowser.open(target.as_uri())


if __name__ == "__main__":
    main()
