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

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from embodichain.gen_sim.scene_providers import SceneGenerationRequest
from embodichain.gen_sim.workflow import load_generated_scene, qualify_task_scene


def _write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_scene_request_binds_image_and_requirements_bytes(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    requirements = tmp_path / "scene_requirements.json"
    image_hash = _write(image, b"image-bytes")
    requirements_hash = _write(requirements, b"{}\n")
    request = SceneGenerationRequest(
        task_id="case",
        image_path=image,
        image_sha256=image_hash,
        requirements_path=requirements,
        requirements_sha256=requirements_hash,
        output=tmp_path / "output",
    )
    copied_image = tmp_path / "moved/image.png"
    copied_requirements = tmp_path / "moved/requirements.json"
    _write(copied_image, b"image-bytes")
    _write(copied_requirements, b"{}\n")
    moved = SceneGenerationRequest(
        task_id="case",
        image_path=copied_image,
        image_sha256=image_hash,
        requirements_path=copied_requirements,
        requirements_sha256=requirements_hash,
        output=tmp_path / "different-output",
    )
    assert moved.request_hash == request.request_hash
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="image changed"):
        SceneGenerationRequest(
            task_id="case",
            image_path=image,
            image_sha256=image_hash,
            requirements_path=requirements,
            requirements_sha256=requirements_hash,
            output=tmp_path / "another-output",
        )


def test_generated_scene_contract_verifies_graph_prims_and_artifacts(
    tmp_path: Path,
) -> None:
    asset_hash = _write(tmp_path / "assets/asset.usdz", b"usd-package")
    table_hash = _write(tmp_path / "assets/table.glb", b"table-mesh")
    cube_hash = _write(tmp_path / "assets/cube.glb", b"cube-mesh")
    manifest = {
        "schema_version": "embodichain.generated-scene/v1",
        "scene_id": "scene-1",
        "source": {"image_sha256": "a" * 64},
        "coordinates": {
            "editable_frame": "Y-up",
            "runtime_frame": "Z-up",
            "units": "m",
        },
        "runtime_asset": {"path": "assets/asset.usdz", "sha256": asset_hash},
        "objects": [
            {
                "uid": "table",
                "kind": "rigid",
                "name": "table",
                "description": "wood table",
                "category": "table",
                "source_prim_path": "/Asset/table",
                "mesh_path": "assets/table.glb",
                "mesh_sha256": table_hash,
                "pose": {
                    "position": [0, 0, 0],
                    "rotation_euler_xyz_degrees": [0, 0, 0],
                    "scale": [1, 1, 1],
                },
                "physics": {"body_type": "kinematic"},
            },
            {
                "uid": "cube",
                "kind": "rigid",
                "name": "cube",
                "description": "red cube",
                "category": "block",
                "source_prim_path": "/Asset/cube",
                "mesh_path": "assets/cube.glb",
                "mesh_sha256": cube_hash,
                "pose": {
                    "position": [0, 0, 0.1],
                    "rotation_euler_xyz_degrees": [0, 0, 0],
                    "scale": [1, 1, 1],
                },
                "physics": {"body_type": "dynamic"},
            },
        ],
        "graph": {
            "nodes": [
                {"object_id": "table", "parent_id": None},
                {"object_id": "cube", "parent_id": "table"},
            ],
            "relations": [],
        },
        "artifacts": {
            "assets/asset.usdz": asset_hash,
            "assets/table.glb": table_hash,
            "assets/cube.glb": cube_hash,
        },
    }
    (tmp_path / "generated_scene.json").write_text(json.dumps(manifest))
    assert load_generated_scene(tmp_path)["scene_id"] == "scene-1"
    manifest["graph"]["nodes"].pop()
    (tmp_path / "generated_scene.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="cover every object"):
        load_generated_scene(tmp_path)


def test_qualification_publishes_rejection_for_unsupported_task_scope(
    tmp_path: Path,
) -> None:
    task = tmp_path / "task"
    scene = tmp_path / "scene"
    task.mkdir()
    scene.mkdir()
    image_hash = _write(task / "source_image.png", b"image")
    requirements = {
        "schema_version": "embodichain.scene-requirements/v1",
        "task_id": "unsupported-task",
        "candidate_id": "candidate-1",
        "image_sha256": image_hash,
    }
    (task / "scene_requirements.json").write_text(json.dumps(requirements))
    (task / "task_candidates.json").write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "candidate_id": "candidate-1",
                        "draft": {
                            "steps": [
                                {
                                    "task_type": "E2",
                                    "relation": "on",
                                }
                            ]
                        },
                    }
                ]
            }
        )
    )
    inventory = {
        path.relative_to(task).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in task.iterdir()
    }
    (task / "task_plan_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "embodichain.task-plan-artifact/v1",
                "task_id": "unsupported-task",
                "candidate_id": "candidate-1",
                "instruction": "unsupported",
                "image_sha256": image_hash,
                "artifacts": inventory,
            }
        )
    )

    asset_hash = _write(scene / "asset.usdz", b"usd")
    mesh_hash = _write(scene / "table.glb", b"mesh")
    requirements_hash = hashlib.sha256(
        (task / "scene_requirements.json").read_bytes()
    ).hexdigest()
    (scene / "generated_scene.json").write_text(
        json.dumps(
            {
                "schema_version": "embodichain.generated-scene/v1",
                "scene_id": "scene-1",
                "requirements_sha256": requirements_hash,
                "source": {"image_sha256": image_hash},
                "coordinates": {
                    "editable_frame": "Y-up",
                    "runtime_frame": "Z-up",
                    "units": "m",
                },
                "runtime_asset": {"path": "asset.usdz", "sha256": asset_hash},
                "objects": [
                    {
                        "uid": "table",
                        "kind": "rigid",
                        "name": "table",
                        "description": "table",
                        "category": "table",
                        "source_prim_path": "/Asset/table",
                        "mesh_path": "table.glb",
                        "mesh_sha256": mesh_hash,
                        "pose": {
                            "position": [0, 0, 0],
                            "rotation_euler_xyz_degrees": [0, 0, 0],
                            "scale": [1, 1, 1],
                        },
                        "physics": {"body_type": "kinematic"},
                    }
                ],
                "graph": {
                    "nodes": [{"object_id": "table", "parent_id": None}],
                    "relations": [],
                },
                "artifacts": {
                    "asset.usdz": asset_hash,
                    "table.glb": mesh_hash,
                },
            }
        )
    )
    runtime = tmp_path / "runtime.yaml"
    policy = tmp_path / "policy.yaml"
    runtime.write_text("{}\n")
    policy.write_text("{}\n")

    output = qualify_task_scene(
        task,
        scene,
        runtime,
        policy,
        tmp_path / "qualification",
    )

    report = json.loads((output / "scene_qualification.json").read_text())
    assert report["status"] == "rejected"
    assert report["issues"] == [
        {
            "code": "unsupported_task_scope",
            "owner": "task_plan",
            "retry_scope": "none",
            "message": "Generated-scene workflow v1 requires one E1 on/inside step.",
        }
    ]
