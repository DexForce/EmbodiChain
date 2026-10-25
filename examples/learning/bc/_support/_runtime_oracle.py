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

"""Bounded source-AST and actual-checkpoint runtime comparison with PR828."""

from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import time
import h5py
import torch


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


from embodichain_models.policy.bc import ChunkBCModel, ChunkBCModelCfg, ChunkBCExecutor
from ._pick_place import decode_request
from embodichain.utils.math import axis_angle_from_quat, quat_from_matrix

__all__: list[str] = []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only actual checkpoint/queue/decode oracle against PR828 source AST."
    )
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    REFERENCE = args.reference_root.resolve()
    CHECKPOINT, CORPUS = args.checkpoint.resolve(), args.manifest.resolve()
    BC_ROOT = Path(__file__).resolve().parents[4]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(
        ["git", "-C", str(REFERENCE), "rev-parse", "HEAD"], text=True
    ).strip()
    assert revision == "51f1f0cd5a7602f23358c142997e9045323f547a"
    from ._reference import _load_reference

    reference_models, visual = _load_reference(REFERENCE)
    visual_file = REFERENCE / "examples/ods/data_stream_demo/visual_policy.py"
    reference_eval_file = REFERENCE / "examples/ods/data_stream_demo/training_eval.py"
    function = next(
        node
        for node in ast.parse(reference_eval_file.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_relative_request"
    )
    namespace = {
        "torch": torch,
        "rotation_from_6d": reference_models.rotation_from_6d,
        "axis_angle_from_quat": axis_angle_from_quat,
        "quat_from_matrix": quat_from_matrix,
    }
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            str(reference_eval_file),
            "exec",
        ),
        namespace,
    )
    reference_decode = namespace["_relative_request"]
    torch.set_num_threads(4)
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    assert state["learner"]["num_updates"] == 2000
    weights = state["learner"]["model"]
    model = ChunkBCModel(ChunkBCModelCfg(**weights["_extra_state"]["config"])).eval()
    model.load_state_dict(weights)
    reference = visual.VisualChunkBC(
        weights["observation_mean"],
        weights["observation_std"],
        weights["target_mean"],
        weights["target_std"],
        tuple(model.cfg.cameras),
        prediction_horizon=model.cfg.prediction_horizon,
    ).eval()
    converted = {
        {"observation_mean": "proprio_mean", "observation_std": "proprio_std"}.get(
            name, name
        ): value
        for name, value in weights.items()
        if name != "_extra_state"
    }
    converted["_extra_state"] = reference.get_extra_state()
    reference.load_state_dict(converted, strict=True)
    normalization_equal = all(
        torch.equal(getattr(model, name), getattr(reference, other))
        for name, other in [
            ("observation_mean", "proprio_mean"),
            ("observation_std", "proprio_std"),
            ("target_mean", "target_mean"),
            ("target_std", "target_std"),
        ]
    )
    manifest = json.loads(CORPUS.read_text())
    assert manifest["complete"] is True
    cameras = tuple(model.cfg.cameras)
    assert cameras == tuple(manifest["cameras"])
    executor = ChunkBCExecutor(model, action_horizon=8)
    forward_outputs = []
    hook = model.register_forward_hook(
        lambda m, inputs, output: forward_outputs.append(output.detach().clone())
    )
    results = []
    failures = []
    maxima = {
        "normalized_h2_forward": 0.0,
        "physical_chunk": 0.0,
        "queued_target": 0.0,
        "decoded_request": 0.0,
    }
    started = time.monotonic()

    def compare(name, actual, expected, episode, frame):
        difference = float((actual - expected).abs().max())
        maxima[name] = max(maxima[name], difference)
        matches = torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)
        if not matches:
            failures.append(
                {
                    "kind": name,
                    "episode": episode,
                    "frame": frame,
                    "max_abs": difference,
                }
            )
        return difference

    try:
        with torch.no_grad():
            for ordinal in [0, 31, 127]:
                record = manifest["episodes"][ordinal]
                episode_path = CORPUS.parent / record["path"]
                file_hash = sha(episode_path)
                assert file_hash == record["sha256"]
                steps = min(128, record["length"])
                assert steps >= 64
                previous_geometry = None
                previous_images = None
                queue = None
                queue_index = 0
                replans = 0
                executor.reset()
                traces = []
                with h5py.File(episode_path, "r") as handle:
                    assert len(handle["geometry"]) == record["length"]
                    for frame in range(steps):
                        geometry = (
                            torch.from_numpy(handle["geometry"][frame])
                            .float()
                            .unsqueeze(0)
                        )
                        images = {
                            camera: torch.from_numpy(handle["images"][camera][frame])
                            .movedim(-1, -3)
                            .unsqueeze(0)
                            .float()
                            .div_(255)
                            for camera in cameras
                        }
                        forward_outputs.clear()
                        target = executor.action(geometry[:, :27], images)
                        replanned = queue is None or queue_index >= 8
                        forward_error = None
                        if replanned:
                            history = torch.stack(
                                (
                                    (
                                        geometry[:, :27]
                                        if previous_geometry is None
                                        else previous_geometry
                                    ),
                                    geometry[:, :27],
                                ),
                                1,
                            )
                            image_history = {
                                camera: torch.stack(
                                    (
                                        (
                                            images[camera]
                                            if previous_images is None
                                            else previous_images[camera]
                                        ),
                                        images[camera],
                                    ),
                                    1,
                                )
                                for camera in cameras
                            }
                            normalized = reference(history, image_history)
                            queue = (
                                normalized * reference.target_std
                                + reference.target_mean
                            )
                            queue_index = 0
                            replans += 1
                            assert len(forward_outputs) == 1
                            forward_error = compare(
                                "normalized_h2_forward",
                                forward_outputs[0],
                                normalized,
                                ordinal,
                                frame,
                            )
                            compare(
                                "physical_chunk",
                                forward_outputs[0] * model.target_std
                                + model.target_mean,
                                queue,
                                ordinal,
                                frame,
                            )
                        else:
                            assert len(forward_outputs) == 0
                        reference_target = queue[:, queue_index]
                        target_error = compare(
                            "queued_target", target, reference_target, ordinal, frame
                        )
                        request = decode_request(target, geometry)
                        reference_request = reference_decode(reference_target, geometry)
                        request_error = compare(
                            "decoded_request",
                            request,
                            reference_request,
                            ordinal,
                            frame,
                        )
                        traces.append(
                            {
                                "frame": frame,
                                "replan": replanned,
                                "normalized_forward_max_abs": forward_error,
                                "target_max_abs": target_error,
                                "request_max_abs": request_error,
                                "sl_request": request[0].tolist(),
                                "reference_request": reference_request[0].tolist(),
                            }
                        )
                        queue_index += 1
                        previous_geometry = geometry[:, :27].clone()
                        previous_images = images
                results.append(
                    {
                        "ordinal": ordinal,
                        "path": str(episode_path),
                        "sha256": file_hash,
                        "frames": steps,
                        "replans": replans,
                        "traces": traces,
                    }
                )
                print(
                    json.dumps(
                        {
                            "episode": ordinal,
                            "frames": steps,
                            "replans": replans,
                            "max_abs_so_far": maxima,
                        }
                    ),
                    flush=True,
                )
    finally:
        hook.remove()

    report = {
        "protocol": "bc-actual-checkpoint-runtime-equivalence-v1",
        "passed": not failures and normalization_equal,
        "bc_commit": subprocess.check_output(
            ["git", "-C", str(BC_ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "model_namespace": "embodichain_models.policy.bc",
        "model_source_sha256": sha(
            BC_ROOT
            / "extensions/embodichain_models/embodichain_models/policy/bc/model.py"
        ),
        "reference_commit": revision,
        "reference_visual_sha256": sha(visual_file),
        "reference_encoder_sha256": sha(
            REFERENCE / "embodichain/learning/models/resnet_encoder.py"
        ),
        "reference_targets_sha256": sha(
            REFERENCE / "examples/ods/simple_bc_demo/models.py"
        ),
        "reference_runtime_sha256": sha(reference_eval_file),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha(CHECKPOINT),
        "policy_step": state["learner"]["num_updates"],
        "manifest": str(CORPUS),
        "manifest_sha256": sha(CORPUS),
        "camera_order": cameras,
        "normalization_equal": normalization_equal,
        "history_length": 2,
        "action_horizon": 8,
        "device": "cpu",
        "torch": torch.__version__,
        "threads": 4,
        "atol": 1e-5,
        "rtol": 1e-4,
        "total_frames": sum(row["frames"] for row in results),
        "forward_replans": sum(row["replans"] for row in results),
        "max_abs": maxima,
        "failures": failures,
        "episodes": results,
        "wall_seconds": time.monotonic() - started,
        "scope": "Read-only inference on recorded training frames; no environment replay, dev/test panel or parameter updates.",
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in [
                    "passed",
                    "total_frames",
                    "forward_replans",
                    "max_abs",
                    "wall_seconds",
                ]
            }
        ),
        flush=True,
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
