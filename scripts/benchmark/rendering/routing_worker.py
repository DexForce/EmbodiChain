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

"""Absolute-script worker: original imports, untimed routing, checkpoint then close."""

from __future__ import annotations

import argparse
from copy import deepcopy
import faulthandler
from pathlib import Path
from typing import Any

import frozen_source as frozen

__all__ = ["main"]


def _pin_sources(source: dict[str, Any], backend: str) -> None:
    """Keep checkout provenance while importing only the original backend's paths."""
    imports = dict(source)
    if backend != "isaaclab":
        imports["isaaclab_root"] = None
    frozen.pin_source_imports(imports)


def main() -> int:
    """Run only after the separate source-pinned launcher explicitly starts it."""
    faulthandler.enable()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    payload = frozen.read_json(args.config)
    source, binding = payload["source"], payload["original_binding"]
    adapter = None
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "original_source_routing_worker",
        "status": "failed",
        "original_binding": binding,
        "throughput_claim": False,
        "publication_eligible": False,
        "cleanup_status": "not_started",
    }
    try:
        for item in payload["proof_sources"]:
            if frozen.sha256(Path(item["path"])) != item["sha256"]:
                raise ValueError(
                    "Follow-up proof implementation changed after freezing"
                )
            snapshot = output / item["snapshot"]["path"]
            if frozen.sha256(snapshot) != item["sha256"]:
                raise ValueError("Frozen follow-up source snapshot changed")
        _pin_sources(source, binding["backend"])
        # Every scripts.* module below comes from the ORIGINAL worktree.
        from scripts.benchmark.rendering import formal_worker as original
        from scripts.benchmark.rendering.resources import ResourceSampler
        import routing_proof

        frozen.runtime_versions(binding)
        config = payload["worker_config"]
        values = dict(config["case"])
        values["modalities"] = tuple(values["modalities"])
        case = original.RenderCase(**values)
        scene, replay = original._preflight(config, case)
        result["observed_source_files_before"] = frozen.verify_imported_sources(source)
        with ResourceSampler(gpu_id=payload["gpu_id"]) as sampler:
            gpu = sampler.summary()["gpu"]
            if not gpu.get("uuid") or gpu["uuid"] != binding["hardware_id"]:
                raise ValueError("Routing GPU UUID differs from the timed original")
            if binding.get("gpu_driver") and gpu.get("driver") != binding["gpu_driver"]:
                raise ValueError("Routing GPU driver differs from the timed original")
            result["observed_gpu"] = gpu
            adapter = original._create_adapter(
                binding["backend"], config, case, payload["gpu_id"]
            )
            result["observed_runtime"] = frozen.runtime_versions(
                binding, getattr(adapter, "torch", None)
            )
            if adapter.metadata.get("physics_enabled") is not False:
                raise ValueError("Routing proof requires explicitly disabled physics")
            state = (
                original.frame_at(replay, 0)
                if replay is not None
                else original._initial_state(scene)
            )
            adapter.apply_state(state)
            views = [state["cameras"][name] for name in original._camera_names(case)]
            proof = routing_proof.run_routing_probe(
                adapter,
                backend=binding["backend"],
                num_envs=case.num_envs,
                camera_specs=views,
                modalities=case.modalities,
                width=case.width,
                height=case.height,
                output=output / "routing",
                binding=binding,
                capture=lambda index: original._capture(adapter, case, index),
                **payload["probe_policy"],
            )
            if hasattr(adapter, "refresh_metadata"):
                adapter.refresh_metadata()
            result["observed_renderer"] = deepcopy(adapter.metadata)
            controls = frozen.renderer_control_signature(
                adapter.metadata, binding["backend"]
            )
            if controls != payload["expected_renderer_controls"]:
                raise ValueError(
                    "Routing renderer/light controls differ from the timed original"
                )
            result["observed_renderer_controls"] = controls
            result["observed_source_files_after"] = frozen.verify_imported_sources(
                source
            )
            result["binding_verified"] = True
            result["routing_status"] = proof["routing_status"]
            result["status"] = proof["status"]
            result["routing_proof"] = frozen.artifact(
                output,
                output / "routing/routing_proof.json",
                "indexed_camera_routing_proof",
            )
        result["resources"] = sampler.summary()
        result["resource_scope"] = (
            "untimed diagnostic only; never included in original performance/memory metrics"
        )
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        proof_path = output / "routing/routing_proof.json"
        if proof_path.is_file():
            result["routing_proof"] = frozen.artifact(
                output, proof_path, "indexed_camera_routing_proof"
            )
        result["cleanup_status"] = "pending" if adapter is not None else "not_needed"
        # Isaac app.close may terminate the interpreter successfully. Persist
        # result and all source bindings before it, as the original worker does.
        frozen.write_json(output / "result.json", result)
        if adapter is not None:
            try:
                adapter.close()
                result["cleanup_status"] = "completed"
            except Exception as error:
                result["cleanup_status"] = "failed"
                result["cleanup_error"] = f"{type(error).__name__}: {error}"
            frozen.write_json(output / "result.json", result)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
