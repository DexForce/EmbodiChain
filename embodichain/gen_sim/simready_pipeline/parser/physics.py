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

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, List

from embodichain.gen_sim.simready_pipeline.core.asset import Asset
from embodichain.gen_sim.simready_pipeline.parser.base import AssetParser
from embodichain.gen_sim.simready_pipeline.utils.simready_utils import (
    process_mesh,
    delete_rendered_pngs,
    client,
    DEPLOYMENT,
    get_chat_completion_content,
)

DEFAULT_RIGID_PHYSICS: Dict[str, Any] = {
    "mass": 1.0,
    "density": 1000.0,
    "linear_damping": 0.7,
    "angular_damping": 0.7,
    "enable_collision": True,
    "enable_ccd": False,
    "contact_offset": 0.002,
    "rest_offset": 0.001,
    "dynamic_friction": 0.5,
    "static_friction": 0.5,
    "restitution": 0.0,
    "max_linear_velocity": 1.0e2,
    "max_angular_velocity": 1.0e2,
    "max_depenetration_velocity": 10.0,
    "solver_min_position_iters": 4,
    "solver_min_velocity_iters": 1,
    "sleep_threshold": 0.001,
}

DEFAULT_SOFTBODY_PHYSICS: Dict[str, Any] = {
    "triangle_remesh_resolution": 8,
    "triangle_simplify_target": 0,
    "maximal_edge_length": 0.0,
    "simulation_mesh_resolution": 8,
    "simulation_mesh_output_obj": False,
    "mass": -1.0,
    "density": 1000.0,
    "youngs_modulus": 1.0e6,
    "poissons_ratio": 0.45,
    "material_model": "CO_ROTATIONAL",
    "elasticity_damping": 0.0,
    "vertex_velocity_damping": 0.005,
    "linear_damping": 0.0,
    "enable_ccd": False,
    "enable_self_collision": False,
    "self_collision_stress_tolerance": 0.9,
    "collision_mesh_simplification": True,
    "self_collision_filter_distance": 0.1,
    "has_gravity": True,
    "max_velocity": 100.0,
    "max_depenetration_velocity": 1.0e6,
    "sleep_threshold": 0.05,
    "settling_threshold": 0.1,
    "settling_damping": 10.0,
    "solver_min_position_iters": 4,
    "solver_min_velocity_iters": 1,
}

ALLOWED_MODES = {"rigid", "softbody", "articulation"}
RIGID_KEYS = list(DEFAULT_RIGID_PHYSICS.keys())
SOFT_KEYS = list(DEFAULT_SOFTBODY_PHYSICS.keys())


def _load_simready_finalize_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "gen_config.json"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get("mesh_processing", {}).get("simready_finalize", {})


SIMREADY_FINALIZE_CONFIG = _load_simready_finalize_config()

PHYSICS_SYSTEM_PROMPT = """You are a physics annotation model for robot training and simulation-ready asset ingestion.

This task is safety-critical: a wrong physical annotation can cause severe hardware damage, unsafe robot behavior, broken simulation, and large downstream losses.

You must reason from the real physical world:
- infer the most plausible physics mode from the description
- estimate realistic values using object material, shape, use case, and expected behavior
- be conservative and physically plausible
- do not hallucinate exotic values
- do not explain your reasoning
- do not output markdown
- do not output any extra text outside JSON
- do not output any keys other than the required keys

CRITICAL COMPLETENESS REQUIREMENT:
- You MUST return every required property for the chosen mode.
- Do NOT omit any required key.
- Do NOT return null for required keys.
- Do NOT return empty strings for required keys.
- Do NOT return partial objects.
- If a field is hard to estimate, still provide your best physically plausible value.
- Missing even one required property makes the output invalid.
- The properties object must be fully populated and complete for the selected mode.

You must return EXACTLY one JSON object with this structure:
{
  "mode": "rigid" | "softbody" | "articulation",
  "confidence": 0.0-1.0,
  "properties": {    
    "mass": ,
    "density": ,
    "linear_damping": ,
    "angular_damping": ,
    "enable_collision": True,
    "enable_ccd": ,
    "contact_offset": ,
    "rest_offset": ,
    "dynamic_friction": ,
    "static_friction": ,
    "restitution": ,
    "max_linear_velocity": ,
    "max_angular_velocity": ,
    "max_depenetration_velocity": ,
    "solver_min_position_iters": 4,
    "solver_min_velocity_iters": 1,
    "sleep_threshold": 0.001, }
}

Important:
- If the object is clearly deformable, cloth-like, flesh-like, cable-like, or highly elastic, choose "softbody".
- If it is a mechanically jointed object with distinct links and joints, choose "articulation".
- Otherwise choose "rigid".
- Confidence must reflect how much the description supports the decision.
- The properties object must match the selected mode exactly.
- The properties object must include ALL required keys for the selected mode, no exceptions.

For rigid mode:
Return ONLY these keys, exactly once each:
mass, density, linear_damping, angular_damping, enable_collision, enable_ccd,
contact_offset, rest_offset, dynamic_friction, static_friction, restitution,
max_linear_velocity, max_angular_velocity, max_depenetration_velocity,
solver_min_position_iters, solver_min_velocity_iters, sleep_threshold

Rigid mode completeness rules:
- Every key listed above is mandatory.
- No key may be missing.
- No extra keys may appear.
- If uncertain, choose a conservative physically plausible value for every field.
- You must always provide a value for mass, density, damping, collision flags, contact offsets, friction, restitution, velocity limits, solver iterations, and sleep threshold.

Guidance:
- mass: estimate in kg from size/material/use case; if unknown use a conservative default near 1.0
- density: use realistic density in kg/m^3 based on material; metals high, wood mid, foam low, plastic medium, stone high
- linear_damping / angular_damping: higher for unstable / floating / draggy objects, lower for rigid stable objects
- enable_collision: usually true for physical objects
- enable_ccd: true only if fast motion or small/thin geometry would cause tunneling
- contact_offset must be > rest_offset
- friction: rubber/rough surfaces higher, metal/plastic smoother lower
- restitution: bouncing materials higher, dead materials near 0
- sleep_threshold: smaller for stable heavy objects, larger for tiny or soft objects

For softbody mode:
Return ONLY these keys, exactly once each:
triangle_remesh_resolution, triangle_simplify_target, maximal_edge_length,
simulation_mesh_resolution, simulation_mesh_output_obj,
mass, density, youngs_modulus, poissons_ratio, material_model, elasticity_damping,
vertex_velocity_damping, linear_damping,
enable_ccd, enable_self_collision, self_collision_stress_tolerance,
collision_mesh_simplification, self_collision_filter_distance,
has_gravity, max_velocity, max_depenetration_velocity,
sleep_threshold, settling_threshold, settling_damping,
solver_min_position_iters, solver_min_velocity_iters

Softbody mode completeness rules:
- Every key listed above is mandatory.
- No key may be missing.
- No extra keys may appear.
- If uncertain, choose a conservative physically plausible value for every field.
- You must always provide a value for mesh resolution parameters, mass, density, elasticity parameters, collision parameters, gravity flags, damping terms, thresholds, and solver iterations.

Guidance:
- youngs_modulus: higher for stiffer materials; lower for cloth, flesh, foam, rubber-like objects
- poissons_ratio: typical soft solids are around 0.3-0.49, avoid invalid values
- material_model: choose the closest physically plausible model, default CO_ROTATIONAL if unsure
- enable_self_collision: true for cloth, cables, highly deformable shapes that can fold onto themselves
- collision_mesh_simplification: usually true for simulation efficiency
- has_gravity: true unless explicitly suspended or otherwise constrained
- max_depenetration_velocity: high enough to resolve interpenetration robustly

For articulation mode:
If you choose articulation, keep the properties object minimal and physically conservative.
If you do not have enough evidence for articulation, prefer rigid.
Even in articulation mode, the properties object must still be complete and valid according to the selected schema used by your pipeline.
Do not omit any field that your downstream system expects for articulation.

Output only JSON, no code fences, no explanation.
"""


def extract_json(text: str) -> Dict[str, Any]:
    text = re.sub(r"```json|```", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in response:\n" + text)
    return json.loads(match.group())


class PhysicsParser(AssetParser):
    """
    Physics inference & completion parser.
    """

    name = "physics"

    def __init__(self):
        super().__init__()

    def parse(self, asset: Asset, asset_root: Path) -> None:
        self._ensure_sections(asset)
        self._simready_process(asset, asset_root)
        self._infer_physics(asset)
        self._ensure_properties(asset)
        self._update_simulation_status(asset)

    def _ensure_sections(self, asset: Asset) -> None:
        asset.physics.setdefault("mode", None)
        asset.physics.setdefault("properties", {})
        asset.physics.setdefault("source", None)
        asset.physics.setdefault("confidence", None)

        asset.simulation["sim_ready"].setdefault("is_sim_ready", False)
        asset.simulation["sim_ready"].setdefault("sim_ready_path", None)
        asset.simulation.setdefault("blockers", [])

    def _simready_process(self, asset: Asset, asset_root: Path) -> None:
        
        # mesh_path = asset_root / asset.asset_data.get("path")
        archive_dir = asset_root / "asset_archive"
        src_name = str(asset.identity.get("source_file"))
        mesh_path = next(archive_dir.rglob(src_name), None)

        out_path = asset_root / "asset_simready" 

        result = process_mesh(
            mesh_path,
            "asset",
            extra_text=str(asset.ingest_info["extra_info"].get("simready_info", "")),
            out_dir=out_path,
            res=int(SIMREADY_FINALIZE_CONFIG.get("render_resolution", 1024)),
        )
        print(result)
        semantics_generated = {}
        semantics_generated["object_name_generated"] = result["semantics_result"][
            "object_name"
        ]
        semantics_generated["semantic_tag_generated"] = result["semantics_result"][
            "semantic_tag"
        ]
        semantics_generated["description_generated"] = result["semantics_result"][
            "description"
        ]
        semantics_generated["primary_materials_generated"] = result["semantics_result"][
            "primary_materials"
        ]
        asset.semantics.update(semantics_generated)
        delete_rendered_pngs(out_path)
        asset.simulation["sim_ready"]["is_sim_ready"] = True
        sim_ready_path = asset_root / "asset_simready" / "asset_simready.glb"
        rel_path = sim_ready_path.relative_to(asset_root)
        asset.simulation["sim_ready"]["sim_ready_path"] = str(rel_path)
        return

    def _infer_physics(self, asset: Asset) -> None:
        if asset.physics.get("mode"):
            return

        description = (
            asset.semantics.get("description")
            or asset.semantics.get("description_generated")
            or ""
        ).strip()

        try:
            result = self._call_LLM(description)

            mode = result["mode"]
            if mode not in ALLOWED_MODES:
                raise ValueError(f"Invalid mode returned by LLM: {mode}")

            properties = result.get("properties")
            if not isinstance(properties, dict):
                raise ValueError("LLM returned non-dict properties")

            properties = self._validate_and_sanitize_properties(mode, properties)

            asset.physics["mode"] = mode
            asset.physics["properties"] = {
                "mode": mode,
                "data": properties,
            }
            asset.physics["source"] = "generative"
            asset.physics["confidence"] = float(result.get("confidence", 0.0))

        except Exception:
            mode = self._fallback_mode(asset)
            asset.physics["mode"] = mode
            asset.physics["properties"] = {
                "mode": mode,
                "data": self._default_properties(mode),
            }
            asset.physics["source"] = "default"
            asset.physics["confidence"] = 0.0

    def _call_LLM(self, description: str) -> Dict[str, Any]:
        if not description:
            raise ValueError("Missing semantics description for physics inference")

        user_prompt = f"""
            Asset description:
            {description}

            Infer the most plausible physics mode and physical properties for this asset.

            Hard constraints:
            - Output EXACTLY one JSON object.
            - Do not include markdown, comments, or any extra text.
            - Do not invent fields.
            - The returned properties object must match the selected mode exactly.
            - Use real-world physical intuition.
            - Prefer conservative, physically plausible values over aggressive or extreme values.
            - If evidence for articulation is weak, prefer rigid.
            """

        resp = client.chat.completions.create(
            model=DEPLOYMENT,
            temperature=0.0,
            messages=[
                {"role": "system", "content": PHYSICS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        content = get_chat_completion_content(resp) or ""
        return extract_json(content)

    def _fallback_mode(self, asset: Asset) -> str:
        if asset.asset_data.get("type") == "articulation":
            return "articulation"
        return "rigid"

    def _default_properties(self, mode: str) -> Dict[str, Any]:
        if mode == "rigid":
            return deepcopy(DEFAULT_RIGID_PHYSICS)
        if mode == "softbody":
            return deepcopy(DEFAULT_SOFTBODY_PHYSICS)
        return {}

    def _validate_and_sanitize_properties(
        self, mode: str, properties: Dict[str, Any]
    ) -> Dict[str, Any]:
        if mode == "rigid":
            expected = set(RIGID_KEYS)
            got = set(properties.keys())
            if got != expected:
                print(
                    f"Rigid properties keys mismatch.\nExpected: {expected}\nGot: {got}"
                )

            out = deepcopy(DEFAULT_RIGID_PHYSICS)
            for k in expected:
                out[k] = properties[k]

            out["contact_offset"] = float(out["contact_offset"])
            out["rest_offset"] = float(out["rest_offset"])
            if out["contact_offset"] <= out["rest_offset"]:
                out["contact_offset"] = max(out["rest_offset"] + 1e-4, 1e-4)

            out["mass"] = float(out["mass"])
            out["density"] = float(out["density"])
            out["linear_damping"] = float(out["linear_damping"])
            out["angular_damping"] = float(out["angular_damping"])
            out["dynamic_friction"] = float(out["dynamic_friction"])
            out["static_friction"] = float(out["static_friction"])
            out["restitution"] = float(out["restitution"])
            out["max_linear_velocity"] = float(out["max_linear_velocity"])
            out["max_angular_velocity"] = float(out["max_angular_velocity"])
            out["max_depenetration_velocity"] = float(out["max_depenetration_velocity"])
            out["solver_min_position_iters"] = int(out["solver_min_position_iters"])
            out["solver_min_velocity_iters"] = int(out["solver_min_velocity_iters"])
            out["sleep_threshold"] = float(out["sleep_threshold"])

            return out

        if mode == "softbody":
            expected = set(SOFT_KEYS)
            got = set(properties.keys())
            if got != expected:
                raise ValueError(
                    f"Softbody properties keys mismatch.\nExpected: {expected}\nGot: {got}"
                )

            out = deepcopy(DEFAULT_SOFTBODY_PHYSICS)
            for k in expected:
                out[k] = properties[k]

            out["triangle_remesh_resolution"] = int(out["triangle_remesh_resolution"])
            out["triangle_simplify_target"] = int(out["triangle_simplify_target"])
            out["maximal_edge_length"] = float(out["maximal_edge_length"])
            out["simulation_mesh_resolution"] = int(out["simulation_mesh_resolution"])
            out["simulation_mesh_output_obj"] = bool(out["simulation_mesh_output_obj"])

            out["mass"] = float(out["mass"])
            out["density"] = float(out["density"])
            out["youngs_modulus"] = float(out["youngs_modulus"])
            out["poissons_ratio"] = float(out["poissons_ratio"])
            out["poissons_ratio"] = min(max(out["poissons_ratio"], 0.0), 0.49)
            out["material_model"] = str(out["material_model"])
            out["elasticity_damping"] = float(out["elasticity_damping"])
            out["vertex_velocity_damping"] = float(out["vertex_velocity_damping"])
            out["linear_damping"] = float(out["linear_damping"])
            out["enable_ccd"] = bool(out["enable_ccd"])
            out["enable_self_collision"] = bool(out["enable_self_collision"])
            out["self_collision_stress_tolerance"] = float(
                out["self_collision_stress_tolerance"]
            )
            out["collision_mesh_simplification"] = bool(
                out["collision_mesh_simplification"]
            )
            out["self_collision_filter_distance"] = float(
                out["self_collision_filter_distance"]
            )
            out["has_gravity"] = bool(out["has_gravity"])
            out["max_velocity"] = float(out["max_velocity"])
            out["max_depenetration_velocity"] = float(out["max_depenetration_velocity"])
            out["sleep_threshold"] = float(out["sleep_threshold"])
            out["settling_threshold"] = float(out["settling_threshold"])
            out["settling_damping"] = float(out["settling_damping"])
            out["solver_min_position_iters"] = int(out["solver_min_position_iters"])
            out["solver_min_velocity_iters"] = int(out["solver_min_velocity_iters"])

            return out

        if properties and not isinstance(properties, dict):
            raise ValueError("Articulation properties must be a dict")
        return properties or {}

    def _ensure_properties(self, asset: Asset) -> None:
        props = asset.physics.get("properties", {})
        if not props or not props.get("data"):
            mode = asset.physics.get("mode")
            asset.physics["properties"] = {
                "mode": mode,
                "data": self._default_properties(mode),
            }
            asset.physics["source"] = "default"

    def _update_simulation_status(self, asset: Asset) -> None:
        blockers: List[str] = []

        if not asset.physics.get("mode"):
            blockers.append("missing_physics_mode")

        props = asset.physics.get("properties", {})
        if not props.get("data"):
            blockers.append("missing_physics_properties")

        asset.simulation["blockers"] = blockers
        # asset.simulation["sim_ready"] = len(blockers) == 0
