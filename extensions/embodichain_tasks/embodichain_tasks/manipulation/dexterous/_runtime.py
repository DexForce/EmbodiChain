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

"""Batched manipulation environments stepped by the DexSim Newton backend."""

from __future__ import annotations

import copy
from pathlib import Path
import re
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
import torch
import warp as wp

import dexsim
from embodichain.data import get_data_path
from dexsim.engine.newton_physics import DexUniSolverCfg, MJWarpSolverCfg, NewtonCfg
from dexsim.engine.newton_physics.backend_registry import get_newton_backend

from ._shadow_task import ShadowState, ShadowTask

__all__ = ["DexSimManipulationEnv", "ENV_NAMES"]

ENV_NAMES = {
    "M1": "ShadowPenFixedDexSim-v1",
    "M2": "ShadowPenContinuousDexSim-v1",
    "M3": "LeapCubeFaceTurnDexSim-v1",
    "M4": "LeapCubeReorientDexSim-v1",
    "M5": "LeapCubeRotateZAxisDexSim-v1",
}


def _named_index(model: Any, kind: str, name: str) -> int:
    count = getattr(model, "n" + ("jnt" if kind == "joint" else kind))
    candidates = [
        i
        for i in range(count)
        if getattr(model, kind)(i).name == name
        or getattr(model, kind)(i).name.endswith("_" + name)
        or (
            kind == "geom"
            and re.sub(r"_\d+$", "", getattr(model, kind)(i).name).split("/")[-1]
            == name
        )
    ]
    if len(candidates) != 1:
        raise ValueError(f"Cannot resolve {kind} {name!r}: {candidates}")
    return candidates[0]


def rotate_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    xyz = q[..., 1:]
    t = 2 * torch.cross(xyz, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(xyz, t, dim=-1)


class DexSimManipulationEnv:
    """Expose real DexSim physics through EmbodiChain's vector-env interface.

    The task-local MJCF adapter uses the World-owned backend to retain tendons,
    passive joints and direct actuators. Native stepping belongs to DexSim;
    MuJoCo forward calls below only update kinematics after explicit reset.
    """

    def __init__(
        self,
        task_id: str,
        num_envs: int = 32,
        device: str = "cuda:0",
        seed: int = 42,
        cuda_graph: bool = True,
        solver_kind: str = "mujoco_warp",
        diagnostics: bool = True,
        solver_iterations: tuple[int, int] | None = None,
    ) -> None:
        if solver_kind not in ("mujoco_warp", "dexuni"):
            raise ValueError(f"Unknown manipulation solver: {solver_kind}")
        self.solver_kind = solver_kind
        self.diagnostics = diagnostics
        device = str(torch.device(device))
        self.task_id = task_id
        self.num_envs = num_envs
        self.device = torch.device(device)
        torch.manual_seed(seed)
        if task_id in ("M1", "M2"):
            self.task = ShadowTask.from_assets(
                Path(get_data_path("ShadowHandPen/shadow_pen.xml")).parent,
                num_envs,
                variant=task_id,
                device=device,
            )
            xml = Path(get_data_path("ShadowHandPen/shadow_pen.xml"))
            self.ctrl_dt, self.sim_dt = 0.04, 0.002
            self.observation_dim, self.critic_dim, self.action_dim = 87, 87, 18
        else:
            from ._leap_tasks import LeapTask

            if task_id == "M3":
                xml = Path(
                    get_data_path("LeapHandCubeFaceTurn/leap_cube_top_layer.xml")
                )
                initial = np.load(
                    get_data_path("LeapHandCubeFaceTurn/LeapCubeFaceTurn-initial.npz")
                )
            else:
                xml = Path(get_data_path(f"LeapHandCube/{task_id}.xml"))
                initial = np.load(get_data_path(f"LeapHandCube/{task_id}-initial.npz"))
            source = mujoco.MjModel.from_xml_path(str(xml))
            self.task = LeapTask(
                {
                    "M3": "LeapCubeFaceTurn",
                    "M4": "LeapCubeReorient",
                    "M5": "LeapCubeRotateZAxis",
                }[task_id],
                num_envs,
                initial["qpos"],
                initial["ctrl"],
                source.actuator_ctrlrange[:, 0],
                source.actuator_ctrlrange[:, 1],
                device=device,
                seed=seed,
            )
            self.ctrl_dt, self.sim_dt = 0.05, (0.005 if task_id == "M3" else 0.01)
            self.observation_dim = {"M3": 65, "M4": 57, "M5": 32}[task_id]
            self.critic_dim = {"M3": 65, "M4": 128, "M5": 105}[task_id]
            self.action_dim = 16
        self.source = mujoco.MjModel.from_xml_path(str(xml))
        self.source.opt.timestep = self.sim_dt
        if task_id in ("M1", "M2", "M3"):
            self.source.opt.ccd_iterations = 10
        self.xml = xml
        self.substeps = round(self.ctrl_dt / self.sim_dt)
        observation_spaces = {
            "obs": gym.spaces.Box(
                -np.inf, np.inf, (self.observation_dim,), dtype=np.float32
            )
        }
        if task_id in ("M4", "M5"):
            observation_spaces["critic_obs"] = gym.spaces.Box(
                -np.inf, np.inf, (self.critic_dim,), dtype=np.float32
            )
        self.single_observation_space = gym.spaces.Dict(observation_spaces)
        self.single_action_space = gym.spaces.Box(
            -1, 1, (self.action_dim,), dtype=np.float32
        )
        self.action_space = self.single_action_space
        self.observation_space = self.single_observation_space
        self.action_format = "tensor"
        self.elapsed_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._episode_return = torch.zeros(num_envs, device=device)
        self._episode_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._last_observation: dict[str, torch.Tensor] | None = None
        self._closed = False
        config = dexsim.WorldConfig()
        config.open_windows = False
        config.backend = dexsim.types.Backend.VULKAN
        config.renderer = dexsim.types.Renderer.NORENDER
        config.init_ground = False
        config.cache_path = str(Path.home() / ".cache/embodichain/dexterous")
        solver_options = {
            "use_mujoco_contacts": True,
            "iterations": int(self.source.opt.iterations),
            "ls_iterations": int(self.source.opt.ls_iterations),
            "ccd_iterations": int(self.source.opt.ccd_iterations),
            "integrator": int(self.source.opt.integrator),
            "solver": int(self.source.opt.solver),
            "cone": ("pyramidal", "elliptic")[int(self.source.opt.cone)],
            "impratio": float(self.source.opt.impratio),
            "tolerance": float(self.source.opt.tolerance),
            "ls_tolerance": float(self.source.opt.ls_tolerance),
            "ccd_tolerance": float(self.source.opt.ccd_tolerance),
            "njmax": 512,
            "nconmax": 256,
        }
        self.backend_option_overrides = {}
        if solver_iterations is not None or solver_kind == "dexuni":
            selected_iterations = solver_iterations or (20, 50)
            for name, value in zip(
                ("iterations", "ls_iterations"), selected_iterations
            ):
                self.backend_option_overrides[name] = {
                    "source": solver_options[name],
                    "configured": value,
                    "reason": (
                        "Explicit benchmark settings"
                        if solver_iterations is not None
                        else "DexUni configuration shared with Unitree validation."
                    ),
                }
                solver_options[name] = value
        if solver_kind == "dexuni":
            solver_cfg = DexUniSolverCfg(mujoco_options=solver_options)
        else:
            solver_cfg = MJWarpSolverCfg(**solver_options)
        config.newton_cfg = NewtonCfg(
            device=device,
            gravity=self.source.opt.gravity.tolist(),
            dt=self.sim_dt,
            num_substeps=1,
            sync_to_dexsim=False,
            solver_cfg=solver_cfg,
        )
        config.newton_cfg.use_cuda_graph = cuda_graph
        self.world = dexsim.World(config)
        self.world.set_manual_update(True)
        self.backend = get_newton_backend(self.world)
        template = self.backend._create_builder()
        template.default_shape_cfg.gap = 0.0
        self.mjcf_import_adapter = {}
        if solver_kind == "dexuni":
            from ._dexuni_adapter import import_dexuni_mjcf

            self.mjcf_import_adapter = import_dexuni_mjcf(template, str(xml))
        else:
            template.add_mjcf(
                str(xml),
                ctrl_direct=True,
                parse_visuals=False,
                parse_sites=True,
                collapse_fixed_joints=False,
            )
        for _ in range(num_envs):
            self.backend.builder.add_world(template)
        if solver_kind == "dexuni":
            from ._dexuni_adapter import select_all_articulations

            select_all_articulations(self.backend)
        self.backend.finalize()
        self.solver = self.backend.solver
        self.mujoco_solver = self.backend.mujoco_solver
        self.dexuni_adapter = None
        if solver_kind == "dexuni":
            from ._dexuni_adapter import DexUniStateAdapter

            self.dexuni_adapter = DexUniStateAdapter(self.backend)
        self.imported = self.mujoco_solver.mj_model
        self.data = self.mujoco_solver.mjw_data
        self.option_import_corrections = {
            name: {
                "source": float(getattr(self.source.opt, name)),
                "exported": float(getattr(self.imported.opt, name)),
            }
            for name in ("timestep", "disableflags", "enableflags")
            if getattr(self.source.opt, name) != getattr(self.imported.opt, name)
        }
        for name in ("disableflags", "enableflags"):
            value = int(getattr(self.source.opt, name))
            setattr(self.imported.opt, name, value)
            setattr(self.mujoco_solver.mjw_model.opt, name, value)
        self.imported.opt.timestep = self.sim_dt
        wp.to_torch(self.mujoco_solver.mjw_model.opt.timestep).fill_(self.sim_dt)
        self.backend.invalidate_cuda_graph("source MJCF options restored")
        self._make_mapping()
        self.reset(seed=seed)

    @property
    def unwrapped(self) -> DexSimManipulationEnv:
        return self

    def _make_mapping(self) -> None:
        source, imported = self.source, self.imported
        for name in ("nq", "nv", "nu", "ntendon"):
            if getattr(source, name) != getattr(imported, name):
                raise ValueError(
                    f"MJCF {name} changed: {getattr(source, name)} -> {getattr(imported, name)}"
                )
        qmap, vmap = [], []
        joint_map = {}
        for j in range(source.njnt):
            name = source.joint(j).name
            if name:
                target = _named_index(imported, "joint", name)
            else:
                body = _named_index(
                    imported, "body", source.body(int(source.jnt_bodyid[j])).name
                )
                candidates = np.flatnonzero(imported.jnt_bodyid == body)
                if len(candidates) != 1:
                    raise ValueError("Unnamed joint cannot be resolved unambiguously")
                target = int(candidates[0])
            joint_map[j] = target
            width = (7, 4, 1, 1)[int(source.jnt_type[j])]
            vwidth = (6, 3, 1, 1)[int(source.jnt_type[j])]
            qmap.extend(
                range(
                    int(imported.jnt_qposadr[target]),
                    int(imported.jnt_qposadr[target]) + width,
                )
            )
            vmap.extend(
                range(
                    int(imported.jnt_dofadr[target]),
                    int(imported.jnt_dofadr[target]) + vwidth,
                )
            )
        amap = []
        for a in range(source.nu):
            source_joint = int(source.actuator_trnid[a, 0])
            candidates = np.flatnonzero(
                imported.actuator_trnid[:, 0] == joint_map[source_joint]
            )
            if len(candidates) != 1:
                raise ValueError(f"Actuator {a} mapping ambiguous: {candidates}")
            amap.append(int(candidates[0]))
        self.qmap = torch.tensor(qmap, device=self.device)
        self.vmap = torch.tensor(vmap, device=self.device)
        self.amap = torch.tensor(amap, device=self.device)
        # Native direct-control slots retain source import order, whereas the
        # solver may reorder actuators in its compiled MuJoCo model.
        self.ctrl = wp.to_torch(self.backend.control.mujoco.ctrl).reshape(
            self.num_envs, -1
        )
        self.model_audit = {
            "source": str(self.xml),
            "mjcf_import_adapter": self.mjcf_import_adapter,
            "backend_option_overrides": self.backend_option_overrides,
            "solver_selection": {
                "requested": self.solver_kind,
                "actual_backend": self.backend.solver_type,
                "actual_solver_class": (
                    f"{type(self.solver).__module__}.{type(self.solver).__name__}"
                ),
                "mujoco_branch_class": (
                    f"{type(self.mujoco_solver).__module__}."
                    f"{type(self.mujoco_solver).__name__}"
                ),
                "diagnostics": self.backend.get_solver_diagnostics(),
                "standalone_mjwarp_fallback": False,
            },
            "option_import_corrections": self.option_import_corrections,
            "counts": {
                k: int(getattr(source, k)) for k in ("nq", "nv", "nu", "ntendon")
            },
            "source_reference_counts": {
                k: int(getattr(source, k)) for k in ("nsite", "nsensor")
            },
            "imported_reference_counts": {
                k: int(getattr(imported, k)) for k in ("nsite", "nsensor")
            },
            "observation_state_source": (
                "Named qpos/qvel, body poses, source-local site transforms, "
                "and actuator forces; imported sensor channels are not consumed."
            ),
            "qpos_map": qmap,
            "qvel_map": vmap,
            "actuator_map": amap,
            "gain_max_abs_error": float(
                np.max(
                    np.abs(source.actuator_gainprm - imported.actuator_gainprm[amap])
                )
            ),
            "bias_max_abs_error": float(
                np.max(
                    np.abs(source.actuator_biasprm - imported.actuator_biasprm[amap])
                )
            ),
            "tendon_range_max_abs_error": (
                float(np.max(np.abs(source.tendon_range - imported.tendon_range)))
                if source.ntendon
                else 0.0
            ),
            "solver_options": {
                key: float(getattr(imported.opt, key))
                for key in (
                    "timestep",
                    "iterations",
                    "ls_iterations",
                    "ccd_iterations",
                    "integrator",
                    "solver",
                    "cone",
                    "impratio",
                )
            },
        }
        if (
            max(
                self.model_audit[k]
                for k in (
                    "gain_max_abs_error",
                    "bias_max_abs_error",
                    "tendon_range_max_abs_error",
                )
            )
            > 1e-5
        ):
            raise ValueError(
                f"Imported actuator/tendon properties changed: {self.model_audit}"
            )
        self.body_map = {
            source.body(i).name: _named_index(imported, "body", source.body(i).name)
            for i in range(1, source.nbody)
            if source.body(i).name
        }
        # Newton 1.7 does not export free-joint damping and adds finite force
        # limits to originally unlimited joints. Preserve the source arrays in
        # both MuJoCo representations before any captured physical step.
        corrections = []
        correction_fields = [
            ("dof_damping", vmap),
            ("jnt_actfrclimited", [joint_map[i] for i in range(source.njnt)]),
            ("jnt_actfrcrange", [joint_map[i] for i in range(source.njnt)]),
        ]
        if self.solver_kind == "dexuni":
            correction_fields.append(
                ("jnt_solref", [joint_map[i] for i in range(source.njnt)])
            )
        for field, indices in correction_fields:
            expected = getattr(source, field)
            actual = getattr(imported, field)
            difference = float(np.max(np.abs(expected.astype(float) - actual[indices])))
            if difference:
                corrections.append({"field": field, "max_abs_error_before": difference})
                actual[indices] = expected
                target = wp.to_torch(getattr(self.mujoco_solver.mjw_model, field))
                values = torch.as_tensor(
                    expected, device=self.device, dtype=target.dtype
                )
                if field == "jnt_actfrclimited":
                    target[indices] = values
                else:
                    target[:, indices] = values
        for geom_id in range(source.ngeom):
            name = source.geom(geom_id).name
            if not name or not (
                source.geom_contype[geom_id] or source.geom_conaffinity[geom_id]
            ):
                continue
            try:
                imported_id = _named_index(imported, "geom", name)
            except ValueError:
                continue
            if (
                abs(source.geom_margin[geom_id] - imported.geom_margin[imported_id])
                > 1e-8
            ):
                corrections.append(
                    {
                        "field": f"geom[{name}].margin",
                        "source": float(source.geom_margin[geom_id]),
                        "exported": float(imported.geom_margin[imported_id]),
                    }
                )
                imported.geom_margin[imported_id] = source.geom_margin[geom_id]
                wp.to_torch(self.mujoco_solver.mjw_model.geom_margin)[
                    :, imported_id
                ] = source.geom_margin[geom_id]
        self.model_audit["property_import_corrections"] = corrections
        from ._audit_model import audit_model

        audit_source = source
        if self.backend_option_overrides:
            audit_source = copy.copy(source)
            for name, item in self.backend_option_overrides.items():
                setattr(audit_source.opt, name, item["configured"])
        if self.solver_kind == "dexuni":
            from ._dexuni_adapter import refresh_collision_filters

            mapping_audit = audit_model(
                audit_source, imported, joint_map, amap, self.body_map
            )
            for source_id, imported_id in mapping_audit["collision_geom_map"].items():
                for field in ("geom_contype", "geom_conaffinity"):
                    before = int(getattr(imported, field)[imported_id])
                    expected = int(getattr(source, field)[source_id])
                    if before != expected:
                        corrections.append(
                            {
                                "field": f"geom[{source.geom(source_id).name}].{field}",
                                "source": expected,
                                "exported": before,
                            }
                        )
                        getattr(imported, field)[imported_id] = expected
            self.model_audit["collision_cache_refresh"] = refresh_collision_filters(
                self.mujoco_solver
            )
            self.backend.invalidate_cuda_graph("source collision filters restored")
        self.model_audit["physical_model"] = audit_model(
            audit_source, imported, joint_map, amap, self.body_map
        )
        self.model_audit["physical_model"]["option_reference"] = (
            "Source model with explicit backend_option_overrides."
            if self.backend_option_overrides
            else "Source model."
        )
        from ._audit_tendons import audit_tendons

        self.model_audit["tendons"] = audit_tendons(source, imported, joint_map)
        if self.model_audit["tendons"]["status"] != "passed":
            raise ValueError(f"Imported tendons differ: {self.model_audit['tendons']}")
        if self.model_audit["physical_model"]["status"] != "passed":
            raise ValueError(
                f"Imported physical model differs: {self.model_audit['physical_model']['mismatches']}"
            )
        if self.task_id == "M3":
            labels = self.backend.model.joint_label
            joints = [
                i
                for i, label in enumerate(labels)
                if label == "face_turn"
                or label.endswith("/face_turn")
                or label.endswith("_face_turn")
            ]
            if len(joints) != self.num_envs:
                raise ValueError(f"Expected one passive face joint per world: {joints}")
            starts = wp.to_torch(self.backend.model.joint_qd_start)
            self.face_force_indices = starts[
                torch.tensor(joints, device=self.device)
            ].long()
            self.model_audit["face_force_indices"] = (
                self.face_force_indices.cpu().tolist()
            )

    def _site(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        source_id = self.source.site(name).id
        body_name = self.source.body(int(self.source.site_bodyid[source_id])).name
        body_id = self.body_map[body_name]
        pos = wp.to_torch(self.data.xpos)[:, body_id]
        quat = wp.to_torch(self.data.xquat)[:, body_id]
        local = torch.as_tensor(
            self.source.site_pos[source_id], dtype=torch.float32, device=self.device
        ).expand(self.num_envs, -1)
        return pos + rotate_wxyz(quat, local), quat

    def _state(self) -> Any:
        qpos = wp.to_torch(self.data.qpos)[:, self.qmap]
        qvel = wp.to_torch(self.data.qvel)[:, self.vmap]
        if self.task_id in ("M1", "M2"):
            palm = wp.to_torch(self.data.xpos)[:, self.body_map["robot0:palm"]]
            return ShadowState(qpos, qvel, palm)
        state = {"qpos": qpos, "qvel": qvel}
        if self.task_id == "M3":
            return state
        grasp, palm_quat = self._site("grasp_site")
        tips = []
        for name in ("th_tip", "if_tip", "mf_tip", "rf_tip"):
            tip, _ = self._site(name)
            inverse = palm_quat * palm_quat.new_tensor([1, -1, -1, -1])
            tips.append(rotate_wxyz(inverse, tip - grasp))
        cube_quat = qpos[:, 19:23]
        state.update(
            hand_qpos=qpos[:, :16],
            hand_qvel=qvel[:, :16],
            actuator_force=wp.to_torch(self.data.actuator_force)[:, self.amap],
            cube_pos=qpos[:, 16:19],
            cube_quat=cube_quat,
            cube_linvel=qvel[:, 16:19],
            cube_angvel=rotate_wxyz(cube_quat, qvel[:, 19:22]),
            palm_pos=grasp,
            fingertip_pos=torch.cat(tips, dim=-1),
        )
        return state

    def _observation(self, raw: Any) -> dict[str, torch.Tensor]:
        if isinstance(raw, torch.Tensor):
            return {"obs": raw}
        result = {"obs": raw["state"]}
        if self.critic_dim != self.observation_dim:
            result["critic_obs"] = raw["privileged_state"]
        return result

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict, dict]:
        if seed is not None:
            torch.manual_seed(seed)
            if hasattr(self.task, "generator"):
                self.task.generator.manual_seed(seed)
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if not options
            else options.get("reset_ids")
        )
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        if self.task_id in ("M1", "M2"):
            qpos, qvel, ctrl = self.task.reset(ids)
        else:
            initial = self.task.reset(ids)
            qpos, qvel, ctrl = initial["qpos"], initial["qvel"], initial["ctrl"]
        wp.to_torch(self.data.qpos)[ids[:, None], self.qmap[None, :]] = qpos
        wp.to_torch(self.data.qvel)[ids[:, None], self.vmap[None, :]] = qvel
        self.ctrl[ids] = ctrl
        # Transfer explicitly set coordinates to both DexSim-owned buffers.
        if self.dexuni_adapter is not None:
            self.dexuni_adapter.publish_reset(self.data, ids)
        else:
            for state in (self.backend.state_0, self.backend.state_1):
                self.mujoco_solver._update_newton_state(
                    self.backend.model, state, self.data, state_prev=state
                )
        mask = torch.zeros(self.num_envs + 1, dtype=torch.bool, device=self.device)
        mask[ids] = True
        self.backend.clear_solver_state(wp.from_torch(mask))
        # Recompute positions/sensors without advancing simulation time.
        self.mujoco_solver._mujoco_warp.forward(self.mujoco_solver.mjw_model, self.data)
        self.elapsed_steps[ids] = 0
        self._episode_return[ids] = 0
        self._episode_success[ids] = False
        state = self._state()
        if self.task_id in ("M1", "M2"):
            fresh = self._observation(self.task.observe(state))
        else:
            selected = self._observation(self.task.observe(state, env_ids=ids))
            if self._last_observation is None:
                fresh = {
                    key: value.new_empty((self.num_envs, value.shape[-1]))
                    for key, value in selected.items()
                }
            else:
                fresh = {
                    key: value.clone() for key, value in self._last_observation.items()
                }
            for key, value in selected.items():
                fresh[key][ids] = value
        self._last_observation = fresh
        return fresh, {}

    def advance_physics(self) -> None:
        """Advance one complete control interval with the configured substeps."""
        for _ in range(self.substeps):
            if self.task_id == "M3":
                # The passive layer detent is updated for each physical substep.
                state = self._state()
                face_force = self.task.face_detent(state["qpos"][:, -1])
                force = wp.to_torch(self.backend.control.joint_f)
                force.zero_()
                force[self.face_force_indices] = face_force
            self.backend.step()

    def step(
        self, actions: torch.Tensor
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if self.diagnostics and not bool(torch.isfinite(actions).all()):
            raise FloatingPointError("Non-finite policy action")
        actions = actions.clamp(-1.0, 1.0)
        if self.task_id in ("M1", "M2"):
            targets = self.task.action_targets(actions)
        else:
            targets = self.task.control_targets(actions)
        self.ctrl.copy_(targets)
        self.advance_physics()
        state = self._state()
        if self.diagnostics and not bool(
            torch.isfinite(
                state.qpos if isinstance(state, ShadowState) else state["qpos"]
            ).all()
        ):
            raise FloatingPointError("Non-finite physical state")
        if self.task_id in ("M1", "M2"):
            obs, reward, terminated, metrics = self.task.step(state, actions)
            self.elapsed_steps += 1
            truncated = self.elapsed_steps >= 500
        else:
            obs, reward, terminated, truncated, metrics = self.task.after_step(
                state, actions
            )
            self.elapsed_steps += 1
        self._episode_return += reward
        done = terminated | truncated
        result = self._observation(obs)
        self._last_observation = result
        info = {"metrics": {key: value.clone() for key, value in metrics.items()}}
        if "success" in metrics:
            # Task success is an event; evaluation consumes the completed episode.
            self._episode_success |= metrics["success"].bool()
            info["success"] = self._episode_success.clone()
        if bool(done.any()):
            ids = torch.where(done)[0]
            info["episode"] = {
                "r": self._episode_return[ids].clone(),
                "l": self.elapsed_steps[ids].clone(),
            }
            final_observation = {k: v.clone() for k, v in result.items()}
            reset_obs, _ = self.reset(options={"reset_ids": ids})
            for key in result:
                result[key][ids] = reset_obs[key][ids]
            info["final_observation"] = final_observation
            info["_final_observation"] = done
        self._last_observation = result
        return result, reward, terminated.bool(), truncated.bool(), info

    def close(self) -> None:
        if not self._closed:
            self.world.quit()
            self._closed = True
