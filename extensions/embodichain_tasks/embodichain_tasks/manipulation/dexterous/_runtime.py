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


def _named_index(
    model: Any, kind: str, name: str, *, default: int | None = None
) -> int:
    count = getattr(model, "n" + ("jnt" if kind == "joint" else kind))
    candidates = [
        i
        for i in range(count)
        if getattr(model, kind)(i).name == name
        or getattr(model, kind)(i).name.endswith("_" + name)
        or getattr(model, kind)(i).name.rsplit("/", 1)[-1] == name
        or (
            kind == "geom"
            and re.sub(r"_\d+$", "", getattr(model, kind)(i).name).split("/")[-1]
            == name
        )
    ]
    if not candidates and default is not None:
        return default
    if len(candidates) != 1:
        raise ValueError(f"Cannot resolve {kind} {name!r}: {candidates}")
    return candidates[0]


def rotate_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    xyz = q[..., 1:]
    t = 2 * torch.cross(xyz, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(xyz, t, dim=-1)


class DexSimManipulationEnv:
    """Expose real DexSim physics through EmbodiChain's vector-env interface.

    SceneBuilder loads the USD physics model. MJCF supplies task metadata and
    the import audit reference. DexSim owns simulation stepping; MuJoCo forward
    calls below update kinematics after explicit reset.
    """

    def __init__(
        self,
        task_id: str,
        num_envs: int = 32,
        device: str = "cuda:0",
        seed: int | None = None,
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
        seed = torch.initial_seed() if seed is None else seed
        if task_id in ("M1", "M2"):
            self.task = ShadowTask.from_assets(
                Path(get_data_path("ShadowHandPen/shadow_pen.xml")).parent,
                num_envs,
                variant=task_id,
                device=device,
            )
            self._reset_generator = torch.Generator(device=self.device).manual_seed(
                seed
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
            self._reset_generator = self.task.generator
            self.ctrl_dt, self.sim_dt = 0.05, (0.005 if task_id == "M3" else 0.01)
            self.observation_dim = {"M3": 65, "M4": 57, "M5": 32}[task_id]
            self.critic_dim = {"M3": 65, "M4": 128, "M5": 105}[task_id]
            self.action_dim = 16
        self.source = mujoco.MjModel.from_xml_path(str(xml))
        self.source.opt.timestep = self.sim_dt
        if task_id in ("M1", "M2", "M3"):
            self.source.opt.ccd_iterations = 10
        self.xml = xml
        self.usd = xml.parent / "usd" / f"{xml.stem}.usdc"
        if not self.usd.is_file():
            raise FileNotFoundError(
                f"The manipulation asset lacks its USD model: {self.usd}"
            )
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
            # Face-turn contacts and joint constraints exceed 512 rows in
            # large batches. Keep headroom for the 256-contact allocation.
            "njmax": 2048 if task_id == "M3" else 512,
            "nconmax": 256,
            "update_data_interval": 0,
        }
        self.backend_option_overrides = {}
        if solver_iterations is not None:
            selected_iterations = solver_iterations
            for name, value in zip(
                ("iterations", "ls_iterations"), selected_iterations
            ):
                self.backend_option_overrides[name] = {
                    "source": solver_options[name],
                    "configured": value,
                    "reason": "Explicit benchmark settings",
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
        builder = dexsim.spawn.SceneBuilder(self.world)
        builder.add_usd("asset", str(self.usd))
        if solver_kind == "dexuni":
            from ._dexuni_adapter import _mujoco_articulation_ids

            self.backend.cfg.solver_cfg.mujoco_articulations = _mujoco_articulation_ids(
                builder.scene_desc, num_envs
            )
        self._explicit_collision_filtering = bool(
            builder.scene_desc.newton
            and builder.scene_desc.newton.use_explicit_collision_filtering
        )
        self._articulation_names = tuple(
            item.name for item in builder.scene_desc.articulations
        )
        builder.replicate(num_envs, spacing=(0.0, 0.0, 0.0))
        self.scene = builder.finalize()
        self.solver = self.backend.solver
        self.mujoco_solver = self.backend.mujoco_solver
        self.dexuni_adapter = None
        if solver_kind == "dexuni":
            from ._dexuni_adapter import DexUniStateAdapter

            self.dexuni_adapter = DexUniStateAdapter(self.backend)
        self.imported = self.mujoco_solver.mj_model
        self.data = self.mujoco_solver.mjw_data
        self.option_import_corrections = {}
        # World.dt owns the step size; Newton's host MjModel keeps the compiler
        # default until explicitly synchronized for the task model audit.
        self.imported.opt.timestep = self.sim_dt
        wp.to_torch(self.mujoco_solver.mjw_model.opt.timestep).fill_(self.sim_dt)
        self._make_mapping()
        model = self.backend.model
        joint_world = wp.to_torch(model.joint_world).long()
        self._reset_field_worlds = {
            "joint_q": torch.repeat_interleave(
                joint_world, torch.diff(wp.to_torch(model.joint_q_start)).long()
            ),
            "joint_qd": torch.repeat_interleave(
                joint_world, torch.diff(wp.to_torch(model.joint_qd_start)).long()
            ),
            "body_q": wp.to_torch(model.body_world).long(),
            "body_qd": wp.to_torch(model.body_world).long(),
        }
        self.reset(seed=seed)

    @property
    def unwrapped(self) -> DexSimManipulationEnv:
        return self

    def _make_mapping(self) -> None:
        from ._audit_model import snapshot_execution_model

        derived_cache_report = {}
        source, imported = self.source, snapshot_execution_model(
            self.mujoco_solver, derived_cache_report=derived_cache_report
        )
        for name in ("nq", "nv", "nu", "ntendon"):
            if getattr(source, name) != getattr(imported, name):
                raise ValueError(
                    f"MJCF {name} changed: {getattr(source, name)} -> {getattr(imported, name)}"
                )
        qmap, vmap = [], []
        joint_map = {}
        for j in range(source.njnt):
            name = source.joint(j).name
            if name and source.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
                target = _named_index(imported, "joint", name)
            else:
                body = _named_index(
                    imported, "body", source.body(int(source.jnt_bodyid[j])).name
                )
                candidates = np.flatnonzero(
                    (imported.jnt_bodyid == body)
                    & (imported.jnt_type == source.jnt_type[j])
                )
                if len(candidates) != 1:
                    raise ValueError("Root joint cannot be resolved unambiguously")
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
        self.ctrl = torch.zeros((self.num_envs, source.nu), device=self.device)
        self._bind_controls()
        self.model_audit = {
            "solver_derived_cache_variation": derived_cache_report,
            "source": str(self.xml),
            "usd_source": str(self.usd),
            "parameter_source": "Active solver device parameters at initialization.",
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
            source.body(i).name: _named_index(
                imported,
                "body",
                source.body(i).name,
                default=(
                    0
                    if source.body_weldid[i] == 0 and source.body_mocapid[i] < 0
                    else None
                ),
            )
            for i in range(1, source.nbody)
            if source.body(i).name
        }
        self.model_audit["property_import_corrections"] = []
        from ._audit_model import audit_model

        audit_source = source
        if self.backend_option_overrides:
            audit_source = copy.copy(source)
            for name, item in self.backend_option_overrides.items():
                setattr(audit_source.opt, name, item["configured"])
        self.model_audit["physical_model"] = audit_model(
            audit_source,
            imported,
            joint_map,
            amap,
            self.body_map,
            explicit_collision_filtering=self._explicit_collision_filtering,
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

    def _bind_controls(self) -> None:
        """Bind source actuator order to homogeneous public articulation batches."""
        self._control_batches = []
        assigned = set()
        faces = []
        for name in self._articulation_names:
            handles = [
                self.scene.get_articulation(f"env_{index}/{name}")
                for index in range(self.num_envs)
            ]
            batch = self.scene.create_articulation_batch(handles)
            layouts = batch.joint_layouts_per_articulation[0]
            offsets, actuators = [], []
            for layout in layouts:
                matches = [
                    aid
                    for aid in range(self.source.nu)
                    if self.source.joint(int(self.source.actuator_trnid[aid, 0])).name
                    == layout.name
                ]
                if matches:
                    if (
                        len(matches) != 1
                        or layout.dof_count != 1
                        or matches[0] in assigned
                    ):
                        raise ValueError(
                            f"Ambiguous USD actuator mapping: {layout.name}"
                        )
                    assigned.add(matches[0])
                    offsets.append(layout.dof_start)
                    actuators.append(matches[0])
                if layout.name == "face_turn":
                    if layout.dof_count != 1:
                        raise ValueError("The passive cube face must have one DOF")
                    faces.append((batch, layout.dof_start))
            if offsets:
                self._control_batches.append(
                    (
                        batch,
                        torch.tensor(offsets, device=self.device),
                        torch.tensor(actuators, device=self.device),
                        torch.zeros(
                            (self.num_envs, batch.dof_width), device=self.device
                        ),
                    )
                )
        if assigned != set(range(self.source.nu)):
            raise ValueError(
                f"USD actuators are missing: {set(range(self.source.nu)) - assigned}"
            )
        if self.task_id == "M3":
            if len(faces) != 1:
                raise ValueError(f"Expected one passive cube face, got {len(faces)}")
            self._face_batch, self._face_dof = faces[0]

    def _publish_controls(self) -> None:
        for batch, offsets, actuators, commands in self._control_batches:
            commands[:, offsets] = self.ctrl[:, actuators]
            batch.apply_joint_target_position(commands)

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
            self._reset_generator.manual_seed(seed)
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if not options
            else options.get("reset_ids")
        )
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        if self.task_id in ("M1", "M2"):
            qpos, qvel, ctrl = self.task.reset(ids, generator=self._reset_generator)
        else:
            initial = self.task.reset(ids)
            qpos, qvel, ctrl = initial["qpos"], initial["qvel"], initial["ctrl"]
        wp.to_torch(self.data.qpos)[ids[:, None], self.qmap[None, :]] = qpos
        wp.to_torch(self.data.qvel)[ids[:, None], self.vmap[None, :]] = qvel
        self.ctrl[ids] = ctrl
        self._publish_controls()
        # Transfer explicitly set coordinates to both DexSim-owned buffers.
        if self.dexuni_adapter is not None:
            self.dexuni_adapter.publish_reset(self.data, ids)
        else:
            for state in (self.backend.state_0, self.backend.state_1):
                retained = []
                for name, worlds in self._reset_field_worlds.items():
                    unselected = ~torch.isin(worlds, ids)
                    values = wp.to_torch(getattr(state, name))
                    retained.append((values, unselected, values[unselected].clone()))
                self.mujoco_solver._update_newton_state(
                    self.backend.model, state, self.data, state_prev=state
                )
                for values, unselected, previous in retained:
                    values[unselected] = previous
        mask = torch.zeros(self.num_envs + 1, dtype=torch.bool, device=self.device)
        mask[ids] = True
        self.backend.clear_solver_state(wp.from_torch(mask))
        # Solver reset clears native controls; restore the selected reset targets
        # before computing their initial actuator forces and observations.
        wp.to_torch(self.data.ctrl)[ids[:, None], self.amap[None, :]] = ctrl
        # Forward updates derived data without advancing time or warm-start state.
        self.mujoco_solver._mujoco_warp.forward(self.mujoco_solver.mjw_model, self.data)
        self.elapsed_steps[ids] = 0
        self._episode_return[ids] = 0
        self._episode_success[ids] = False
        state = self._state()
        if self.task_id in ("M1", "M2"):
            selected = {
                key: value[ids]
                for key, value in self._observation(self.task.observe(state)).items()
            }
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
        self._publish_controls()
        for _ in range(self.substeps):
            if self.task_id == "M3":
                # The passive layer detent is updated for each physical substep.
                state = self._state()
                face_force = self.task.face_detent(state["qpos"][:, -1])
                self._face_batch.apply_joint_force(
                    face_force[:, None], dof_ids=[self._face_dof]
                )
            self.world.update(self.sim_dt)

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
            truncated = (self.elapsed_steps >= 500) & ~terminated
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
            self.scene.close()
            self.world.quit()
            self._closed = True
