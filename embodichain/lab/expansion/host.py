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

"""Native World owner composing recipe, preparation and evidence adapters."""

from __future__ import annotations
from collections.abc import Mapping, Callable
from typing import Any
from contextlib import ExitStack
from pathlib import Path
import time
from embodichain.lab.expansion.configured import ExpansionRunCfg
from .contracts import (
    CandidateIdentity,
    ExpertEpisode,
    ValidationResult,
)

from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.gym.utils.registration import REGISTERED_ENVS
from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.scene import SceneView
from embodichain.lab.sim.scene.physics import (
    capture_physics_properties,
    capture_rigid_physics,
)
from embodichain.utils.utility import load_config
from .engine import PreparedEpisodeWorkItem, ExpansionWorkItem, ExpansionAssignment
from .recipes import ExpansionRecipe, OperatorCapability, BatchCompatibility
from .journal import RecipeAttempt, ExpansionJournal
from .session import ExpansionSession

from ._gym_utils import _hash, _observation_schema
from ._gym_context import GymHostContext, PreparedRow
from ._gym_recipe import GymRecipeBuilder
from ._gym_prepare import GymCohortPreparer
from ._gym_execution import GymCohortExecutor

__all__ = ["GymExpansionHost"]


class GymExpansionHost:
    """Native World owner composing recipe, preparation and measured-evidence components.

    Args:
        cfg: Resolved deployment and collection configuration.
        report: Successful preflight report pinning recipe semantics.
        root: Job artifact directory owned by the caller.
        journal: Open journal under the caller's exclusive job lock.
        source_artifacts: Content-verified source references to retain in evidence.
    """

    def __init__(
        self,
        cfg: ExpansionRunCfg,
        report: Mapping[str, object],
        root: Path,
        journal: ExpansionJournal,
        source_artifacts: Mapping[str, object],
    ) -> None:
        self._context = GymHostContext(
            cfg, report, root, journal, source_artifacts, cfg.batch_size
        )
        self._recipe_builder = GymRecipeBuilder(self._context)
        self._context.recipe = self._recipe_builder.recipe
        self._preparer = GymCohortPreparer(self._context)
        self._executor = GymCohortExecutor(self._context)
        try:
            self._open()
        except BaseException:
            self.close()
            raise

    def _open(self) -> None:
        cfg, report, root = self._context.cfg, self._context.report, self._context.root
        collision = None
        # Source artifacts and camera buffers belong to this run, never normal reset-time writers.
        path = Path(cfg.deployment)
        raw = load_config(path)
        environment_cfg = config_to_cfg(
            raw, source_path=path, task_program_path_override=cfg.program
        )
        environment_cfg.num_envs = environment_cfg.sim_cfg.num_envs = cfg.batch_size
        environment_cfg.seed = cfg.seed
        environment_cfg.dataset = None
        environment_cfg.init_rollout_buffer = False
        environment_cfg.record_trajectory = False
        environment_cfg.sim_cfg.headless = True
        if not cfg.capture_media:
            environment_cfg.enable_sensor = False
            environment_cfg.sim_cfg.render_cfg.renderer = "no-render"
        else:
            environment_cfg.enable_sensor = True
            if environment_cfg.sim_cfg.render_cfg.renderer == "no-render":
                environment_cfg.sim_cfg.render_cfg.renderer = "hybrid"
        from embodichain.lab.sim.profiler import ProfilerCfg

        environment_cfg.profiler = ProfilerCfg(
            warmup_steps=0, output_path=str(root / "profile.json")
        )
        appearance = None
        visual_registry = None
        if cfg.visual_profile_file is not None:
            from .visual import load_visual_profiles

            visual_registry = load_visual_profiles(cfg.visual_profile_file)
        started = time.monotonic()
        timings = {
            "prepare": 0.0,
            "execute": 0.0,
            "commit": 0.0,
            "render_variants": 0.0,
            "export": 0.0,
        }
        env = REGISTERED_ENVS[raw["id"]].make(cfg=environment_cfg)
        self._context.env = env
        timings["startup"] = time.monotonic() - started
        if getattr(
            env, "event_manager", None
        ) is not None and env.event_manager.active_functors.get("interval"):
            raise ValueError(
                "offline expansion requires fixed episode parameters; interval events are unsupported"
            )
        view = SceneView(env.sim, env_ids=tuple(range(cfg.batch_size)))
        managed = tuple(sorted(set(cfg.mass_scale) | set(cfg.friction)))
        baseline = capture_rigid_physics(view, managed) if managed else None
        auxiliary_baseline = (
            capture_physics_properties(view)
            if (
                cfg.center_of_mass
                or cfg.joint_stiffness_scale
                or cfg.joint_damping_scale
            )
            else None
        )
        if cfg.collision_profile is not None:
            from .collision import _MeasuredCollision

            collision = _MeasuredCollision(
                env, cfg.collision_profile, root / "collision"
            )
        self._context.collision = collision
        profile = None
        if cfg.source == "real2sim":
            from embodichain.lab.task_program.integrations.simulation.real2sim_collection import (
                load_real2sim_tabletop_profile,
            )

            profile = load_real2sim_tabletop_profile(cfg.source_profile)
        scene_source = None
        if cfg.scene_atlas is not None:
            from .scene import JointSceneSource

            scene_source = JointSceneSource(cfg.scene_atlas)
        motion_profile = None
        motion_recipes = ()
        if cfg.motion_profile is not None:
            from embodichain.lab.sim.motion.expansion.profile import (
                load_expansion_profile,
            )
            from embodichain.lab.sim.motion.expansion.combined import (
                enumerate_candidate_recipes,
            )

            motion_profile = load_expansion_profile(cfg.motion_profile)
            motion_recipes = enumerate_candidate_recipes(motion_profile)
        names = tuple(env.robot.joint_names[i] for i in env.active_joint_ids)
        limits = env.robot.get_qpos_limits()
        if limits.ndim == 3:
            limits = limits[0]
        limits = limits[list(env.active_joint_ids)].cpu()
        self._context.environment_cfg = environment_cfg
        self._context.env = env
        self._context.view = view
        self._context.managed = managed
        self._context.baseline = baseline
        self._context.auxiliary_baseline = auxiliary_baseline
        self._context.collision = collision
        self._context.profile = profile
        self._context.scene_source = scene_source
        self._context.motion_profile = motion_profile
        self._context.motion_recipes = motion_recipes
        self._context.names = names
        self._context.limits = limits
        self._context.visual_registry = visual_registry
        self._context.appearance = appearance
        self._context.timings = timings
        self._context.started = started

        self._context.compatibility = BatchCompatibility(
            robot=env.robot.uid,
            joint_names=tuple(env.robot.joint_names),
            topology_revision=report["semantic_id"],
            program_revision=_hash(environment_cfg.task_program.to_dict()),
            backend=env.sim.physics_backend,
            control_dt=env.step_dt,
            observation_revision=_hash(
                _observation_schema(env.single_observation_space)
            ),
            world_profile=_hash(
                visual_registry[cfg.visual_profile] if cfg.visual_profile else {}
            ),
            shared_resources=tuple(
                sorted(
                    (
                        visual_registry[cfg.visual_profile].get("materials", {})
                        if cfg.visual_profile
                        else {}
                    ).keys()
                )
            ),
        )
        self._context.supported_operations = {
            "physical_parameters": OperatorCapability(
                "physics", "prepare", "environment"
            ),
            "joint_scene": OperatorCapability(
                "scene", "prepare", "environment", sources=("task_program",)
            ),
            "visual": OperatorCapability("visual", "prepare", "world"),
            "motion": OperatorCapability("motion", "plan", "environment"),
        }

    def preflight(self, items: tuple[ExpansionWorkItem, ...]) -> None:
        """Reject foreign prepared work and resource mismatches without stepping."""
        for item in items:
            if (
                not isinstance(item, PreparedEpisodeWorkItem)
                or item.compatibility_key != self._context.compatibility.key
            ):
                raise ValueError("native host requires its prepared compatible recipes")
            if (
                item.env_id not in self._context.prepared
                or self._context.prepared[item.env_id][0] != item.candidate.identity
            ):
                raise ValueError("prepared candidate does not belong to this cohort")

    def validate_plan(self, item: ExpansionWorkItem) -> ValidationResult:
        """Return preparation evidence; motion is generated lazily during execution."""
        return self._context.preparation_checks[item.env_id]

    def on_deferred(self, item: PreparedEpisodeWorkItem) -> None:
        """Release an unstarted row for a new attempt with the same recipe."""
        self._context.journal.update(
            self._context.prepared[item.env_id][3],
            status="interrupted",
            executed_steps_lower_bound=0,
        )

    def on_acceptance(
        self, assignment: ExpansionAssignment, episode: ExpertEpisode, accepted: bool
    ) -> None:
        """Record measured acceptance without granting committed episode credit."""
        self._context.journal.update(
            self._context.prepared[assignment.env_id][3],
            status=(
                "accepted"
                if accepted
                else (
                    "coverage_rejected" if episode.validation.accepted else "rejected"
                )
            ),
            measured_accepted=episode.validation.accepted,
        )

    def close(self) -> None:
        """Release every owned resource even when an earlier cleanup operation fails."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        errors = []
        for name in ("appearance", "collision"):
            resource = getattr(self._context, name, None)
            setattr(self._context, name, None)
            if resource is not None:
                try:
                    resource.close()
                except BaseException as error:
                    errors.append(error)
        (self._context.services or {}).clear()
        (self._context.bridges or {}).clear()
        self._context.view = self._context.initial = self._context.initial_obs = (
            self._context.physics_before
        ) = None
        self._context.baseline = self._context.auxiliary_baseline = None
        env = self._context.env
        self._context.env = None
        if env is not None:
            try:
                env._active_task_program_bridge = None
                env.close(exit_process=False)
            except BaseException as error:
                errors.append(error)
        try:
            SimulationManager.flush_cleanup_queue()
        except BaseException as error:
            errors.append(error)
        if errors:
            fatal = next(
                (error for error in errors if not isinstance(error, Exception)), None
            )
            if fatal is not None:
                raise fatal
            raise RuntimeError(
                "native host cleanup failed: " + "; ".join(map(str, errors))
            ) from errors[0]

    @property
    def cfg(self) -> ExpansionRunCfg:
        """Return the host's cfg binding."""
        return self._context.cfg

    @property
    def report(self) -> Mapping[str, object]:
        """Return the host's report binding."""
        return self._context.report

    @property
    def root(self) -> Path:
        """Return the host's root binding."""
        return self._context.root

    @property
    def journal(self) -> ExpansionJournal:
        """Return the host's journal binding."""
        return self._context.journal

    @property
    def source_artifacts(self) -> Mapping[str, object]:
        """Return the host's source_artifacts binding."""
        return self._context.source_artifacts

    @property
    def capacity(self) -> int:
        """Return the physical environment capacity owned by this host."""
        return self._context.capacity

    @property
    def env(self) -> Any:
        """Expose the current native env for compatible host integrations."""
        return self._context.env

    @env.setter
    def env(self, value: Any) -> None:
        self._context.env = value

    @property
    def appearance(self) -> ExitStack | None:
        """Expose the current native appearance for compatible host integrations."""
        return self._context.appearance

    @appearance.setter
    def appearance(self, value: ExitStack | None) -> None:
        self._context.appearance = value

    @property
    def collision(self) -> Any:
        """Expose the current native collision for compatible host integrations."""
        return self._context.collision

    @collision.setter
    def collision(self, value: Any) -> None:
        self._context.collision = value

    @property
    def timings(self) -> dict[str, float]:
        """Expose the current native timings for compatible host integrations."""
        return self._context.timings

    @timings.setter
    def timings(self, value: dict[str, float]) -> None:
        self._context.timings = value

    @property
    def compatibility(self) -> BatchCompatibility | None:
        """Expose the current native compatibility for compatible host integrations."""
        return self._context.compatibility

    @compatibility.setter
    def compatibility(self, value: BatchCompatibility | None) -> None:
        self._context.compatibility = value

    @property
    def supported_operations(self) -> dict[str, OperatorCapability]:
        """Expose the current native supported_operations for compatible host integrations."""
        return self._context.supported_operations

    @supported_operations.setter
    def supported_operations(self, value: dict[str, OperatorCapability]) -> None:
        self._context.supported_operations = value

    @property
    def prepared(self) -> dict[int, PreparedRow]:
        """Expose the current native prepared for compatible host integrations."""
        return self._context.prepared

    @prepared.setter
    def prepared(self, value: dict[int, PreparedRow]) -> None:
        self._context.prepared = value

    @property
    def requests(self) -> dict[int, RecipeAttempt]:
        """Expose the current native requests for compatible host integrations."""
        return self._context.requests

    @requests.setter
    def requests(self, value: dict[int, RecipeAttempt]) -> None:
        self._context.requests = value

    @property
    def profile(self) -> Any:
        """Expose the current native profile for compatible host integrations."""
        return self._context.profile

    @profile.setter
    def profile(self, value: Any) -> None:
        self._context.profile = value

    def recipe(self, ordinal: int) -> ExpansionRecipe:
        """Sample a pinned recipe without mutating the World."""
        return self._recipe_builder.recipe(ordinal)

    def prepare_batch(
        self, requests: tuple[RecipeAttempt, ...], session: ExpansionSession
    ) -> tuple[PreparedEpisodeWorkItem, ...]:
        """Prepare all rows through the cohort-preparation component."""
        return self._preparer.prepare_batch(requests, session)

    def execute_batch(
        self,
        assigned: tuple[ExpansionAssignment, ...],
        *,
        on_started: Callable[[CandidateIdentity], None],
    ) -> tuple[ExpertEpisode | None, ...]:
        """Execute prepared candidates and return measured evidence."""
        return self._executor.execute_batch(assigned, on_started=on_started)
