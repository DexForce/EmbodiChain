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

"""Offline asset preparation and sequential Task Program batch collection.

Batches reuse one single-row environment and replace selected bodies through
the existing asset event. Initial and measured checks gate recording; one final
close drains the shared recorders after all batches.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import islice
from functools import partial
import json
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Any

from embodichain.data.asset_catalog import (
    AssetCandidate,
    AssetProvider,
    AssetQuery,
    PreparationContext,
    PreparedAsset,
)
from embodichain.lab.gym.envs.managers.cfg import EventCfg
from embodichain.lab.gym.envs.managers.event_manager import EventManager
from embodichain.lab.gym.envs.managers.events import replace_assets_from_group
from embodichain.lab.sim.motion.expansion import ValidationResult
from embodichain.lab.sim.scene.expansion import SceneVariant
from embodichain.lab.task_program.integrations.asset_materialization import (
    AssetAdaptationCfg,
    MaterializedAssetDeployment,
    load_asset_requirements_from_deployment,
    materialize_asset_deployment,
    validate_materialized_deployment,
)
from embodichain.lab.task_program.integrations.scene_expansion import (
    execute_scene_variant,
)
from embodichain.lab.task_program.integrations.simulation.scene_expansion import (
    SimulationSceneExpansionHost,
)
from embodichain.lab.task_program.semantics.asset_matching import resolve_asset_set
from embodichain.utils import configclass

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv

__all__ = [
    "AssetGeneralizationCfg",
    "AssetEpisodeOutcome",
    "AssetBatchOutcome",
    "prepare_asset_batches",
    "collect_asset_batches",
]


@configclass
class AssetGeneralizationCfg:
    """Finite preparation and execution budgets for rigid asset batches.

    Args:
        seed: Base uint32 seed; batch ``i`` uses ``seed + i``.
        max_batches: Maximum prepared or collected asset combinations.
        episodes_per_batch: Number of attempts on each fixed asset combination.
        candidates_per_role: Maximum candidates acquired once for each role.
        max_search_nodes: Matcher budget for partial assignment extensions.
        settle_steps: Physics substeps before initial scene validation.
        record_trajectory: Explicitly enable accepted-episode trajectory saving
            during collection with the default factory; preparation never records.

    Count limits are at most one million; zero is allowed only for seed and
    settle_steps. Mutated configurations are revalidated at operation entry.
    """

    seed: int = 0
    max_batches: int = 1
    episodes_per_batch: int = 1
    candidates_per_role: int = 32
    max_search_nodes: int = 1000
    settle_steps: int = 0
    record_trajectory: bool = False

    def __post_init__(self) -> None:
        if type(self.record_trajectory) is not bool:
            raise TypeError("record_trajectory must be a bool.")
        for name in (
            "seed",
            "max_batches",
            "episodes_per_batch",
            "candidates_per_role",
            "max_search_nodes",
            "settle_steps",
        ):
            value = getattr(self, name)
            lower = 0 if name in ("seed", "settle_steps") else 1
            upper = 2**32 - 1 if name == "seed" else 1_000_000
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer in [{lower}, {upper}].")
        if self.seed + self.max_batches - 1 >= 2**32:
            raise ValueError("Batch seeds must remain within uint32 range.")


@dataclass(frozen=True)
class AssetEpisodeOutcome:
    """Execution acceptance and reset submission for one episode.

    ``reset_submitted`` means the reset returned after the commit/discard request;
    it is not a durable persistence receipt. ``accepted`` is physical/program
    acceptance, independent of whether recording is configured.
    """

    episode_index: int
    accepted: bool
    reset_submitted: bool


@dataclass(frozen=True)
class AssetBatchOutcome:
    """Batch outcomes; close_finalized refers to the shared session's final close."""

    batch_index: int
    asset_set_id: str
    episodes: tuple[AssetEpisodeOutcome, ...]
    close_finalized: bool


def _checked_cfg(cfg: AssetGeneralizationCfg) -> AssetGeneralizationCfg:
    if not isinstance(cfg, AssetGeneralizationCfg):
        raise TypeError("cfg must be AssetGeneralizationCfg.")
    owned = cfg.copy()
    owned.validate()
    owned.__post_init__()
    return owned


def _write_report(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def prepare_asset_batches(
    task_config_path: str | Path,
    providers_by_role: Mapping[str, AssetProvider],
    *,
    cfg: AssetGeneralizationCfg,
    output_dir: str | Path,
    adaptations: AssetAdaptationCfg | None = None,
) -> tuple[MaterializedAssetDeployment, ...]:
    """Prepare a bounded candidate snapshot and materialize seeded task batches.

    Provider failures reject their candidate and remain in
    ``preparation.report.json``. If a role has no eligible assets, matching fails
    with its ordinary structured diagnostics. Candidate acquisition is performed
    once per role; every batch selects from the same prepared snapshot. No online
    provider is implicitly enabled or substituted.

    Args:
        task_config_path: Deployment whose integration declares asset roles.
        providers_by_role: Exactly one explicitly supplied provider per role.
        cfg: Candidate, batch and deterministic selection budgets.
        output_dir: Absent or empty directory for cache and batch deployments.
        adaptations: Explicit scene geometry adaptation passed to materialization.

    Returns:
        Fresh offline deployment artifacts, in batch order.
    """
    cfg = _checked_cfg(cfg)
    requirements = load_asset_requirements_from_deployment(task_config_path)
    if set(providers_by_role) != set(requirements.roles):
        raise ValueError("Providers must cover exactly the declared asset roles.")
    for role, provider in providers_by_role.items():
        if not isinstance(provider, AssetProvider):
            raise TypeError(f"Provider for {role!r} must implement AssetProvider.")
    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Preparation output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    context = PreparationContext(str(output / "cache"))
    report: dict[str, Any] = {
        "schema_version": 1,
        "roles": {},
        "batches": [],
        "status": "preparing",
    }
    report_path = output / "preparation.report.json"
    pools = {}
    try:
        for role, requirement in sorted(requirements.roles.items()):
            provider = providers_by_role[role]
            report["roles"][role] = {"prepared": 0, "rejections": []}
            diagnostics = report["roles"][role]["rejections"]
            query = requirement.query
            if requirements.category_aliases:
                # A provider need not know task-owned category aliases. Broaden
                # acquisition and enforce exact aliases in authoritative matching.
                query = AssetQuery(None, query.attributes, query.capabilities)
            pools[role] = []
            try:
                candidates = tuple(
                    islice(
                        provider.candidates(query, limit=cfg.candidates_per_role),
                        cfg.candidates_per_role + 1,
                    )
                )
                if len(candidates) > cfg.candidates_per_role:
                    raise ValueError(
                        "Provider returned more candidates than its requested limit."
                    )
            except Exception as error:
                diagnostics.append(
                    {
                        "stage": "candidates",
                        "error_type": type(error).__name__,
                        "detail": str(error),
                    }
                )
                continue
            for candidate in candidates:
                try:
                    if not isinstance(candidate, AssetCandidate):
                        raise TypeError(
                            "Provider.candidates must return AssetCandidate values."
                        )
                    prepared = provider.materialize(candidate, context=context)
                    if not isinstance(prepared, PreparedAsset):
                        raise TypeError(
                            "Provider.materialize must return PreparedAsset."
                        )
                    pools[role].append(prepared)
                except Exception as error:
                    diagnostics.append(
                        {
                            "stage": "materialize",
                            "candidate_id": getattr(candidate, "candidate_id", None),
                            "error_type": type(error).__name__,
                            "detail": str(error),
                        }
                    )
            report["roles"][role]["prepared"] = len(pools[role])
        deployments = []
        for batch_index in range(cfg.max_batches):
            resolved = resolve_asset_set(
                requirements,
                pools,
                seed=cfg.seed + batch_index,
                max_candidates_per_role=cfg.candidates_per_role,
                max_search_nodes=cfg.max_search_nodes,
            )
            deployment = materialize_asset_deployment(
                task_config_path,
                requirements,
                resolved,
                output_dir=output / f"batch_{batch_index:04d}",
                adaptations=adaptations,
            )
            deployments.append(deployment)
            report["batches"].append(
                {
                    "batch_index": batch_index,
                    "asset_set_id": resolved.set_id,
                    "deployment_path": str(deployment.deployment_path),
                }
            )
        report["status"] = "prepared"
        _write_report(report_path, report)
        return tuple(deployments)
    except BaseException as error:
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
        report["detail"] = str(error)
        if hasattr(error, "rejections"):
            report["matching_rejections"] = [asdict(item) for item in error.rejections]
        try:
            _write_report(report_path, report)
        except Exception as report_error:
            error.add_note(f"Preparation report failed: {report_error}")
        raise


def _create_environment(
    deployment: MaterializedAssetDeployment,
    *,
    seed: int,
    record_trajectory: bool = False,
) -> EmbodiedEnv:
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.gym.utils.registration import build_env
    from embodichain.utils.utility import load_config

    config = load_config(deployment.deployment_path)
    env_cfg = config_to_cfg(config, source_path=deployment.deployment_path)
    if env_cfg.num_envs != 1:
        raise ValueError(
            "Asset collection requires the deployment to configure num_envs=1."
        )
    env_cfg.seed = seed
    if record_trajectory:
        env_cfg.record_trajectory = True
        env_cfg.trajectory_auto_save = True
    # Every fresh recorder starts its own counters. Keep every invocation in a
    # distinct directory, including repeated collection of the same deployment.
    recordings = deployment.deployment_path.parent / "recordings"
    recordings.mkdir(parents=True, exist_ok=True)
    run_root = Path(mkdtemp(prefix="run_", dir=recordings))
    if env_cfg.record_trajectory and env_cfg.trajectory_auto_save:
        env_cfg.trajectory_save_dir = str(run_root / "trajectories")
    for index, recorder in enumerate(
        vars(env_cfg.dataset).values() if env_cfg.dataset is not None else ()
    ):
        if getattr(recorder, "mode", None) == "save":
            recorder.params["save_path"] = str(run_root / f"dataset_{index:03d}")
    from embodichain.lab.gym.envs.managers.record import record_camera_data

    for index, event in enumerate(
        vars(env_cfg.events).values() if env_cfg.events is not None else ()
    ):
        func = getattr(event, "func", None)
        if isinstance(func, type) and issubclass(func, record_camera_data):
            event.params["save_path"] = str(run_root / f"camera_{index:03d}")
    env = build_env(config["id"], env_cfg)
    env._asset_generalization_recording_root = str(run_root)
    return env


def _flush_cleanup() -> None:
    from embodichain.lab.sim import SimulationManager

    SimulationManager.flush_cleanup_queue()


def _collect_batch(
    env: EmbodiedEnv,
    deployment: MaterializedAssetDeployment,
    *,
    cfg: AssetGeneralizationCfg,
    batch_index: int,
    replacement_events: EventManager,
    initial_validator: Callable[[EmbodiedEnv], ValidationResult],
    measured_validator: Callable[[EmbodiedEnv], ValidationResult],
    report: dict[str, Any],
) -> None:
    target = getattr(env, "unwrapped", env)
    validate_materialized_deployment(deployment)
    manifest = json.loads(deployment.manifest_path.read_text(encoding="utf-8"))
    batch_seed = cfg.seed + batch_index
    replacement_events.get_functor_cfg("replace_assets").params[
        "deployment"
    ] = deployment
    replacement_events.set_seed(batch_seed)
    report["recording_root"] = getattr(
        target, "_asset_generalization_recording_root", None
    )
    # This reset consumes/discards old buffers before the event changes geometry.
    # Ordinary reset events run first; episode preparation then observes the new
    # objects. The next scene host owns a new topology and physical-state snapshot.
    target.reset(
        seed=batch_seed,
        options={
            "save_data": False,
            "scene_expansion_prepare": lambda: replacement_events.apply(
                "reset", env_ids=[0]
            ),
        },
    )
    report["replacement"] = deepcopy(target._asset_generalization_replacement)
    host = SimulationSceneExpansionHost(
        target,
        initial_validator=lambda: initial_validator(target),
        settle_steps=cfg.settle_steps,
    )
    for episode_index in range(cfg.episodes_per_batch):
        episode = {
            "episode_index": episode_index,
            "accepted": False,
            "reset_submitted": False,
        }
        report["episodes"].append(episode)
        variant = SceneVariant(
            parent_scene_id=deployment.asset_set_id,
            seed=batch_seed,
            ordinal=episode_index,
            pose_changes=(),
            metadata=(("asset_set_id", deployment.asset_set_id),),
        )
        result = execute_scene_variant(
            env,
            variant,
            prepare_scene=host,
            measured_validator=lambda: measured_validator(target),
            episode_index=episode_index,
        )
        episode["accepted"] = result.accepted
        episode["validation"] = result.to_metadata()
        target._demo_episode_metadata[0]["asset_generalization"] = {
            "schema_version": 1,
            "batch_index": batch_index,
            "episode_index": episode_index,
            "seed": batch_seed,
            "recording_root": report["recording_root"],
            "replacement": deepcopy(report["replacement"]),
            "manifest": deepcopy(manifest),
        }
        target.reset(
            options={
                "save_data": result.accepted,
                "commit_env_ids": [0] if result.accepted else [],
            }
        )
        episode["reset_submitted"] = True


def _collect_shared_environment(
    selected: tuple[MaterializedAssetDeployment, ...],
    *,
    cfg: AssetGeneralizationCfg,
    environment_factory: Callable[..., EmbodiedEnv],
    initial_validator: Callable[[EmbodiedEnv], ValidationResult],
    measured_validator: Callable[[EmbodiedEnv], ValidationResult],
    report: dict[str, Any],
) -> tuple[AssetBatchOutcome, ...]:
    batch_report = {
        "batch_index": 0,
        "asset_set_id": selected[0].asset_set_id,
        "episodes": [],
        "close_finalized": False,
    }
    report["batches"].append(batch_report)
    validate_materialized_deployment(selected[0])
    env = environment_factory(selected[0], seed=cfg.seed)
    target = getattr(env, "unwrapped", env)
    report["environment_constructions"] = 1
    original_error = None
    try:
        if target.num_envs != 1:
            raise ValueError("Asset collection requires B=1")
        # Use the ordinary EventManager and the existing replacement functor.
        # Its explicit batch dispatch leaves the environment's reset/interval
        # terms and stateful recorder instances intact.
        replacement_events = EventManager(
            {
                "replace_assets": EventCfg(
                    func=replace_assets_from_group,
                    mode="reset",
                    params={"deployment": None},
                )
            },
            target,
        )
        for index, deployment in enumerate(selected):
            if index:
                batch_report = {
                    "batch_index": index,
                    "asset_set_id": deployment.asset_set_id,
                    "episodes": [],
                    "close_finalized": False,
                }
                report["batches"].append(batch_report)
            _collect_batch(
                env,
                deployment,
                cfg=cfg,
                batch_index=index,
                replacement_events=replacement_events,
                initial_validator=initial_validator,
                measured_validator=measured_validator,
                report=batch_report,
            )
    except BaseException as error:
        original_error = error
        batch_report["error_type"] = type(error).__name__
        batch_report["detail"] = str(error)
        raise
    finally:
        try:
            target.close(exit_process=False)
            for item in report["batches"]:
                item["close_finalized"] = True
        except BaseException as close_error:
            batch_report["close_error"] = {
                "error_type": type(close_error).__name__,
                "detail": str(close_error),
            }
            if original_error is None:
                raise
            original_error.add_note(f"Environment close also failed: {close_error}")
    return tuple(
        AssetBatchOutcome(
            item["batch_index"],
            item["asset_set_id"],
            tuple(
                AssetEpisodeOutcome(
                    ep["episode_index"], ep["accepted"], ep["reset_submitted"]
                )
                for ep in item["episodes"]
            ),
            item["close_finalized"],
        )
        for item in report["batches"]
    )


def collect_asset_batches(
    deployments: Sequence[MaterializedAssetDeployment],
    *,
    cfg: AssetGeneralizationCfg,
    initial_validator: Callable[[EmbodiedEnv], ValidationResult],
    measured_validator: Callable[[EmbodiedEnv], ValidationResult],
    environment_factory: Callable[..., EmbodiedEnv] | None = None,
    report_path: str | Path | None = None,
) -> tuple[AssetBatchOutcome, ...]:
    """Collect sequential B=1 batches using canonical fail-closed scene execution.

    One environment and its recorders are retained across batches. Only selected
    bodies are replaced; every batch obtains fresh task bindings and a new scene
    acceptance snapshot. Cleanup is drained after the final close and successful
    helper-frame unwind; a failed native
    constructor or execution traceback must be released by its caller before
    deferred cleanup can safely be drained. Exceptions preserve their original
    identity when close or reporting also fails.

    Args:
        deployments: Offline materialized artifacts, bounded by max_batches.
        cfg: Execution count, seed and settling budgets.
        initial_validator: Required measured checks of the prepared scene.
        measured_validator: Required independent final task checks.
        environment_factory: Optional factory accepting (deployment, seed=...).
            Called once for the first deployment. Custom factories own recording
            configuration and directory isolation. The default creates one fresh
            recordings/run_* directory beside the first deployment; recorder
            counters and provenance distinguish the batches within that session.
        report_path: Optional JSON lifecycle report, also written on failure.

    Returns:
        Acceptance, reset-submission and close-finalization outcomes. These are
        lifecycle observations, not per-episode persistence receipts.
    """
    cfg = _checked_cfg(cfg)
    selected = tuple(islice(deployments, cfg.max_batches + 1))
    if not selected or len(selected) > cfg.max_batches:
        raise ValueError("Deployment count must be between one and max_batches.")
    if any(not isinstance(item, MaterializedAssetDeployment) for item in selected):
        raise TypeError("deployments must contain MaterializedAssetDeployment values.")
    if not callable(initial_validator) or not callable(measured_validator):
        raise TypeError("Both validators must be callable.")
    factory = (
        partial(_create_environment, record_trajectory=cfg.record_trajectory)
        if environment_factory is None
        else environment_factory
    )
    if not callable(factory):
        raise TypeError("environment_factory must be callable.")
    report: dict[str, Any] = {
        "schema_version": 1,
        "batches": [],
        "status": "collecting",
    }
    original_error: BaseException | None = None
    try:
        outcomes = _collect_shared_environment(
            selected,
            cfg=cfg,
            environment_factory=factory,
            initial_validator=initial_validator,
            measured_validator=measured_validator,
            report=report,
        )
        # No native-bearing helper frame remains on the successful path.
        _flush_cleanup()
        report["status"] = "completed"
        return outcomes
    except BaseException as error:
        original_error = error
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
        report["detail"] = str(error)
        raise
    finally:
        if report_path is not None:
            try:
                _write_report(Path(report_path), report)
            except BaseException as report_error:
                if original_error is None:
                    raise
                original_error.add_note(
                    f"Collection report also failed: {report_error}"
                )
