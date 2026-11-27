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

"""Declarative scene/motion jobs composed through the native workload engine."""

from __future__ import annotations
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, TYPE_CHECKING
import hashlib
import json
import os
from embodichain.utils.utility import load_config as load_mapping
from embodichain.data_pipeline.recording.media import (
    _digest as file_digest,
    _atomic_json as atomic_json,
)
from ._task_program_config import load_task_program_expansion, load_expansion_viewer

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnvCfg
    from embodichain.lab.sim.scene.feasibility import JointSceneCandidate
    from embodichain.lab.task_program.runtime.expansion import (
        TaskProgramNominalSourceFactory,
    )
    from embodichain.lab.task_program.integrations.invocation_policy import (
        InvocationPolicyBinding,
    )
    from .host_extensions import ExpansionRowContext, ExpansionValidator

    from .program import ProgramBatchResult

__all__ = [
    "load_task_program_expansion",
    "load_expansion_viewer",
    "run_task_program_expansion",
    "resolve_task_program_expansion_bindings",
    "derive_grasp_expansion_deployment",
]


def _load_deployment(path: Path) -> EmbodiedEnvCfg:
    from embodichain.lab.gym.utils.registration import (
        discover_task_packages,
        execute_init_hooks,
    )
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg

    discover_task_packages()
    execute_init_hooks()
    return config_to_cfg(load_mapping(path), source_path=path)


from ._pose_sampling import _BoundedPoseSource, _AdaptivePoseSource


def _accepted_local_poses(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, list[float]]:
    """Express each recorded grasp in its own measured reference frame."""
    import numpy as np

    poses = {}
    for record in records:
        for sample in record.get("samples", {}).values():
            if (
                sample.get("pose_frame") == "world"
                and sample.get("success") == [True]
                and sample.get("reference_poses") is not None
                and sample.get("selected_poses") is not None
                and sample.get("key")
            ):
                local = np.linalg.solve(
                    sample["reference_poses"][0], sample["selected_poses"][0]
                )
                poses[sample["key"]] = local.reshape(16).tolist()
    return poses


class _AffordanceBranchSchedule:
    """Select non-nominal branches using measured outcomes from one workload."""

    def __init__(self, count: int, exploration_fraction: float) -> None:
        self.count = count - 1  # Branch zero is the nominal grasp.
        self.exploration_fraction = exploration_fraction
        self.assignments: dict[int, int] = {}
        self.allocations: Counter[int] = Counter()
        self.successes: Counter[int] = Counter()
        self.failures: Counter[int] = Counter()
        self.verified_segments: Counter[int] = Counter()
        self.finished: set[int] = set()
        self.exemplars: dict[int, dict] = {}
        self.preferences: dict[int, dict] = {}

    def select(self, ordinal: int) -> int:
        """Freeze one choice per recipe, including concurrently admitted rows."""
        if ordinal in self.assignments:
            return self.assignments[ordinal]
        if len(self.allocations) < self.count:
            branch = len(self.allocations) + 1
        else:
            n = len(self.assignments) - self.count
            explore = int((n + 1) * self.exploration_fraction) > int(
                n * self.exploration_fraction
            )

            compare_progress = not self.successes and len(self.failures) == self.count

            def priority(branch: int) -> tuple[int, float, int, int]:
                wins = self.successes[branch]
                # Beta(1, 1) prior. Only observed physical outcomes change it.
                rate = (1 + wins) / (2 + wins + self.failures[branch])
                return (
                    (
                        self.verified_segments[branch]
                        if compare_progress and not explore
                        else 0
                    ),
                    rate if self.failures and not explore else 0.0,
                    -self.allocations[branch],
                    -branch,
                )

            branch = max(self.allocations, key=priority)
            if not explore and self.failures and branch in self.exemplars:
                self.preferences[ordinal] = self.exemplars[branch]
        self.assignments[ordinal] = branch
        self.allocations[branch] += 1
        return branch

    def observe(self, ordinal: int, accepted: bool, result: ProgramBatchResult) -> None:
        """Credit an applied branch once, independently of dataset admission."""
        if ordinal not in self.finished:
            branch = self.assignments[ordinal]
            counts = self.successes if accepted else self.failures
            counts[branch] += 1
            verified_segments = sum(
                segment.get("success") is True for segment in result.segments[0]
            )
            self.verified_segments[branch] = max(
                self.verified_segments[branch], verified_segments
            )
            if accepted:
                poses = _accepted_local_poses(result.expansion_records[0])
                if poses:
                    self.exemplars[branch] = {"ordinal": ordinal, "local_poses": poses}
            self.finished.add(ordinal)


def _task_expansion_extensions(
    *,
    options,
    operator,
    seed,
    report,
    nominal_source_factory=None,
    invocation_policy_bindings=(),
):
    """Declare task proposals; upstream owns collection and persistence."""
    from .host_extensions import GymExpansionHostExtensions

    pose_source = None
    affordance = options["trajectory"].get("affordance", {})
    branches = None
    if (
        operator == "affordance"
        and affordance.get("branch_selection") == "outcome_guided"
    ):
        branches = _AffordanceBranchSchedule(
            affordance["branches"],
            options.get("scheduling", {}).get("exploration_fraction", 0.25),
        )

    def source(env):
        nonlocal pose_source
        from ._scene_first_sampling import _LayoutPoseSource

        revision = report["semantic_id"]
        if "layout" in options.get("scene_sampling", {}):
            pose_source = _LayoutPoseSource(env, options, seed=seed, revision=revision)
        elif "scene_sampling" in options:
            pose_source = _AdaptivePoseSource(
                env, options, seed=seed, revision=revision
            )
        else:
            pose_source = _BoundedPoseSource(
                env, options["scene"], seed=seed, revision=revision
            )
        return pose_source

    def feedback(env, source, path):
        source.output = path
        source.checkpoint()
        return source

    def recipe_transform(recipe):
        operations = []
        for operation in recipe.operations:
            payload = json.loads(operation.payload_json)
            if operation.name == "motion":
                payload.update(operator=operator, options=options["trajectory"])
                if branches is not None:
                    payload["affordance_branch"] = branches.select(
                        recipe.context.ordinal
                    )
                    if (
                        prior := branches.preferences.get(recipe.context.ordinal)
                    ) is not None:
                        payload["accepted_grasps"] = prior
            elif operation.name == "joint_scene":
                candidate = pose_source.candidate(recipe.context.ordinal)
                payload["pose_changes"] = {
                    change.entity_uid: change.arena_pose
                    for change in candidate.scene.pose_changes
                }
            operations.append(replace(operation, payload_json=json.dumps(payload)))
        return replace(recipe, operations=tuple(operations))

    def plans(env, row):
        from embodichain.lab.task_program.integrations.expansion import (
            TaskProgramSameGridExpansionFactory,
        )

        payload = {}
        if branches is not None:
            motion = next(op for op in row.recipe.operations if op.name == "motion")
            payload = json.loads(motion.payload_json)
        return TaskProgramSameGridExpansionFactory.from_native_plan(
            operator,
            options["trajectory"],
            seed=seed + row.ordinal,
            scene_case=row.scene_case,
            revision=report["recipe_revision"],
            output=row.directory / "plans",
            episode_id=row.ordinal,
            affordance_branch=payload.get("affordance_branch"),
            preferred_local_poses=tuple(
                (key, tuple(pose))
                for key, pose in payload.get("accepted_grasps", {})
                .get("local_poses", {})
                .items()
            ),
        )

    return GymExpansionHostExtensions(
        scene_source_factory=source,
        feedback_factory=feedback if "scene_sampling" in options else None,
        recipe_transform=recipe_transform,
        plan_transform_factory=plans if operator != "object_pose" else None,
        nominal_source_factory=nominal_source_factory,
        invocation_policy_bindings=invocation_policy_bindings,
        require_plan_transform=operator != "object_pose",
        plan_outcome=None if branches is None else branches.observe,
    )


def _prepare_deployment(deployment: Path, output: Path) -> Path:
    """Snapshot configuration without opening a reference simulation World."""
    from embodichain.utils.config_paths import resolve_config_path
    import yaml

    original = load_mapping(deployment)

    def paths(value):
        if isinstance(value, dict):
            return {key: paths(item) for key, item in value.items()}
        if isinstance(value, list):
            return [paths(item) for item in value]
        if isinstance(value, str) and value.endswith((".yaml", ".yml", ".json")):
            return str(resolve_config_path(value, base_dir=deployment.parent).resolve())
        return value

    snapshot = paths(deepcopy(original))
    owner = "environment" if "environment" in snapshot else "scene"
    if "component" in snapshot[owner]:
        from embodichain.lab.gym.utils._component_composition import (
            _resolve_local_resource_paths,
        )

        component_path = Path(snapshot[owner]["component"])
        physical = load_mapping(component_path)
        _resolve_local_resource_paths(physical, base_dir=component_path.parent)
        snapshot[owner] = {"component": str((output / "environment.yaml").resolve())}
    else:
        from embodichain.lab.gym.utils.composition import resolve_gym_components

        physical = resolve_gym_components(
            {owner: snapshot[owner]}, base_dir=deployment.parent
        ).config
    (output / "environment.yaml").write_text(yaml.safe_dump(physical, sort_keys=False))
    policy_path = snapshot.get("task_program", {}).get("execution_policy")
    if policy_path is not None:
        from embodichain.lab.task_program.integrations._configured_composition import (
            _resolve_execution_policy,
        )

        _, policy = _resolve_execution_policy(policy_path, base_dir=deployment.parent)
        policy = deepcopy(policy)
        policy.setdefault("tracking", {"kind": "timed"}).setdefault(
            "boundary_stop", True
        )
        policy["tracking"].setdefault(
            "boundary_reference_bridge", policy["tracking"]["boundary_stop"]
        )
        boundary_policy = output / "execution_policy.yaml"
        boundary_policy.write_text(yaml.safe_dump(policy, sort_keys=False))
        snapshot["task_program"]["execution_policy"] = str(boundary_policy.resolve())
    snapshot["id"] += "-Expansion-v1"
    path = output / "deployment.json"
    atomic_json(path, snapshot)
    atomic_json(
        output / "source.json",
        {
            "deployment": str(deployment),
            "sha256": file_digest(deployment),
            "config": original,
        },
    )
    return path


def derive_grasp_expansion_deployment(
    deployment: str | Path, output: str | Path, settings: Mapping[str, Any]
) -> Path:
    """Snapshot a native grasp policy without mutating the source deployment.

    Args:
        deployment: Native deployment with integration and embodiment references.
        output: Fresh directory for derived component files.
        settings: Validated affordance action_options and grasp_generators.

    Returns:
        Derived deployment path, usable by native configuration preflight or
        ordinary task execution. Physical task criteria remain unchanged.
    """
    deployment, output = Path(deployment), Path(output)
    from embodichain.utils.config_paths import resolve_config_path
    import yaml

    snapshot = load_mapping(deployment)
    integration_path = resolve_config_path(
        snapshot["task_program"]["integration"], base_dir=deployment.parent
    )
    integration = load_mapping(integration_path)
    profile = integration["profile"]
    presets = profile.get("skill_options", profile.get("action_options", {}))
    for name, changes in settings["action_options"].items():
        if name not in presets or "kind" in changes:
            raise ValueError(
                "Affordance options must retain existing skill preset kinds"
            )
        for key, value in changes.items():
            if value is None:
                presets[name].pop(key, None)
            else:
                presets[name][key] = value
    output.mkdir(parents=True)
    integration_path = output / "integration.yaml"
    integration_path.write_text(yaml.safe_dump(integration, sort_keys=False))
    snapshot["task_program"]["integration"] = str(integration_path.resolve())
    generator_overrides = settings.get("grasp_generators", {})
    if generator_overrides:
        embodiment_path = resolve_config_path(
            snapshot["embodiment"]["component"], base_dir=deployment.parent
        )
        embodiment = load_mapping(embodiment_path)
        generators = (
            embodiment["skill_profile"]
            .setdefault("runtime_services", {})
            .setdefault("grasp_pose_generators", {})
        )
        for name, changes in generator_overrides.items():
            changes = dict(changes)
            if "root_from_contact_frame" in changes:
                # Bound physical data belongs to the actual gripper declaration,
                # not its generator policy. Keep both native views consistent.
                bound = [
                    item
                    for item in embodiment.get("grippers", {}).values()
                    if item.get("control_part") == name
                ]
                if len(bound) != 1:
                    raise ValueError(
                        "A contact frame override requires one bound gripper"
                    )
                bound[0]["root_from_contact_frame"] = changes.pop(
                    "root_from_contact_frame"
                )
            if name not in generators:
                if not {"kind", "gripper"} <= changes.keys():
                    raise ValueError(f"Unknown embodiment grasp generator: {name}")
                generators[name] = {}
            generators[name].update(changes)
        embodiment_path = output / "embodiment.yaml"
        embodiment_path.write_text(yaml.safe_dump(embodiment, sort_keys=False))
        snapshot["embodiment"]["component"] = str(embodiment_path.resolve())
    snapshot["id"] += "-Affordance-v1"
    path = output / "deployment.json"
    atomic_json(path, snapshot)
    return path


def _run_operators(
    deployment,
    output,
    options,
    selected,
    *,
    seed,
    sources,
    viewer,
    runtime_versions,
    validator_factory=None,
    nominal_source_factory=None,
    invocation_policy_bindings=(),
):
    """Call the public upstream collector without operator processes."""
    from embodichain.lab.expansion import (
        ExpansionRunCfg,
        GymExpansionHostExtensions,
        GymExpansionWorkload,
        MeasuredExpansionValidator,
        WorkloadExecutionCfg,
        run_gym_expansion_workloads,
    )
    from embodichain.utils.nms import pose_nms_scope
    import torch

    settings = options["engine"]
    versions = dict(runtime_versions)
    workloads = []
    for name in selected:
        operator_deployment = deployment
        artifacts = dict(sources)
        affordance = options["trajectory"].get("affordance", {})
        if name == "affordance" and (
            affordance["action_options"] or affordance.get("grasp_generators")
        ):
            operator_deployment = derive_grasp_expansion_deployment(
                deployment, output / "inputs" / "affordance", affordance
            )
            for path in operator_deployment.parent.iterdir():
                if path.is_file():
                    artifacts[f"affordance_{path.name}"] = {
                        "path": str(path),
                        "sha256": file_digest(path),
                    }
        _load_deployment(operator_deployment)
        identity = hashlib.sha256(
            json.dumps(
                {
                    "options": options,
                    "operator": name,
                    "seed": seed,
                    "source_hashes": sorted(
                        value["sha256"] for value in artifacts.values()
                    ),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        report = {
            "job_id": identity,
            "semantic_id": identity,
            "execution_id": identity,
            "recipe_revision": identity,
            "input_files": {"deployment": file_digest(operator_deployment)},
            "operator": name,
            "adapter": "embodichain.task-program-workloads-v1",
            "runtime_versions": versions,
            "viewer": viewer,
        }
        cfg = ExpansionRunCfg(
            deployment=str(operator_deployment),
            output_dir=str((output / "dataset" / name).resolve()),
            batch_size=settings["batch_size"],
            target_episodes=settings["episodes_per_operator"],
            max_attempts=settings["max_attempts"],
            max_steps=settings["max_steps"],
            max_wall_time_s=settings["max_wall_time_s"],
            settling_steps=settings["settling_steps"],
            capture_media=settings["capture_media"],
            seed=seed,
        )
        workloads.append(
            GymExpansionWorkload(
                name,
                cfg,
                report,
                artifacts,
                _task_expansion_extensions(
                    options=options,
                    operator=name,
                    seed=seed,
                    report=report,
                    nominal_source_factory=nominal_source_factory,
                    invocation_policy_bindings=invocation_policy_bindings,
                ),
            )
        )
    renderer = settings["collection_renderer"]
    if renderer == "auto":
        renderer = viewer["renderer"] if settings["capture_media"] else None
    common = GymExpansionHostExtensions(
        collection_renderer=renderer,
        validator_factory=validator_factory
        or (
            lambda env, path: MeasuredExpansionValidator(
                env,
                options["validation"],
                path,
                backend=(
                    "native_contacts"
                    if settings["collision_backend"] == "native_contacts"
                    else "geometry"
                ),
                device=settings["collision_device"],
            )
        ),
    )
    execution = WorkloadExecutionCfg(
        workload_count=min(settings["workload_count"], len(selected)),
        batch_size=settings["batch_size"],
        max_steps=settings["max_steps"],
        settling_steps=settings["settling_steps"],
        seed=seed,
        reuse_idle_rows=settings.get("reuse_idle_rows", False),
        nominal_seed=settings.get("nominal_seed"),
    )
    chunks = settings["grasp_nms_chunk_size"] if torch.cuda.is_available() else 2048
    with pose_nms_scope(chunk_size=chunks):
        result = run_gym_expansion_workloads(
            workloads, execution, extensions=common, output_dir=output
        )
    atomic_json(output / "workload-execution.json", result)
    return result["workloads"]


def resolve_task_program_expansion_bindings(
    deployment: str | Path,
    *,
    source_artifacts: Mapping[str, object] | None = None,
    invocation_policy_bindings: tuple[InvocationPolicyBinding, ...] = (),
) -> tuple[tuple[InvocationPolicyBinding, ...], dict[str, object]]:
    """Resolve checked per-invocation policies and their provenance for a workload.

    Args:
        deployment: Original deployment before deriving any expansion resources.
        source_artifacts: Existing workload evidence, copied before adding the
            four fitted artifacts.
        invocation_policy_bindings: Optional explicit bindings that must agree
            with a sealed deployment.

    Returns:
        Per-invocation policies and the source artifacts for the collector.
    """
    deployment = Path(deployment).resolve(strict=True)
    sources = dict(source_artifacts or {})
    if (deployment.parent / "invocation_bindings.json").is_file():
        from embodichain.toolkits.skill_fitting.task_program import (
            load_invocation_bindings,
        )

        sealed = load_invocation_bindings(deployment)
        if invocation_policy_bindings and invocation_policy_bindings != sealed:
            raise ValueError(
                "Explicit invocation policies differ from the sealed deployment."
            )
        invocation_policy_bindings = sealed
        for name in (
            "manifest.json",
            "compile_report.json",
            "source_map.json",
            "invocation_bindings.json",
        ):
            path = deployment.parent / name
            sources[f"fitted_{name}"] = {"path": str(path), "sha256": file_digest(path)}
    return invocation_policy_bindings, sources


def run_task_program_expansion(
    deployment: str | Path,
    configuration: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 0,
    operators: Sequence[str] | None = None,
    source_artifacts: Mapping[str, object] | None = None,
    runtime_versions: Mapping[str, object] | None = None,
    viewer: Mapping[str, object] | None = None,
    preview_output: str | Path | None = None,
    validator_factory: Callable[[Any, Path], ExpansionValidator] | None = None,
    nominal_source_factory: (
        Callable[[Any, ExpansionRowContext], TaskProgramNominalSourceFactory] | None
    ) = None,
    invocation_policy_bindings: tuple[InvocationPolicyBinding, ...] = (),
) -> Mapping[str, object]:
    """Collect declared scene/motion operators through the single-World engine.

    Args:
        deployment: Native Gym deployment; task discovery uses installed providers.
        configuration: Closed YAML/JSON scene, trajectory and measured criteria.
        output_dir: Fresh collection directory; aggregate results are saved here.
        seed: Candidate seed, independent of physical row allocation.
        operators: Optional enabled subset in its desired group order.
        source_artifacts: Caller-owned hashed inputs retained in native evidence.
        runtime_versions: Caller provenance; package versions are read when absent.
        viewer: Headless/renderer declarations. A non-headless job opens the native
            read-only preview after collection, qualification and export.
        preview_output: Optional service status/log directory outside saved data.
        validator_factory: Optional task-owned measured validation port.
        nominal_source_factory: Optional independent factory for each attempted row.
        invocation_policy_bindings: Named presets bound to the exact program calls.

    Returns:
        Owned operator quotas, confirmed readback, native dataset paths and preview
        status. Incomplete groups remain explicit and never earn success credit.
    """
    path = Path(configuration)
    return _collect_task_program_expansion(
        Path(deployment),
        load_task_program_expansion(path),
        Path(output_dir),
        configuration_sha256=file_digest(path),
        seed=seed,
        operators=operators,
        source_artifacts=source_artifacts,
        runtime_versions=runtime_versions,
        viewer=viewer,
        preview_output=preview_output,
    )


def _collect_task_program_expansion(
    deployment: Path,
    options: dict[str, Any],
    output: Path,
    *,
    configuration_sha256: str,
    seed: int,
    operators: Sequence[str] | None = None,
    source_artifacts: Mapping[str, object] | None = None,
    runtime_versions: Mapping[str, object] | None = None,
    viewer: Mapping[str, object] | None = None,
    preview_output: str | Path | None = None,
) -> Mapping[str, object]:
    from importlib import import_module, metadata
    import torch

    view = load_expansion_viewer(None if viewer is None else dict(viewer))
    selected = list(
        options["trajectory"]["operators"] if operators is None else operators
    )
    if (
        not selected
        or len(set(selected)) != len(selected)
        or set(selected) - set(options["trajectory"]["operators"])
    ):
        raise ValueError("Requested operators must be a distinct enabled subset")
    import_module("lerobot.datasets.lerobot_dataset")
    threads = options["engine"]["cpu_threads"]
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(threads)
    torch.set_num_threads(threads)
    if runtime_versions is None:
        runtime_versions = {}
        for name in (
            "embodichain",
            "dexsim-engine",
            "torch",
            "torchvision",
            "torchcodec",
            "lerobot",
            "numpy",
        ):
            try:
                runtime_versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                runtime_versions[name] = None
    output.mkdir(parents=True, exist_ok=True)
    inputs = output / "inputs"
    inputs.mkdir(exist_ok=True)
    invocation_policy_bindings, sources = resolve_task_program_expansion_bindings(
        deployment,
        source_artifacts=source_artifacts,
        invocation_policy_bindings=invocation_policy_bindings,
    )
    derived = _prepare_deployment(deployment, inputs)
    atomic_json(inputs / "expansion.json", options)
    for path in (derived, inputs / "environment.yaml", inputs / "expansion.json"):
        sources[f"derived_{path.name}"] = {
            "path": str(path),
            "sha256": file_digest(path),
        }
    groups = _run_operators(
        derived,
        output,
        options,
        selected,
        seed=seed,
        sources=sources,
        viewer=view,
        runtime_versions=runtime_versions,
        validator_factory=validator_factory,
        nominal_source_factory=nominal_source_factory,
        invocation_policy_bindings=invocation_policy_bindings,
    )
    result = {
        "complete": all(value.get("complete", False) for value in groups.values()),
        "operators": groups,
        "configuration_sha256": configuration_sha256,
        "viewer": view,
    }
    atomic_json(output / "expansion.json", result)
    if not view["headless"]:
        from .preview import launch_expansion_preview

        result["preview"] = launch_expansion_preview(
            [
                Path(group["directory"])
                for group in groups.values()
                if group.get("committed_episodes", 0)
            ],
            output=(
                Path(preview_output)
                if preview_output is not None
                else output.parent / ".previews" / output.name
            ),
            open_browser=True,
            max_episodes=64,
        )
        atomic_json(output / "expansion.json", result)
    return result
