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

"""Current-package bundle loading for CLI/workflow hosts with independent file checks."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from .bundle_io import read_bundle_identity

if TYPE_CHECKING:
    from .bundle_deployment import LoadedTaskProgramBundle
    from .bundle_io import BundleSnapshot

__all__ = ["load_current_task_program_bundle"]


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _external_inputs(names: tuple[str, ...]) -> dict[str, str]:
    result = {}
    for name in names:
        path = Path(name).expanduser()
        if not path.is_absolute():
            raise ValueError(
                "CLI bundle inputs require absolute filesystem paths; regenerate this bundle with current owner declarations."
            )
        if not path.is_file():
            raise FileNotFoundError(
                f"Required bundle input file is unavailable: {name}"
            )
        result[name] = _digest(path)
    return result


def _required_runtime_inputs(snapshot: BundleSnapshot, *, base_dir: Path) -> set[Path]:
    """Derive actual asset dependencies from sealed effective owner declarations."""
    from embodichain.data.asset_dependencies import collect_declared_asset_files
    from embodichain.lab.gym.utils.composition import resolve_gym_components
    from embodichain.lab.gym.utils.gym_utils import (
        robot_config_asset_files,
        robot_config_from_dict,
    )
    from embodichain.lab.sim.cfg import RobotCfg
    from embodichain.lab.task_program.integrations import (
        DeclaredAssetFileProvider,
        load_configured_task_program_deployment,
    )
    from .bundle_deployment import _decode

    # Recompose the sealed physical declarations instead of trusting mutable
    # assembly views exposed by LoadedTaskProgramBundle.
    sources = {
        name: _decode(content, name)
        for name, content in snapshot.files.items()
        if name.endswith((".yaml", ".yml"))
    }
    physical = resolve_gym_components(
        sources["task_program_deployment.yaml"],
        base_dir=base_dir,
        component_sources=sources,
    )
    robot_cfg = robot_config_from_dict(physical.config["robot"])
    model_files: tuple[Path, ...] = ()
    if type(robot_cfg) is not RobotCfg:
        if not isinstance(robot_cfg, DeclaredAssetFileProvider):
            raise ValueError(
                "Current bundle qualification requires a model-owned "
                "DeclaredAssetFileProvider for a concrete robot configuration; "
                "source-only inventory is insufficient."
            )
        model_files = robot_cfg.declared_asset_files()
        if type(model_files) is not tuple or any(
            not isinstance(path, Path) for path in model_files
        ):
            raise TypeError("Model asset inventory must return a tuple of Path files.")
        if any(not path.is_file() for path in model_files):
            raise ValueError("Model asset inventory contains a missing local file.")
    declarations = [
        {name: value for name, value in physical.config.items() if name != "robot"},
        physical.embodiment_skill_profile,
    ]
    # Physical components and task services can each own asset references. The
    # input-name list in the identity is deliberately not a source here.
    for name in (
        "task_program/integration.yaml",
        "components/scene.yaml",
        "components/execution_policy.yaml",
    ):
        declarations.append(sources[name])
    result: set[Path] = set()
    result.update(model_files)
    result.update(robot_config_asset_files(physical.config["robot"], base_dir=base_dir))
    for declaration in declarations:
        result.update(collect_declared_asset_files(declaration, base_dir=base_dir))
    if physical.embodiment_skill_profile is None:
        raise ValueError(
            "Runtime asset inventory requires an embodiment skill profile."
        )
    deployment = load_configured_task_program_deployment(
        task_program={
            "program": {"inline": sources["task_program/program.yaml"]},
            "integration": {"inline": sources["task_program/integration.yaml"]},
            "execution_policy": {"inline": sources["components/execution_policy.yaml"]},
        },
        skill_profile=physical.embodiment_skill_profile,
        base_dir=base_dir,
    )
    provider = deployment.integration.adapter_factory
    if not isinstance(provider, DeclaredAssetFileProvider):
        raise TypeError("Runtime asset inventory requires a declared asset provider.")
    result.update(provider.declared_asset_files())
    return {path.resolve() for path in result}


def _verify_required_inputs(
    snapshot: BundleSnapshot, *, base_dir: Path, names: tuple[str, ...]
) -> set[Path]:
    required = _required_runtime_inputs(snapshot, base_dir=base_dir)
    declared = {Path(name).expanduser().resolve() for name in names}
    missing = required - declared
    if missing:
        raise ValueError(
            "Bundle external input set omits runtime dependencies: "
            + ", ".join(str(path) for path in sorted(missing))
            + ". Regenerate from frozen task declarations with current input owners."
        )
    return required


def load_current_task_program_bundle(root: str | Path) -> LoadedTaskProgramBundle:
    """Verify a source bundle against actual files and current packaged implementation.

    Args:
        root: Existing sealed Task Engine bundle directory.

    Returns:
        Statically preflighted declarations and lazy factories. The runtime
        envelope is checked before loading Lab integration code; external files
        are hashed independently rather than trusting the envelope's values.
        Required runtime asset paths are independently enumerated from the sealed
        physical/service declarations, so omitting a file from the envelope cannot
        hide a live dependency. Additional provenance inputs remain permitted.
        Input or implementation changes during preflight reject. No simulator, Robot, solver or
        skill is instantiated.

    Explicit application-owned revision/input contracts remain available through
    ``load_configured_bundle``. This CLI/workflow entry always requires the actual
    current package/dependency revision hashes and never converts old bundles.
    """
    directory = Path(root).expanduser().absolute()
    if not directory.is_dir():
        raise FileNotFoundError(f"Bundle directory is unavailable: {directory}")
    identity = read_bundle_identity(directory)
    names = tuple(identity.to_dict()["external_inputs"])
    inputs = _external_inputs(names)
    from .task_program_bundle import task_program_implementation_revisions
    from .bundle_deployment import load_configured_bundle

    revisions = task_program_implementation_revisions()
    loaded = load_configured_bundle(
        directory, external_input_digests=inputs, implementation_revisions=revisions
    )
    required = _verify_required_inputs(loaded.snapshot, base_dir=directory, names=names)
    if _external_inputs(names) != inputs:
        raise ValueError("Bundle external input files changed during static preflight.")
    if (
        _verify_required_inputs(loaded.snapshot, base_dir=directory, names=names)
        != required
    ):
        raise ValueError(
            "Bundle runtime asset dependency set changed during static preflight."
        )
    if task_program_implementation_revisions() != revisions:
        raise ValueError("Bundle implementation changed during static preflight.")
    digest = loaded.snapshot.identity.digest

    def validate_current(selected: LoadedTaskProgramBundle) -> None:
        if selected.root != directory or selected.snapshot.identity.digest != digest:
            raise ValueError("Loaded bundle differs from its original source identity.")
        if _external_inputs(names) != inputs:
            raise ValueError("Bundle external input files changed after loading.")
        if (
            _verify_required_inputs(selected.snapshot, base_dir=directory, names=names)
            != required
        ):
            raise ValueError(
                "Bundle runtime asset dependency set changed after loading."
            )
        if _external_inputs(names) != inputs:
            raise ValueError("Bundle external input files changed during revalidation.")
        if (
            _verify_required_inputs(selected.snapshot, base_dir=directory, names=names)
            != required
        ):
            raise ValueError(
                "Bundle runtime asset dependency set changed during revalidation."
            )
        from . import task_program_bundle as implementation_owner

        if implementation_owner.task_program_implementation_revisions() != revisions:
            raise ValueError("Bundle implementation changed after loading.")

    return replace(loaded, _current_input_validator=validate_current)
