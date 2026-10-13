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

"""Preview registered or file-configured robots with native and Viser controls.

Examples::

    embodichain preview-robot --robot FrankaPanda
    embodichain preview-robot --robot DexforceW1 --variant v025/DH_PGC --viser
    embodichain preview-robot --robot-config robot.yaml --viser
    embodichain preview-robot --robot ur5 --headless --max-steps 10
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from embodichain.cli.sim import add_sim_args_to_parser

if TYPE_CHECKING:
    from embodichain.lab.sim.cfg import RobotCfg
    from embodichain.lab.sim.sim_manager import SimulationManagerCfg

__all__ = ["build_robot_cfg", "cli", "main"]


def _robot_params(value: str) -> dict[str, Any]:
    try:
        params = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(
            f"Invalid --robot-params JSON: {error.msg}"
        ) from error
    if not isinstance(params, dict):
        raise argparse.ArgumentTypeError("--robot-params must be a JSON object.")
    if "class_type" in params:
        raise argparse.ArgumentTypeError(
            "Select class_type using --robot or the robot configuration file."
        )
    return params


def _positive_steps(value: str) -> int:
    try:
        steps = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--max-steps must be a positive integer."
        ) from error
    if steps < 1:
        raise argparse.ArgumentTypeError("--max-steps must be a positive integer.")
    return steps


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="embodichain preview-robot",
        description="Preview a robot in a native window or a Viser browser.",
        epilog="Use list-robot for exact robot and variant IDs. In the native window, press I to activate or toggle configured control-part IK Gizmos.",
    )
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--robot", help="Exact registered robot ID.")
    sources.add_argument(
        "--robot-config", type=Path, help="Standalone RobotCfg JSON/YAML file."
    )
    parser.add_argument("--variant", help="Exact catalog variant ID.")
    parser.add_argument(
        "--robot-params",
        type=_robot_params,
        default={},
        help="JSON robot fields overriding the catalog, variant, or file configuration.",
    )
    parser.add_argument(
        "--joint-control",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Viser joint controls and pose reset.",
    )
    parser.add_argument(
        "--gizmo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable configured control-part IK Gizmos.",
    )
    parser.add_argument(
        "--max-steps",
        type=_positive_steps,
        help="Exit after this many physics steps, including in headless mode.",
    )
    add_sim_args_to_parser(parser)
    return parser


def _resolve_local_assets(config: dict[str, Any], base_dir: Path) -> None:
    """Resolve known local asset paths without rewriting registered data IDs."""

    def local_path(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        path = Path(value).expanduser()
        if path.is_absolute():
            return str(path)
        candidate = base_dir / path
        if candidate.exists() or value.startswith(("./", "../")):
            return str(candidate.resolve())
        return value

    if "fpath" in config:
        config["fpath"] = local_path(config["fpath"])
    urdf = config.get("urdf_cfg")
    if isinstance(urdf, dict):
        components = urdf.get("components", [])
        if isinstance(components, Mapping):
            components = components.values()
        for component in components:
            if isinstance(component, dict) and "urdf_path" in component:
                component["urdf_path"] = local_path(component["urdf_path"])
        sensors = urdf.get("sensors") or []
        if isinstance(sensors, Mapping):
            sensors = sensors.values()
        for sensor in sensors:
            if isinstance(sensor, dict) and "sensor_source" in sensor:
                sensor["sensor_source"] = local_path(sensor["sensor_source"])
        for field in ("fpath", "fpath_prefix"):
            if isinstance(urdf.get(field), str):
                urdf[field] = str((base_dir / Path(urdf[field]).expanduser()).resolve())

    def solver_paths(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "urdf_path":
                    value[key] = local_path(child)
                else:
                    solver_paths(child)

    solver_paths(config.get("solver_cfg"))


def build_robot_cfg(args: argparse.Namespace) -> RobotCfg:
    """Decode a catalog or standalone robot configuration.

    Args:
        args: Parsed preview arguments. File fields and JSON overrides take
            precedence over catalog and explicitly selected variant defaults.

    Returns:
        A validated robot configuration with local file assets resolved.

    Raises:
        ValueError: If selection is invalid or no robot asset is configured.
        TypeError: If the file root is not a mapping.
        OSError: If the configuration file cannot be read.
    """
    from embodichain.lab.sim.cfg import RobotCfg
    from embodichain.robot_registry import resolve_robot_cfg
    from embodichain.utils.config_paths import resolve_config_path
    from embodichain.utils.utility import load_config

    fields: dict[str, Any] = {}
    name = args.robot
    base_dir: Path | None = None
    if args.robot_config is not None:
        path = resolve_config_path(args.robot_config).resolve()
        loaded = load_config(path)
        if not isinstance(loaded, Mapping):
            raise TypeError("Robot configuration root must be a mapping.")
        fields = deepcopy(dict(loaded))
        base_dir = path.parent
        name = fields.pop("class_type", None)
        if "class_type" in loaded and (not isinstance(name, str) or not name):
            raise ValueError("Robot class_type must be a nonempty catalog ID.")
    fields.update(deepcopy(args.robot_params))
    if base_dir is not None:
        _resolve_local_assets(fields, base_dir)

    if name is not None:
        cfg = resolve_robot_cfg(name, fields, variant=args.variant)
    else:
        if args.variant is not None:
            raise ValueError(
                "--variant requires a catalog ID in --robot or file class_type."
            )
        # Optional absent asset fields are equivalent to their generic defaults.
        # The existing asset decoders expect a mapping/path when these fields
        # are supplied, including in a serialized standalone configuration.
        for field in ("fpath", "urdf_cfg"):
            if fields.get(field) is None:
                fields.pop(field, None)
        cfg = RobotCfg.from_dict(fields)
    cfg.validate()
    if not cfg.fpath and cfg.urdf_cfg is None:
        raise ValueError("Robot configuration must define fpath or urdf_cfg.")
    return cfg


def _run_simulation(
    args: argparse.Namespace, robot_cfg: RobotCfg, sim_cfg: SimulationManagerCfg
) -> None:
    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain.lab.scripts._preview import (
        _publish_loaded_assets,
        _run_preview_loop,
        _setup_viser_joint_control,
    )
    from embodichain.utils.logger import log_info, log_warning

    sim = SimulationManager(sim_cfg)
    try:
        robot = sim.add_robot(cfg=robot_cfg)
        if robot is None:
            raise RuntimeError("Robot declaration failed.")
        sim.prepare()
        if not args.headless and not args.viser:
            if not sim.open_window():
                raise RuntimeError("The native preview window could not be opened.")
        sim.update_gizmos()
        controller = _setup_viser_joint_control(
            sim, [robot], args, follow_external_targets=args.gizmo
        )
        _publish_loaded_assets(sim, args)
        log_info(
            f"Robot {robot.uid!r}: {len(robot.active_joint_ids)} independent joints; "
            f"control parts: {tuple(robot.control_parts or {})}."
        )
        if args.gizmo:
            parts = [
                gizmo.control_part
                for _, gizmo in sim.get_gizmo_items()
                if gizmo.target is robot
            ]
            if parts:
                log_info(
                    f"IK Gizmo control parts: {tuple(parts)}. Native activation key: I."
                )
            elif not args.headless or args.viser:
                log_warning(
                    "No IK Gizmos are available; check control-part chain/TCP metadata and the single-environment requirement."
                )
        interactive = bool(args.viser or not args.headless)
        if interactive:
            log_info("Robot preview ready. Press Ctrl+C to exit.", color="green")
        _run_preview_loop(
            sim, controller, interactive=interactive, max_steps=args.max_steps
        )
    finally:
        original_error = sys.exc_info()[1]
        try:
            sim.destroy(exit_process=False)
        except Exception as error:
            if original_error is None:
                raise
            log_warning(f"Preview cleanup failed after {original_error!r}: {error!r}")


def _clear_failure_frames(error: BaseException) -> None:
    """Release unwound native views in the primary error and chained causes."""
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        traceback.clear_frames(current.__traceback__)
        pending.extend(
            chained
            for chained in (current.__cause__, current.__context__)
            if chained is not None
        )


def main(args: argparse.Namespace) -> None:
    """Load and preview a robot, releasing the owned simulation before returning.

    Args:
        args: Parsed robot-preview arguments.

    Raises:
        Exception: The original configuration, loading, or runtime failure.
    """
    from embodichain.lab.scripts._preview import _build_sim_cfg
    from embodichain.lab.sim.objects import GizmoCfg
    from embodichain.lab.sim.sim_manager import SimulationManager

    robot_cfg = build_robot_cfg(args)
    sim_cfg = _build_sim_cfg(args, headless=True)
    sim_cfg.robot_ik_gizmo = GizmoCfg() if args.gizmo else None
    failure: BaseException | None = None
    try:
        _run_simulation(args, robot_cfg, sim_cfg)
    except BaseException as error:
        # Native views may be retained by unwound constructor/update frames.
        # Clear those frames before draining deferred destruction outside the
        # exception handler, retaining the original exception type and message.
        _clear_failure_frames(error)
        failure = error.with_traceback(None)
    try:
        SimulationManager.flush_cleanup_queue()
    except BaseException as error:
        if failure is None:
            raise
        from embodichain.utils.logger import log_warning

        log_warning(f"Deferred preview cleanup failed after {failure!r}: {error!r}")
    if failure is not None:
        raise failure


def cli(argv: Sequence[str] | None = None) -> None:
    """Run the robot-preview command.

    Args:
        argv: Arguments excluding the command name, or None for sys.argv.
    """
    parser = _create_parser()
    args = parser.parse_args(argv)
    try:
        main(args)
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    cli()
