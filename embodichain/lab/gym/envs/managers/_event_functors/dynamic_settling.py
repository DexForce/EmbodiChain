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

"""Dynamic-object settling event functor."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import torch

from embodichain.lab.gym.envs.managers.cfg import SceneEntityCfg
from embodichain.lab.gym.envs.settling import (
    DynamicSettleMonitor,
    DynamicSettleMonitorCfg,
    DynamicSettleSample,
)
from embodichain.lab.sim.objects import Articulation, RigidObject, RigidObjectGroup
from embodichain.utils import logger

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv


__all__ = ["wait_for_dynamic_objects_to_settle"]

_DynamicEntity = RigidObject | RigidObjectGroup | Articulation
_SettleEntity = tuple[str, SceneEntityCfg, _DynamicEntity]


def _validate_settle_parameters(
    linear_velocity_threshold: float,
    angular_velocity_threshold: float,
    min_steps: int,
    max_steps: int,
    check_interval_steps: int,
    required_stable_checks: int,
    timeout_behavior: str,
    allow_partial_envs: bool,
) -> DynamicSettleMonitorCfg:
    """Validate parameters and return the reusable monitor policy."""
    cfg = DynamicSettleMonitorCfg(
        linear_velocity_threshold=linear_velocity_threshold,
        angular_velocity_threshold=angular_velocity_threshold,
        min_steps=min_steps,
        max_steps=max_steps,
        check_interval_steps=check_interval_steps,
        required_stable_checks=required_stable_checks,
    )
    if timeout_behavior not in ("warn", "raise"):
        raise ValueError("timeout_behavior must be either 'warn' or 'raise'.")
    if not isinstance(allow_partial_envs, bool):
        raise TypeError("allow_partial_envs must be a boolean.")
    return cfg


def _normalize_settle_env_ids(
    env: EmbodiedEnv,
    env_ids: torch.Tensor | Sequence[int] | slice | None,
) -> torch.Tensor:
    """Normalize environment IDs into a unique one-dimensional tensor."""
    all_env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    if env_ids is None:
        return all_env_ids
    if isinstance(env_ids, slice):
        return all_env_ids[env_ids]

    raw_env_ids = torch.as_tensor(env_ids, device=env.device)
    if raw_env_ids.dtype == torch.bool:
        raise TypeError("env_ids must contain integer indices, not booleans.")
    if raw_env_ids.ndim == 0:
        raw_env_ids = raw_env_ids.unsqueeze(0)
    elif raw_env_ids.ndim != 1:
        raise ValueError("env_ids must be a one-dimensional sequence of indices.")
    if raw_env_ids.is_floating_point() and not bool(
        torch.equal(raw_env_ids, raw_env_ids.round())
    ):
        raise ValueError("env_ids must contain integer-valued indices.")

    normalized = torch.unique(raw_env_ids.to(dtype=torch.long), sorted=True)
    if normalized.numel() > 0 and bool(
        ((normalized < 0) | (normalized >= env.num_envs)).any().item()
    ):
        raise IndexError(f"env_ids must be within [0, {env.num_envs - 1}].")
    return normalized


def _get_dynamic_entity_catalog(
    env: EmbodiedEnv,
) -> dict[str, tuple[str, _DynamicEntity]]:
    """Collect settle-capable non-robot scene entities by UID."""
    catalog: dict[str, tuple[str, _DynamicEntity]] = {}

    for uid in env.sim.get_rigid_object_uid_list():
        entity = env.sim.get_rigid_object(uid)
        if entity is not None:
            catalog[uid] = ("rigid_object", entity)
    for uid in env.sim.get_rigid_object_group_uid_list():
        entity = env.sim.get_rigid_object_group(uid)
        if entity is not None:
            catalog[uid] = ("rigid_object_group", entity)
    for uid in env.sim.get_articulation_uid_list():
        entity = env.sim.get_articulation(uid)
        if entity is not None:
            catalog[uid] = ("articulation", entity)

    return catalog


def _is_dynamic_entity(kind: str, entity: _DynamicEntity) -> bool:
    """Return whether an entity participates in dynamic physics.

    Articulation links are physics-backed even when ``fix_base`` constrains the
    root link, so every non-robot articulation is a valid settle target.
    """
    if kind == "articulation":
        return True
    return not entity.is_non_dynamic


def _resolve_settle_entities(
    env: EmbodiedEnv,
    entity_cfgs: Sequence[SceneEntityCfg] | None,
) -> list[_SettleEntity]:
    """Resolve explicit or automatically discovered dynamic entities."""
    catalog = _get_dynamic_entity_catalog(env)
    explicit = entity_cfgs is not None
    if entity_cfgs is None:
        configs = [SceneEntityCfg(uid=uid) for uid in catalog]
    else:
        configs = list(entity_cfgs)
        if not configs:
            raise ValueError("entity_cfgs must not be empty when provided.")

    resolved: list[_SettleEntity] = []
    seen_uids: set[str] = set()
    robot_uids = set(env.sim.get_robot_uid_list())
    for entity_cfg in configs:
        if not isinstance(entity_cfg, SceneEntityCfg):
            raise TypeError(
                "entity_cfgs must contain only SceneEntityCfg instances, got "
                f"{type(entity_cfg).__name__}."
            )
        if entity_cfg.uid in seen_uids:
            continue
        seen_uids.add(entity_cfg.uid)

        catalog_entry = catalog.get(entity_cfg.uid)
        if catalog_entry is None:
            if entity_cfg.uid in robot_uids:
                raise ValueError(
                    f"Robot '{entity_cfg.uid}' cannot be used as a settle target."
                )
            raise ValueError(
                f"Settle target '{entity_cfg.uid}' is not a rigid object, rigid "
                "object group, or articulation."
            )

        kind, entity = catalog_entry
        if not _is_dynamic_entity(kind, entity):
            if explicit:
                raise ValueError(
                    f"Settle target '{entity_cfg.uid}' is static or kinematic."
                )
            continue
        resolved.append((kind, entity_cfg, entity))

    return resolved


def _measure_settle_speeds(
    entities: Sequence[_SettleEntity],
    env_ids: torch.Tensor,
) -> list[DynamicSettleSample]:
    """Measure per-body linear and angular speeds for selected environments."""
    samples: list[DynamicSettleSample] = []
    for kind, entity_cfg, entity in entities:
        if kind == "articulation":
            velocity = entity.body_data.body_link_vel[env_ids]
            velocity = velocity[:, entity_cfg.body_ids, :]
            linear_velocity = velocity[..., :3]
            angular_velocity = velocity[..., 3:]
        else:
            body_data = entity.body_data
            if body_data is None:
                raise RuntimeError(
                    f"Dynamic settle target '{entity_cfg.uid}' has no body data."
                )
            linear_velocity = body_data.lin_vel[env_ids]
            angular_velocity = body_data.ang_vel[env_ids]

        if linear_velocity.numel() == 0 or angular_velocity.numel() == 0:
            raise ValueError(
                f"Settle target '{entity_cfg.uid}' selected no physical bodies."
            )
        linear_speed = torch.linalg.vector_norm(linear_velocity, dim=-1).reshape(
            env_ids.numel(), -1
        )
        angular_speed = torch.linalg.vector_norm(angular_velocity, dim=-1).reshape(
            env_ids.numel(), -1
        )
        samples.append(
            DynamicSettleSample(
                entity_id=entity_cfg.uid,
                linear_speed=linear_speed,
                angular_speed=angular_speed,
            )
        )
    return samples


def _format_settle_timeout(
    samples: Sequence[DynamicSettleSample],
    env_ids: torch.Tensor,
    linear_velocity_threshold: float,
    angular_velocity_threshold: float,
    max_steps: int,
    stable_checks: int,
    required_stable_checks: int,
) -> str:
    """Build a timeout message with per-entity environment diagnostics."""
    unsettled: list[str] = []
    all_linear_speeds: list[torch.Tensor] = []
    all_angular_speeds: list[torch.Tensor] = []
    for sample in samples:
        linear_speed = sample.linear_speed
        angular_speed = sample.angular_speed
        stable = (
            torch.isfinite(linear_speed)
            & torch.isfinite(angular_speed)
            & (linear_speed <= linear_velocity_threshold)
            & (angular_speed <= angular_velocity_threshold)
        )
        unsettled_mask = ~stable.all(dim=1)
        if bool(unsettled_mask.any().item()):
            unsettled_env_ids = env_ids[unsettled_mask].detach().cpu().tolist()
            unsettled.append(f"{sample.entity_id}(env_ids={unsettled_env_ids})")
        all_linear_speeds.append(linear_speed.reshape(-1))
        all_angular_speeds.append(angular_speed.reshape(-1))

    linear_speeds = torch.cat(all_linear_speeds)
    angular_speeds = torch.cat(all_angular_speeds)
    infinity = torch.tensor(float("inf"), device=linear_speeds.device)
    max_linear_speed = torch.where(
        torch.isfinite(linear_speeds), linear_speeds, infinity
    ).max()
    infinity = infinity.to(device=angular_speeds.device)
    max_angular_speed = torch.where(
        torch.isfinite(angular_speeds), angular_speeds, infinity
    ).max()
    unsettled_summary = ", ".join(unsettled) or "none at the final check"
    return (
        f"Dynamic objects did not settle within {max_steps} physics steps. "
        f"Stable checks: {stable_checks}/{required_stable_checks}; unsettled: "
        f"{unsettled_summary}; maximum linear speed: "
        f"{max_linear_speed.item():.6g} m/s; maximum angular speed: "
        f"{max_angular_speed.item():.6g} rad/s."
    )


def wait_for_dynamic_objects_to_settle(
    env: EmbodiedEnv,
    env_ids: torch.Tensor | Sequence[int] | slice | None,
    entity_cfgs: Sequence[SceneEntityCfg] | None = None,
    linear_velocity_threshold: float = 0.03,
    angular_velocity_threshold: float = 0.20,
    min_steps: int = 10,
    max_steps: int = 240,
    check_interval_steps: int = 2,
    required_stable_checks: int = 3,
    timeout_behavior: Literal["warn", "raise"] = "warn",
    allow_partial_envs: bool = False,
) -> None:
    """Advance physics until selected dynamic objects remain stationary.

    The functor waits at least ``min_steps`` and then polls every
    ``check_interval_steps``. Every selected body in every selected environment
    must remain below both velocity thresholds for
    ``required_stable_checks`` consecutive polls. It never clears dynamics and
    never advances beyond ``max_steps``.

    When ``entity_cfgs`` is ``None``, all dynamic rigid objects, rigid object
    groups, and non-robot articulations are selected automatically. Static and
    kinematic entities are ignored during automatic discovery.

    .. attention::
        :meth:`SimulationManager.update` advances the entire vectorized physics
        world. Partial ``env_ids`` are therefore rejected by default. Set
        ``allow_partial_envs=True`` only when advancing non-target environments
        is acceptable.

    Args:
        env: The environment instance.
        env_ids: Target environment IDs. ``None`` or ``slice(None)`` selects all
            environments.
        entity_cfgs: Explicit settle targets. ``None`` discovers all supported
            dynamic non-robot entities.
        linear_velocity_threshold: Maximum stable linear speed in meters per
            second.
        angular_velocity_threshold: Maximum stable angular speed in radians per
            second.
        min_steps: Physics steps to run before the first stability check.
        max_steps: Maximum total number of physics steps to run.
        check_interval_steps: Physics steps between stability checks.
        required_stable_checks: Consecutive stable checks required before return.
        timeout_behavior: ``"warn"`` to log and continue or ``"raise"`` to
            raise :class:`TimeoutError` when ``max_steps`` is reached.
        allow_partial_envs: Whether to permit a partial environment selection
            despite whole-world physics advancement.

    Raises:
        IndexError: If an environment ID is outside the valid range.
        RuntimeError: If a dynamic target has no readable body data.
        TimeoutError: If objects do not settle and ``timeout_behavior`` is
            ``"raise"``.
        TypeError: If a parameter or entity configuration has the wrong type.
        ValueError: If parameters, targets, or environment selection are invalid.
    """
    monitor_cfg = _validate_settle_parameters(
        linear_velocity_threshold=linear_velocity_threshold,
        angular_velocity_threshold=angular_velocity_threshold,
        min_steps=min_steps,
        max_steps=max_steps,
        check_interval_steps=check_interval_steps,
        required_stable_checks=required_stable_checks,
        timeout_behavior=timeout_behavior,
        allow_partial_envs=allow_partial_envs,
    )
    target_env_ids = _normalize_settle_env_ids(env, env_ids)
    if target_env_ids.numel() == 0:
        return

    all_env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    if not allow_partial_envs and not torch.equal(target_env_ids, all_env_ids):
        raise ValueError(
            "Partial env_ids would still advance every environment in the physics "
            "world. Pass allow_partial_envs=True to accept this side effect."
        )

    entities = _resolve_settle_entities(env, entity_cfgs)
    if not entities:
        logger.log_warning("No dynamic objects were found to settle.")
        return

    step_count = 0
    if min_steps > 0:
        env.sim.update(step=min_steps)
        step_count = min_steps

    monitor = DynamicSettleMonitor(monitor_cfg, target_env_ids)
    samples: list[DynamicSettleSample]
    settle_state = None
    while True:
        samples = _measure_settle_speeds(entities, target_env_ids)
        settle_state = monitor.observe(samples, elapsed_steps=step_count)
        if bool(settle_state.settled_mask.all().item()):
            return

        if step_count >= max_steps:
            break
        update_steps = min(check_interval_steps, max_steps - step_count)
        env.sim.update(step=update_steps)
        step_count += update_steps

    timeout_message = _format_settle_timeout(
        samples=samples,
        env_ids=target_env_ids,
        linear_velocity_threshold=linear_velocity_threshold,
        angular_velocity_threshold=angular_velocity_threshold,
        max_steps=max_steps,
        stable_checks=int(settle_state.stable_counts.min().item()),
        required_stable_checks=required_stable_checks,
    )
    if timeout_behavior == "raise":
        raise TimeoutError(timeout_message)
    logger.log_warning(timeout_message)
