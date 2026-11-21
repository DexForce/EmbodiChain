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

"""Provider-independent configuration and connectivity diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._cfg import GenSimServicesCfg
from .config import load_services_config, resolve_profile
from .errors import CapabilityError, ServiceConfigurationError
from .runtime import GenSimServices, ServiceRegistry

__all__: list[str] = []


def validate_services(
    config_path: str | Path | None = None,
    *,
    env_path: str | Path | None = None,
) -> tuple[dict[str, Any], GenSimServicesCfg, ServiceRegistry]:
    """Resolve selected profiles and verify role-to-adapter compatibility.

    Args:
        config_path: Explicit configuration, or ``None`` for normal discovery.
        env_path: Optional dotenv supplying explicit profile references.

    Returns:
        A safe report, resolved configuration and run-local adapter registry.

    Raises:
        ServiceConfigurationError: If no explicit profile configuration is
            selected or a profile cannot be resolved.
        CapabilityError: If an adapter does not support a bound role.
    """
    configured = load_services_config(config_path)
    if configured is None:
        raise ServiceConfigurationError(
            "No explicit model_services configuration is selected"
        )
    selected_profiles = set(configured.roles.values())
    selected_env = Path(env_path).expanduser() if env_path is not None else None
    resolved_profiles = {
        name: resolve_profile(configured.profiles[name], selected_env)
        for name in selected_profiles
    }
    resolved = GenSimServicesCfg(
        profiles=resolved_profiles,
        roles=dict(configured.roles),
    )
    registry = ServiceRegistry.defaults(
        adapter_ids={profile["adapter"] for profile in resolved.profiles.values()}
    )
    roles = []
    profile_roles: dict[str, str] = {}
    for role, profile_name in sorted(resolved.roles.items()):
        profile = resolved.profiles[profile_name]
        adapter = profile["adapter"]
        if not registry.supports(adapter, role):
            raise CapabilityError(f"No registered adapter for role {role}")
        profile_roles.setdefault(profile_name, role)
        roles.append(
            {
                "role": role,
                "profile": profile_name,
                "adapter": adapter,
                **(
                    {"model": profile["model"]}
                    if isinstance(profile.get("model"), str)
                    else {}
                ),
            }
        )
    for profile_name, role in profile_roles.items():
        instance = registry.create(role, resolved.profiles[profile_name])
        if hasattr(instance, "close"):
            instance.close()
    return {"status": "ok", "roles": roles}, resolved, registry


def doctor_services(
    config_path: str | Path | None = None,
    *,
    env_path: str | Path | None = None,
    roles: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Initialize selected roles and run adapter health checks when available.

    Args:
        config_path: Explicit configuration, or ``None`` for normal discovery.
        env_path: Optional dotenv supplying explicit profile references.
        roles: Bound roles to inspect. An empty tuple selects every bound role.

    Returns:
        A safe per-role report. Failed roles do not prevent checking the rest.

    Raises:
        ServiceConfigurationError: If a requested role is not configured.
        CapabilityError: If an adapter does not support a bound role.
    """
    _, resolved, registry = validate_services(config_path, env_path=env_path)
    selected = tuple(dict.fromkeys(roles)) or tuple(sorted(resolved.roles))
    unknown = [role for role in selected if role not in resolved.roles]
    if unknown:
        raise ServiceConfigurationError(
            "Requested service role is not configured: " + ", ".join(unknown)
        )
    results = []
    with GenSimServices(resolved, registry=registry) as services:
        for role in selected:
            profile_name = resolved.roles[role]
            adapter = resolved.profiles[profile_name]["adapter"]
            try:
                service = services.get(role)
                health_checked = callable(getattr(service, "check_health", None))
                results.append(
                    {
                        "role": role,
                        "profile": profile_name,
                        "adapter": adapter,
                        "status": "ready" if health_checked else "initialized",
                        "health_checked": health_checked,
                    }
                )
            except Exception as error:
                results.append(
                    {
                        "role": role,
                        "profile": profile_name,
                        "adapter": adapter,
                        "status": "failed",
                        "error": getattr(error, "code", type(error).__name__),
                        "detail": str(error),
                    }
                )
    return {
        "status": (
            "failed" if any(item["status"] == "failed" for item in results) else "ok"
        ),
        "roles": results,
    }
