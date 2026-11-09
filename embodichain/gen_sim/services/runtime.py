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

"""Run-scoped service assembly, ownership and safe call records."""

from __future__ import annotations
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Any, TYPE_CHECKING, Iterator
from .errors import CapabilityError, ServiceConfigurationError, ServiceError

if TYPE_CHECKING:
    from .config import GenSimServicesCfg
__all__: list[str] = []

CHAT_ROLES = frozenset(
    {
        "task_interpretation",
        "task_image_grounding",
        "task_semantic_planning",
        "graph_localization",
        "scene_understanding",
        "scene_edit",
        "scene_asset_orientation",
        "scene_asset_dimensions",
        "asset_orientation",
        "asset_dimensions",
        "asset_semantics",
        "asset_physics",
    }
)
ROLE_METHODS = {
    **{role: "complete" for role in CHAT_ROLES},
    "image_generation": "generate",
    "image_segmentation": "segment",
    "geometry_reconstruction": "reconstruct",
    "articulated_generation": "generate",
    "articulation_jobs": "submit",
}


class ServiceRegistry:
    """Explicit adapter factories, keyed by stable IDs and supported roles.

    Factories receive a resolved profile mapping and own cleanup if their
    construction fails before returning an instance. No entry-point discovery
    or process-global mutable registry is used.
    """

    def __init__(self) -> None:
        self._factories: dict[
            str, tuple[Callable[[Mapping[str, Any]], Any], frozenset[str]]
        ] = {}

    def register(
        self,
        adapter_id: str,
        factory: Callable[[Mapping[str, Any]], Any],
        *,
        roles: tuple[str, ...] | frozenset[str],
    ) -> None:
        """Register a factory for declared roles; duplicate IDs are rejected.

        Args:
            adapter_id: Stable ID referenced by a service profile.
            factory: Callable accepting the resolved profile mapping.
            roles: Supported role names, checked before factory invocation.
        """
        if adapter_id in self._factories:
            raise ServiceConfigurationError("Adapter ID already registered")
        if not adapter_id or not roles or set(roles) - ROLE_METHODS.keys():
            raise ServiceConfigurationError("Adapter registration has unknown roles")
        self._factories[adapter_id] = (factory, frozenset(roles))

    def create(self, role: str, profile: Mapping[str, Any]) -> Any:
        """Construct an adapter after checking its declared role capability.

        Args:
            role: Required capability role.
            profile: Resolved profile with an adapter ID.

        Returns:
            A new caller-owned adapter instance.
        """
        entry = self._factories.get(profile["adapter"])
        if entry is None or role not in entry[1]:
            raise CapabilityError(f"No registered adapter for role {role}")
        return entry[0](profile)

    @classmethod
    def defaults(cls) -> ServiceRegistry:
        """Construct a fresh registry containing the shipped adapters."""
        registry = cls()

        def chat(profile: Mapping[str, Any]) -> Any:
            from .adapters.chat import ChatCompletionsAdapter

            try:
                return ChatCompletionsAdapter(
                    **{k: v for k, v in profile.items() if k != "adapter"}
                )
            except TypeError:
                raise ServiceConfigurationError(
                    "Chat profile requires model and api_key"
                ) from None

        registry.register("openai_compatible", chat, roles=CHAT_ROLES)
        for adapter, role in (
            ("image_http", "image_generation"),
            ("segmentation_http", "image_segmentation"),
            ("geometry_http", "geometry_reconstruction"),
            ("articulation_http", "articulated_generation"),
        ):
            registry.register(
                adapter,
                lambda profile, selected=role: _http_adapter(selected, profile),
                roles=(role,),
            )

        def jobs(profile: Mapping[str, Any]) -> Any:
            from .adapters.articulation_jobs import ArticulationServerClient

            return ArticulationServerClient(
                profile["base_url"],
                timeout_seconds=profile.get("timeout_s", 30),
                api_key=profile.get("api_key"),
            )

        registry.register("articulation_jobs_http", jobs, roles=("articulation_jobs",))
        return registry


def _http_adapter(role: str, profile: Mapping[str, Any]) -> Any:
    from .adapters.normalized import (
        ImageAdapter,
        SegmentationAdapter,
        GeometryAdapter,
        ArticulationAdapter,
    )

    cls = {
        "image_generation": ImageAdapter,
        "image_segmentation": SegmentationAdapter,
        "geometry_reconstruction": GeometryAdapter,
        "articulated_generation": ArticulationAdapter,
    }[role]
    options = dict(profile.get("options", {}))
    if set(options) & {
        "base_url",
        "api_key",
        "model",
        "timeout_s",
        "max_attempts",
        "session",
    }:
        raise ServiceConfigurationError(
            "HTTP adapter options cannot override profile connection fields"
        )
    kwargs = {
        key: profile[key]
        for key in ("base_url", "timeout_s", "max_attempts")
        if key in profile
    }
    kwargs.update(options)
    api_key = profile.get("api_key")
    if api_key:
        import requests

        session = requests.Session()
        session.headers["Authorization"] = f"Bearer {api_key}"
        kwargs["session"] = session
    try:
        return cls(**kwargs)
    except TypeError:
        if "session" in kwargs:
            kwargs["session"].close()
        raise ServiceConfigurationError(f"Invalid adapter options for {role}") from None
    except BaseException:
        if "session" in kwargs:
            kwargs["session"].close()
        raise


def _legacy_service(role: str, model: str | None) -> tuple[Any, dict[str, Any]]:
    from .adapters.chat import ChatCompletionsAdapter
    from . import legacy

    if role == "articulation_jobs":
        from .adapters.articulation_jobs import ArticulationServerClient

        local = legacy._load_local_env()
        base_url = legacy._first_env_value(local, "ARTICULATION_SERVER_BASE_URL")
        if not base_url:
            raise ServiceConfigurationError("ARTICULATION_SERVER_BASE_URL is required")
        timeout = float(
            legacy._first_env_value(local, "ARTICULATION_SERVER_TIMEOUT_S") or "30"
        )
        return ArticulationServerClient(base_url, timeout_seconds=timeout), {
            "adapter": "articulation_jobs_http"
        }
    if role in CHAT_ROLES:
        if role in {"task_interpretation", "graph_localization"}:
            settings = legacy._load_llm_settings(model=model)
            options = {
                "default_query": settings.get("default_query", {}),
                "temperature": 0,
            }
            if role == "graph_localization":
                options["max_completion_tokens"] = 6000
        elif role.startswith("asset_"):
            settings = legacy.simready_settings()
            options = {"default_query": settings.get("default_query", {})}
        else:
            from .adapters.vlm_config import load_llm_config

            cfg = load_llm_config()
            settings = {
                "api_key": cfg.api_key,
                "base_url": cfg.base_url,
                "model": cfg.model,
            }
            body_defaults = cfg.default_query
            if role in {"task_image_grounding", "task_semantic_planning"}:
                from .adapters.chat import image_semantic_defaults

                body_defaults = image_semantic_defaults(settings, body_defaults)
            options = {"body_defaults": body_defaults}
        settings["model"] = model or settings["model"]
        profile = {
            "adapter": "openai_compatible",
            **{key: settings[key] for key in ("model", "api_key", "base_url")},
            "options": options,
            "timeout_s": 30 if role == "graph_localization" else 120,
            "max_attempts": 1,
        }
        if role in {
            "scene_understanding",
            "scene_edit",
            "scene_asset_orientation",
            "scene_asset_dimensions",
            "task_image_grounding",
            "task_semantic_planning",
        }:
            profile["max_attempts"] = cfg.max_attempts
        return (
            ChatCompletionsAdapter(
                **{k: v for k, v in profile.items() if k != "adapter"}
            ),
            profile,
        )
    from .adapters.normalized import (
        ImageAdapter,
        SegmentationAdapter,
        GeometryAdapter,
        ArticulationAdapter,
    )

    cls = {
        "image_generation": ImageAdapter,
        "image_segmentation": SegmentationAdapter,
        "geometry_reconstruction": GeometryAdapter,
        "articulated_generation": ArticulationAdapter,
    }[role]
    return cls.from_dotenv(), {"adapter": cls.__name__}


class _ObservedService:
    def __init__(
        self,
        owner: GenSimServices,
        service: Any,
        role: str,
        profile: Mapping[str, Any],
        method: str,
    ) -> None:
        self._owner, self._service, self._role, self._profile, self._method = (
            owner,
            service,
            role,
            profile,
            method,
        )

    def __getattr__(self, name: str) -> Any:
        if name != self._method and not (
            self._role == "articulation_jobs"
            and name in {"status", "cancel", "download"}
        ):
            return getattr(self._service, name)

        def invoke(*args: Any, **kwargs: Any) -> Any:
            request = args[0] if args else None
            with self._owner._lock:
                if self._owner._closed:
                    raise ServiceError("Service run is closed")
            from ._telemetry import attempts

            token = attempts.set(0)
            started = perf_counter()
            record = {
                "role": self._role,
                "operation": name,
                "adapter": self._profile.get("adapter", "injected"),
                "source": self._profile.get("source"),
                "model": getattr(request, "model", None) or self._profile.get("model"),
                "attempts": 1,
                "status": "succeeded",
                "artifacts": [],
            }
            try:
                result = getattr(self._service, name)(*args, **kwargs)
                if hasattr(result, "path"):
                    record["artifacts"] = [str(result.path)]
                if hasattr(result, "objects"):
                    record["artifacts"] = [str(obj.mesh_path) for obj in result.objects]
                if getattr(result, "usage", None) is not None:
                    record["usage"] = {
                        key: value
                        for key, value in result.usage.items()
                        if isinstance(value, (int, float))
                        and not isinstance(value, bool)
                    }
                return result
            except Exception as error:
                record.update(
                    status="failed",
                    error=getattr(error, "code", type(error).__name__),
                    attempts=getattr(error, "attempts", 1),
                )
                raise
            finally:
                record["elapsed_seconds"] = perf_counter() - started
                record["attempts"] = max(record["attempts"], attempts.get())
                attempts.reset(token)
                with self._owner._lock:
                    self._owner._records.append(record)

        return invoke


class GenSimServices:
    """One run's lazy services; injected instances are always borrowed.

    Use as a context manager or explicitly close after concurrent work joins.
    Absent explicit/new JSON config, each role uses its historic configuration
    source. Health checks run only on first use of a role.
    """

    def __init__(
        self,
        cfg: GenSimServicesCfg | None = None,
        *,
        registry: ServiceRegistry | None = None,
        injected: Mapping[str, Any] | None = None,
        model_overrides: Mapping[str, str] | None = None,
    ) -> None:
        if cfg is not None:
            self.cfg, self._source = deepcopy(cfg), "python"
        else:
            from .config import _load_services_config

            self.cfg, self._source = _load_services_config()
        self.registry = registry or ServiceRegistry.defaults()
        self._injected = dict(injected or {})
        self._models = dict(model_overrides or {})
        self._instances: dict[str, Any] = {}
        self._owned: list[Any] = []
        self._profiles: dict[str, dict[str, Any]] = {}
        self._records: list[dict[str, Any]] = []
        self._lock = RLock()
        self._closed = False

    def __enter__(self) -> GenSimServices:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc is None:
            self.close()
        else:
            try:
                self.close()
            except Exception:
                # Resource cleanup must not replace the original operation error.
                pass

    def model(self, role: str, override: str | None = None) -> str | None:
        """Resolve a role model without requiring unused transport credentials."""
        if override or self._models.get(role):
            return override or self._models[role]
        if self.cfg is not None:
            profile = self.cfg.profiles.get(self.cfg.roles.get(role, ""), {})
            value = profile.get("model")
            if isinstance(value, str):
                return value
            if value is not None:
                from .config import resolve_profile

                return resolve_profile(
                    {"adapter": profile["adapter"], "model": value}
                ).get("model")
            return None
        from .legacy import instruction_model

        return (
            instruction_model(None)
            if role in {"task_interpretation", "graph_localization"}
            else None
        )

    def get(self, role: str) -> Any:
        """Resolve and health-check a role on first use, or return its instance."""
        with self._lock:
            if self._closed:
                raise ServiceError("Service run is closed")
            if role not in ROLE_METHODS:
                raise CapabilityError(f"Unknown service role: {role}")
            if role in self._instances:
                return self._instances[role]
            owned = role not in self._injected
            if not owned:
                instance, profile = self._injected[role], {"adapter": "injected"}
            elif self.cfg is None:
                instance, profile = _legacy_service(role, self._models.get(role))
            else:
                from .config import resolve_profile

                name = self.cfg.roles.get(role)
                if name is None:
                    raise ServiceConfigurationError(f"No profile bound to role {role}")
                profile = resolve_profile(self.cfg.profiles[name])
                if self._models.get(role):
                    profile["model"] = self._models[role]
                instance = self.registry.create(role, profile)
            try:
                if not callable(getattr(instance, ROLE_METHODS[role], None)):
                    raise CapabilityError(
                        f"Adapter lacks {ROLE_METHODS[role]} for {role}"
                    )
                check = getattr(instance, "check_health", None)
                if check is not None:
                    check()
            except BaseException as error:
                self._records.append(
                    {
                        "role": role,
                        "operation": "initialize",
                        "status": "failed",
                        "error": getattr(error, "code", type(error).__name__),
                        "attempts": 0,
                    }
                )
                if owned and hasattr(instance, "close"):
                    try:
                        instance.close()
                    except Exception:
                        pass
                raise
            if owned:
                if not any(client is instance for client in self._owned):
                    self._owned.append(instance)
            self._profiles[role] = {
                key: profile[key] for key in ("adapter", "model") if key in profile
            }
            self._profiles[role]["source"] = self._source if owned else "injected"
            self._instances[role] = _ObservedService(
                self, instance, role, self._profiles[role], ROLE_METHODS[role]
            )
            return self._instances[role]

    def records(self) -> list[dict[str, Any]]:
        """Return a copy of safe call records, without prompts or credentials."""
        with self._lock:
            return deepcopy(self._records)

    def selections(self) -> dict[str, dict[str, Any]]:
        """Return only used roles and their non-secret adapter/model identity."""
        with self._lock:
            return deepcopy(self._profiles)

    def write_records(self, path: str | Path) -> None:
        """Atomically publish a safe sidecar outside sealed bundle directories."""
        import json
        import os
        import tempfile

        selected = Path(path).expanduser().resolve()
        if any(
            (parent / "integration_fingerprint.json").exists()
            for parent in selected.parents
        ):
            raise ServiceConfigurationError(
                "Service records cannot change a sealed bundle"
            )
        selected.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".service-calls-", dir=selected.parent)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(
                    {"selections": self.selections(), "calls": self.records()},
                    stream,
                    ensure_ascii=False,
                    indent=2,
                )
                stream.write("\n")
            os.replace(temporary, selected)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def close(self) -> None:
        """Release every owned instance once, even if another close fails."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            owned, self._owned = self._owned, []
        first = None
        for instance in reversed(owned):
            try:
                if hasattr(instance, "close"):
                    instance.close()
            except Exception as error:
                first = first or error
        if first is not None:
            raise first


@contextmanager
def service_scope(services: GenSimServices | None = None) -> Iterator[GenSimServices]:
    """Borrow a supplied run or own a fresh run until the scope exits."""
    if services is not None:
        yield services
    else:
        with GenSimServices() as owned:
            yield owned


def service_entry(function: Callable[..., Any]) -> Callable[..., Any]:
    """Own an implicit run and keep a supplied run borrowed through nested calls."""
    from functools import wraps

    @wraps(function)
    def wrapped(
        *args: Any, services: GenSimServices | None = None, **kwargs: Any
    ) -> Any:
        owner = args[0] if args else kwargs.get("self")
        bound = getattr(owner, "_services", None)
        if isinstance(bound, GenSimServices):
            # Match the method's existing constructor-injection precedence
            # without first discovering an unrelated implicit configuration.
            services = bound
        with service_scope(services) as run:
            return function(*args, services=run, **kwargs)

    return wrapped
