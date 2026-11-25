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

"""Opt-in stdio tools for catalog-backed policy runs in DexSim Worlds."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
from time import monotonic
from typing import Any

from dexsim.runtime import Session
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

__all__: list[str] = []

_FIELDS = ("robot", "policy", "scene", "task")
_ID = re.compile(r"[a-z0-9][a-z0-9._/-]*@[1-9][0-9]*")
_RUN = re.compile(r"ecrun_[0-9a-f]{24}")
_SESSION = re.compile(r"dsws_[0-9a-f]{32}")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:24]


def _read_catalog(path: Path) -> dict[tuple[str, ...], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    profiles: dict[tuple[str, ...], dict[str, Any]] = {}
    plans: set[str] = set()
    for profile in data["profiles"]:
        ids = tuple(profile[field] for field in _FIELDS)
        if any(not isinstance(item, str) or not _ID.fullmatch(item) for item in ids):
            raise ValueError("Policy catalog IDs must use the name@version form")
        if ids in profiles:
            raise ValueError(f"Duplicate policy composition {ids}")
        entrypoint = profile["entrypoint"]
        if not re.fullmatch(
            r"[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*:[a-zA-Z_]\w*", entrypoint
        ):
            raise ValueError("Policy entrypoints must use module:function form")
        if not isinstance(profile["world"], dict) or not isinstance(
            profile.get("options", {}), dict
        ):
            raise ValueError("Policy world requirements and options must be objects")
        files = {
            # Preserve symlinked virtualenv executables and their environment.
            name: str((path.parent / value).absolute())
            for name, value in profile.get("files", {}).items()
        }
        plan_id = f"ecplan_{_digest(profile)}"
        if plan_id in plans:
            raise ValueError(f"Duplicate plan ID {plan_id}")
        plans.add(plan_id)
        profiles[ids] = {**profile, "files": files, "plan_id": plan_id}
    if not profiles:
        raise ValueError("Policy catalog must register at least one composition")
    return profiles


def _check_files(profile: dict[str, Any]) -> None:
    for name, filename in profile["files"].items():
        if not Path(filename).exists():
            raise ToolError(f"Missing policy resource {name!r}: {filename}")


def _write_receipt(path: Path, data: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, encoding="utf-8", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(data, stream, allow_nan=False)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _read_receipt(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ToolError(f"Run receipt is unavailable: {path.parent.name}") from exc


class _PolicyRuns:
    def __init__(
        self, catalog: Path, output: Path, directory: str | None, session_id: str | None
    ) -> None:
        self.profiles = _read_catalog(catalog.resolve())
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.directory = directory
        self.session_id = session_id
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy-run")

    def prepare(self, spec: dict[str, str]) -> dict[str, Any]:
        if set(spec) != set(_FIELDS):
            raise ToolError("Supply exactly robot, policy, scene and task")
        profile = self.profiles.get(tuple(spec[field] for field in _FIELDS))
        if profile is None:
            for i, field in enumerate(_FIELDS):
                values = sorted({ids[i] for ids in self.profiles})
                if spec[field] not in values:
                    raise ToolError(
                        f"Unknown {field} {spec[field]!r}. Available: {values}"
                    )
            raise ToolError(
                "This robot, policy, scene and task combination is incompatible"
            )
        _check_files(profile)
        return {
            "plan_id": profile["plan_id"],
            "resolved": {field: profile[field] for field in _FIELDS},
            "world": profile["world"],
        }

    def start(self, plan_id: str, session_id: str) -> dict[str, Any]:
        if not _SESSION.fullmatch(session_id):
            raise ToolError("Invalid DexSim Session ID")
        if self.session_id is not None and session_id != self.session_id:
            raise ToolError("The policy service is bound to another DexSim Session")
        profile = next(
            (item for item in self.profiles.values() if item["plan_id"] == plan_id),
            None,
        )
        if profile is None:
            raise ToolError("Unknown plan ID. Call embodichain_prepare_run first")
        run_id = f"ecrun_{_digest(session_id)}"
        folder = self.output / run_id
        receipt_path = folder / "receipt.json"
        if not folder.exists():
            _check_files(profile)
        try:
            folder.mkdir(mode=0o700)
        except FileExistsError:
            receipt = _read_receipt(receipt_path)
            if receipt["plan_id"] != plan_id:
                raise ToolError("The World already has an assigned policy run")
            return receipt
        receipt = {
            "run_id": run_id,
            "plan_id": plan_id,
            "session_id": session_id,
            "resolved": {field: profile[field] for field in _FIELDS},
            "status": "queued",
            "done": False,
        }
        _write_receipt(receipt_path, receipt)
        self.worker.submit(self._execute, profile, receipt, receipt_path)
        return receipt

    def _execute(
        self, profile: dict[str, Any], receipt: dict[str, Any], path: Path
    ) -> None:
        started = monotonic()
        _write_receipt(path, {**receipt, "status": "running"})
        try:
            with Session.connect(
                receipt["session_id"],
                directory=self.directory,
                client={"kind": "embodichain-policy-mcp", "run_id": receipt["run_id"]},
            ) as session:
                config = session.info.get("config") or {}
                for field, expected in profile["world"].items():
                    actual = config.get(field)
                    if actual != expected:
                        raise ValueError(
                            f"World {field}: expected {expected!r}, got {actual!r}"
                        )
                result = session.invoke(
                    "embodichain.lab.sim.dexsim_session._policy_run",
                    run_id=receipt["run_id"],
                    entrypoint=profile["entrypoint"],
                    files=profile["files"],
                    options=profile.get("options", {}),
                    output=str(path.parent),
                )
            if not isinstance(result, dict) or type(result.get("passed")) is not bool:
                raise ValueError(
                    "Policy runners must return a mapping with a passed boolean"
                )
            json.dumps(result, allow_nan=False)
            receipt = {
                **receipt,
                "status": "succeeded" if result["passed"] else "failed",
                "task_passed": result["passed"],
                "result": result,
            }
        except Exception as exc:
            receipt = {
                **receipt,
                "status": "failed",
                "task_passed": False,
                "error": str(exc),
            }
        _write_receipt(
            path, {**receipt, "duration_s": monotonic() - started, "done": True}
        )

    async def get(self, run_id: str, wait_seconds: float) -> dict[str, Any]:
        if not _RUN.fullmatch(run_id):
            raise ToolError("Invalid policy run ID")
        if not math.isfinite(wait_seconds) or not 0 <= wait_seconds <= 20:
            raise ToolError("wait_seconds must be between 0 and 20")
        deadline = monotonic() + wait_seconds
        path = self.output / run_id / "receipt.json"
        while True:
            receipt = _read_receipt(path)
            remaining = deadline - monotonic()
            if receipt["status"] in ("succeeded", "failed") or remaining <= 0:
                return receipt
            await asyncio.sleep(min(0.2, remaining))


def _create_server(runs: _PolicyRuns) -> MCPServer:
    server = MCPServer("EmbodiChain policy runs")
    entries = "\n".join(" / ".join(ids) for ids in runs.profiles)

    @server.tool(
        description=(
            "Validate four versioned IDs (robot, policy, scene, task) and return a plan ID "
            "and World creation requirements. No World is needed. Registered combinations:\n"
            + entries
        ),
        annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False),
    )
    def embodichain_prepare_run(spec: dict[str, str]) -> dict[str, Any]:
        return runs.prepare(spec)

    @server.tool()
    def embodichain_start_run(plan_id: str, session_id: str) -> dict[str, Any]:
        """Start one prepared policy run in an empty World and return its run ID."""
        return runs.start(plan_id, session_id)

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
    async def embodichain_get_run(
        run_id: str, wait_seconds: float = 0
    ) -> dict[str, Any]:
        """Read a policy run receipt, optionally waiting up to 20 seconds."""
        return await runs.get(run_id, wait_seconds)

    return server


def serve(
    catalog: Path, output: Path, *, directory: str | None, session_id: str | None
) -> None:
    """Serve registered policy tools using an operator-selected catalog and World scope."""
    if directory is None and session_id is None:
        raise ValueError(
            "Policy runs require a Session ID or an explicit discovery directory"
        )
    runs = _PolicyRuns(catalog, output, directory, session_id)
    with runs.worker:
        _create_server(runs).run(transport="stdio")
