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

"""Single-World jobs advanced between DexSim owner-thread requests."""

from __future__ import annotations

from collections.abc import Callable, Generator
from dataclasses import dataclass
import hashlib
import hmac
import json
from pathlib import Path
import re
from time import time
from typing import TYPE_CHECKING, Any

from .writer import _signature

if TYPE_CHECKING:
    from dexsim.runtime import SessionContext

__all__: list[str] = []

MODULE = __name__
RUN_ID = re.compile(r"ecmotion_[0-9a-f]{24}")
_COMMAND_ID = re.compile(r"[a-zA-Z0-9_.:-]{1,128}")
_UPDATE_NAME = "embodichain-motion"
Work = Generator[dict[str, Any], None, dict[str, Any]]


class JobRejected(ValueError):
    """Reject an execution before applying its next native command."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def sign(key: bytes, session_id: str, operation: str, arguments: dict[str, Any]) -> str:
    return _signature(key, session_id, f"{MODULE}/{operation}", arguments)


@dataclass
class _ActiveJob:
    receipt: dict[str, Any]
    work: Work
    cancel_requested: bool = False


class MotionJobs:
    """Own a job cursor and durable receipts for one manual World."""

    def __init__(
        self,
        context: SessionContext,
        output: Path,
        progress: Callable[[], dict[str, Any]],
    ) -> None:
        self.context = context
        self.output = output
        self.output.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.progress = progress
        self.active: _ActiveJob | None = None
        self.binding_id: str | None = None

    def _folder(self, run_id: str) -> Path:
        if not RUN_ID.fullmatch(run_id):
            raise ValueError("Unknown motion run ID")
        return self.output / run_id

    @staticmethod
    def _save(path: Path, data: dict[str, Any]) -> None:
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
        temp.replace(path)

    def _publish(self, job: _ActiveJob) -> None:
        job.receipt.update(self.progress())
        self._save(self._folder(job.receipt["run_id"]) / "receipt.json", job.receipt)

    def start(
        self,
        command_id: str,
        kind: str,
        request: dict[str, Any],
        make_work: Callable[[Path], Work],
    ) -> dict[str, Any]:
        """Admit an idempotent command and return its owner-thread job ID."""
        if not _COMMAND_ID.fullmatch(command_id):
            raise ValueError("Invalid command_id")
        digest = hashlib.sha256(
            f"{self.context.session_id}:{command_id}".encode()
        ).hexdigest()[:24]
        run_id = f"ecmotion_{digest}"
        folder = self._folder(run_id)
        intent = {"kind": kind, "request": request}
        if self.binding_id is not None:
            intent["binding_id"] = self.binding_id
        if folder.exists():
            recorded = json.loads((folder / "intent.json").read_text(encoding="utf-8"))
            if recorded != intent:
                raise ValueError("COMMAND_ID_CONFLICT")
            return json.loads((folder / "receipt.json").read_text(encoding="utf-8"))
        if self.active is not None:
            raise ValueError("WORLD_BUSY")
        folder.mkdir(mode=0o700)
        self._save(folder / "intent.json", intent)
        job = _ActiveJob(
            receipt={
                "run_id": run_id,
                "session_id": self.context.session_id,
                "command_id": command_id,
                "receipt_path": str(folder / "receipt.json"),
                "kind": kind,
                "status": "queued",
                "done": False,
                "physical_verdict": "pending",
                "submitted_at_s": time(),
            },
            work=make_work(folder),
        )
        self._publish(job)
        self.active = job
        self.context.register_update(_UPDATE_NAME, self._tick, interval=0.001)
        return dict(job.receipt)

    def cancel(self, run_id: str) -> dict[str, Any]:
        """Request a stop at the next owner-thread boundary."""
        folder = self._folder(run_id)
        job = self.active
        if job is not None and job.receipt["run_id"] == run_id:
            job.cancel_requested = True
            job.receipt["status"] = "cancelling"
            self._publish(job)
            return dict(job.receipt)
        return json.loads((folder / "receipt.json").read_text(encoding="utf-8"))

    def _finish(self, job: _ActiveJob, status: str, verdict: str) -> None:
        self.context.remove_update(_UPDATE_NAME)
        self.active = None
        try:
            job.work.close()
        finally:
            job.receipt.update(status=status, done=True, physical_verdict=verdict)
            self._publish(job)

    def _tick(self, _now: float) -> None:
        job = self.active
        assert job is not None
        if job.cancel_requested:
            self._finish(job, "cancelled", "unverified")
            return
        try:
            job.receipt["status"] = "running"
            job.receipt.update(next(job.work))
        except StopIteration as finished:
            result = finished.value
            job.receipt["result"] = result
            passed = result["passed"]
            self._finish(
                job, "succeeded" if passed else "failed", "pass" if passed else "fail"
            )
        except Exception as exc:
            rejected = isinstance(exc, JobRejected)
            job.receipt["error"] = {
                "code": exc.code if rejected else "EXECUTION_ERROR",
                "message": str(exc),
            }
            self._finish(job, "failed", "unverified")
        else:
            self._publish(job)

    def close(self) -> None:
        """Close an active job without executing another command."""
        if self.active is not None:
            self._finish(self.active, "interrupted", "unverified")


def executors(
    context: SessionContext,
    operations: dict[str, Callable[..., dict[str, Any]]],
    writer_key: bytes,
) -> dict[str, Callable[..., Any]]:
    """Install motion RPC and the allowed native reads on a dedicated World."""
    from dexsim.kit.mcp import dispatch
    from dexsim.runtime.errors import InvalidParamsError, UnsupportedOperationError

    from . import robot_state

    if len(writer_key) != 32:
        raise ValueError("The motion writer key must contain 32 bytes")

    native_reads = {
        "world.status",
        "scene.list_objects",
        "scene.get_object_info",
        "articulation.get_state",
        "articulation.get_info",
        "simulation.get_rigid_body_state",
    }

    def invoke(owner: SessionContext, params: dict[str, Any]) -> dict[str, Any]:
        if owner is not context:
            raise InvalidParamsError("Motion authority is scoped to the original World")
        module = params.get("module")
        kwargs = params.get("arguments", {})
        if not isinstance(kwargs, dict):
            raise InvalidParamsError("Expected object arguments")
        if module == robot_state.__name__:
            return robot_state.run(context, **kwargs)
        if module == dispatch.__name__:
            method = kwargs.get("method")
            params = kwargs.get("params", {})
            if not isinstance(method, str) or not isinstance(params, dict):
                raise InvalidParamsError("Invalid native observation request")
            if method in native_reads or (
                method == "render.capture"
                and params.get("camera") == "demo"
                and params.get("outputs") == ["rgb"]
                and "cameras" not in params
            ):
                return dispatch.run(context, **kwargs)
        if module != MODULE:
            raise UnsupportedOperationError(
                "Dedicated World allows only motion operations and native observations"
            )
        operation = kwargs.get("operation")
        args = kwargs.get("arguments")
        if (
            not isinstance(operation, str)
            or operation not in operations
            or not isinstance(args, dict)
        ):
            raise InvalidParamsError("Invalid motion operation")
        if not hmac.compare_digest(
            str(kwargs.get("signature", "")),
            sign(writer_key, context.session_id, operation, args),
        ):
            raise InvalidParamsError("Invalid motion writer grant")
        return operations[operation](**args)

    def reject_shutdown(_context: SessionContext, _params: dict[str, Any]) -> None:
        raise UnsupportedOperationError("The World owner controls shutdown")

    return {"adapter.invoke": invoke, "session.shutdown": reject_shutdown}


def run(_context: SessionContext, **_arguments: Any) -> None:
    """Reject invocations on the unrestricted DexSim runtime host."""
    from dexsim.runtime.errors import UnsupportedOperationError

    raise UnsupportedOperationError(
        "Motion operations require the dedicated World owner"
    )
