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

"""Admission for a dedicated owner-thread queue with one configured writer."""

from __future__ import annotations

import fcntl
import os
import tempfile
from pathlib import Path
import hashlib
import hmac
import json
from collections.abc import Callable
from typing import Any, TYPE_CHECKING

from dexsim.runtime.errors import InvalidParamsError, UnsupportedOperationError

if TYPE_CHECKING:
    from dexsim.runtime import Session, SessionContext
    from .control import WorldControl

__all__ = ["writer_executors", "WriterClient"]

_MODULE = __name__
_READ = {"status", "receipt"}
_WRITE = {"target_and_step", "stop"}


def _signature(
    key: bytes, session_id: str, operation: str, arguments: dict[str, Any]
) -> str:
    payload = json.dumps(
        [session_id, operation, arguments],
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode()
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _call_control(
    session: Session, operation: str, arguments: dict[str, Any], *, writer_key: bytes
) -> dict[str, Any]:
    """Invoke a dedicated control operation. On transport loss reconcile by command ID."""
    return session.invoke(
        _MODULE,
        operation=operation,
        arguments=arguments,
        signature=_signature(writer_key, session.session_id, operation, arguments),
    )


def writer_executors(
    controller: WorldControl, writer_key: bytes
) -> dict[str, Callable]:
    """Build the exclusive adapter executor for a dedicated manual host.

    The host installs this table in its sole CommandQueue after scene setup,
    with GUI/other local writers disabled. No script executor or unrestricted
    adapter loader is installed. The host key is never sent or journaled.
    """
    from dexsim.kit.mcp import dispatch
    from . import robot_state

    if len(writer_key) < 32:
        raise ValueError("Writer key must contain at least 32 random bytes")
    native_reads = {
        "world.status",
        "scene.list_objects",
        "articulation.get_state",
        "articulation.get_info",
        "simulation.get_rigid_body_state",
    }

    def invoke(context: SessionContext, params: dict[str, Any]) -> dict[str, Any]:
        if context is not controller.context:
            raise RuntimeError("Writer authority belongs to a different host")
        module = params.get("module")
        kwargs = params.get("arguments", {})
        if module == robot_state.__name__:
            return robot_state.run(context, **kwargs)
        if module == dispatch.__name__ and kwargs.get("method") in native_reads:
            return dispatch.run(context, **kwargs)
        if module != _MODULE:
            raise UnsupportedOperationError(
                "Dedicated writer session rejects other mutations and scripts"
            )
        operation = kwargs.get("operation")
        arguments = kwargs.get("arguments")
        if operation not in _READ | _WRITE or not isinstance(arguments, dict):
            raise InvalidParamsError("Unknown control operation or invalid arguments")
        if not hmac.compare_digest(
            str(kwargs.get("signature", "")),
            _signature(writer_key, context.session_id, operation, arguments),
        ):
            raise InvalidParamsError("Invalid writer grant")
        return getattr(controller, operation)(**arguments)

    def reject_shutdown(
        context: SessionContext, params: dict[str, Any]
    ) -> dict[str, Any]:
        raise UnsupportedOperationError("Only the host owner may shut down this World")

    return {"adapter.invoke": invoke, "session.shutdown": reject_shutdown}


def run(context: SessionContext, **kwargs: Any) -> None:
    """Reject writes on ordinary DexSim Sessions with unrestricted executors."""
    raise UnsupportedOperationError(
        "World control requires a dedicated host writer grant"
    )


class WriterClient:
    """Serialize a local writer grant and recover ambiguous commands across reconnects.

    The operator supplies the same private grant file to each frontend. A
    session-specific outbox beside it records intent before transport I/O.
    """

    def __init__(self, session: Session, key_file: Path) -> None:
        self.session = session
        self.key_file = key_file.resolve(strict=True)
        self.key = self.key_file.read_bytes()
        if len(self.key) != 32:
            raise ValueError("Expected the 32-byte private host writer grant")
        self.outbox = self.key_file.with_name(
            f"{self.key_file.name}.{session.session_id}.pending"
        )

    def call(self, operation: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute or query one authorized operation, fencing uncertain writes."""
        from dexsim.runtime.errors import SessionRemoteError

        with self.key_file.open("rb") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            pending = (
                json.loads(self.outbox.read_bytes()) if self.outbox.exists() else None
            )
            if pending and operation == "target_and_step":
                if pending != arguments:
                    raise RuntimeError(
                        f"Reconcile pending command {pending['command_id']!r} before writing"
                    )
                # Never replay an unacknowledged intent on faith.
                return self._recover(pending["command_id"])
            if (
                operation == "receipt"
                and pending
                and pending["command_id"] == arguments.get("command_id")
            ):
                return self._recover(pending["command_id"])
            if operation != "target_and_step":
                return _call_control(
                    self.session, operation, arguments, writer_key=self.key
                )
            if not arguments.get("command_id"):
                raise ValueError("A target requires a command ID")
            with tempfile.NamedTemporaryFile(
                dir=self.outbox.parent, delete=False
            ) as stream:
                tmp = Path(stream.name)
                try:
                    stream.write(json.dumps(arguments, allow_nan=False).encode())
                    stream.flush()
                    os.fsync(stream.fileno())
                    os.replace(tmp, self.outbox)
                finally:
                    tmp.unlink(missing_ok=True)
            try:
                result = _call_control(
                    self.session, operation, arguments, writer_key=self.key
                )
            except SessionRemoteError as exc:
                # The host returns INVALID_PARAMS for definitive pre-write validation.
                # Other errors can be ambiguous (including owner-thread timeouts).
                if exc.code == "INVALID_PARAMS":
                    self.outbox.unlink()
                raise
            if result["state"] != "effect_unknown":
                self.outbox.unlink()
            return result

    def _recover(self, command_id: str) -> dict[str, Any]:
        receipt = _call_control(
            self.session, "receipt", {"command_id": command_id}, writer_key=self.key
        )
        if receipt["state"] != "effect_unknown":
            self.outbox.unlink()
        return receipt
