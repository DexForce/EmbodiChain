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

"""Host-driven, non-blocking execution of a compiled skill sequence.

:meth:`~embodichain.lab.visualization.authoring.session.AuthoringSession.execute`
runs a whole trajectory inside one call. That is convenient for a script, but a
browser-facing host loop cannot publish anything while it blocks: the card
states only reach the panel once the last waypoint is done, and the viser scene
freezes for the duration of the run.

:class:`StepwiseExecution` wraps the session's
:meth:`~embodichain.lab.visualization.authoring.session.AuthoringSession.execute_stepwise`
iterator so a host loop can own the pacing. Each :meth:`StepwiseExecution.advance`
call performs a bounded number of simulation updates and returns, letting the
caller publish a panel state and capture a visualization frame in between. The
simulation-side behavior is identical to a blocking run; only who drives the
loop changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterator

from .protocol import ExecutionProgress

if TYPE_CHECKING:
    from .session import AuthoringSession

__all__ = ["StepwiseExecution"]


class StepwiseExecution:
    """Drive one compiled sequence forward under host-loop control.

    Constructing the driver validates the session's compilation and resets the
    card states of every compiled card, exactly like the start of a blocking
    :meth:`~embodichain.lab.visualization.authoring.session.AuthoringSession.execute`
    call. No simulation update happens until :meth:`advance` is called.

    Every :meth:`advance` tick steps physics, so the host loop must not step
    the simulation itself while :attr:`is_active` is true.

    Args:
        session: Authoring session holding a successful compilation.
        on_step: Optional callback invoked as ``on_step(step_index,
            total_steps)`` after every trajectory simulation update.
        hold_steps: Number of final-pose hold updates after the last waypoint.

    Raises:
        RuntimeError: If the session has no successful compilation.
        ValueError: If ``hold_steps`` is negative.
    """

    def __init__(
        self,
        session: AuthoringSession,
        on_step: Callable[[int, int], None] | None = None,
        *,
        hold_steps: int = 0,
    ) -> None:
        self._steps: Iterator[ExecutionProgress] = session.execute_stepwise(
            on_step,
            hold_steps=hold_steps,
        )
        self._progress: ExecutionProgress | None = None
        self._finished = False
        self._succeeded = False

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    @property
    def progress(self) -> ExecutionProgress | None:
        """Newest progress value, or ``None`` before the first tick."""
        return self._progress

    @property
    def is_active(self) -> bool:
        """Whether more simulation updates are still pending."""
        return not self._finished

    @property
    def is_finished(self) -> bool:
        """Whether the run ended, either successfully, failed, or closed."""
        return self._finished

    @property
    def succeeded(self) -> bool:
        """Whether a finished run replayed every waypoint without raising."""
        return self._succeeded

    # ------------------------------------------------------------------
    # Driving
    # ------------------------------------------------------------------

    def advance(self, count: int = 1) -> ExecutionProgress | None:
        """Perform up to ``count`` simulation updates and report the last one.

        Args:
            count: Maximum number of ticks to perform in this call.

        Returns:
            The newest progress value, or ``None`` when the run finished
            during this call or had already finished.

        Raises:
            ValueError: If ``count`` is not at least one.
        """
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("count must be an integer.")
        if count < 1:
            raise ValueError("count must be at least one.")
        if self._finished:
            return None
        progress: ExecutionProgress | None = None
        for _ in range(count):
            try:
                progress = next(self._steps)
            except StopIteration as stop:
                self._finished = True
                self._succeeded = bool(stop.value)
                return None
            self._progress = progress
        return progress

    def run_to_completion(self) -> bool:
        """Advance until the run ends and return whether it succeeded."""
        while not self._finished:
            self.advance()
        return self._succeeded

    def close(self) -> None:
        """Abandon a running execution without advancing it further.

        The simulation keeps whatever state the last completed tick produced
        and the card states are left untouched, so the panel still shows which
        card was running when the run was abandoned.
        """
        if self._finished:
            return
        self._finished = True
        close = getattr(self._steps, "close", None)
        if close is not None:
            close()
