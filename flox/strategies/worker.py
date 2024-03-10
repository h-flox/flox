from __future__ import annotations

import typing as t

from flox.flock.states import WorkerState

if t.TYPE_CHECKING:
    from flox.runtime import JobResult


class WorkerStrategy(t.Protocol):
    def work_start(self, state: WorkerState) -> WorkerState:
        return state

    def before_training(
        self, state: WorkerState, data: t.Any
    ) -> tuple[WorkerState, t.Any]:
        return state, data

    def after_training(self, state: WorkerState) -> WorkerState:
        return state

    def work_end(self, result: JobResult) -> JobResult:
        return result
