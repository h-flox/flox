from __future__ import annotations

import typing as t

from flox.flock.states import WorkerState

if t.TYPE_CHECKING:
    import torch

    from flox.runtime import JobResult


class WorkerStrategy(t.Protocol):
    def work_start(self, state: WorkerState) -> WorkerState:
        pass

    def before_training(
        self, state: WorkerState, data: t.Any
    ) -> tuple[WorkerState, t.Any]:
        pass

    def after_training(
        self, state: WorkerState, optimizer: torch.optim.Optimizer
    ) -> WorkerState:
        pass

    def work_end(self, result: JobResult) -> JobResult:
        pass
