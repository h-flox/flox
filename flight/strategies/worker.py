from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    import torch
    from flight.federation.jobs.result import Result

    NodeState: t.TypeAlias = t.Any

@t.runtime_checkable
class WorkerStrategy(t.Protocol):
    def start_work(self, state: NodeState) -> NodeState:
        pass

    def before_training(self, state: NodeState, data: t.Any) -> tuple[NodeState, t.Any]:
        pass

    def after_training(
        self, state: NodeState, optimizer: torch.optim.Optimizer
    ) -> NodeState:
        pass

    def end_work(self, result: Result) -> Result:
        pass
