from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from flox.federation.topologies import WorkerState
    from flox.learn.types import Loss


class TrainerStrategy(t.Protocol):
    def trainer_hparams(self) -> dict[str, t.Any]:
        pass

    def before_backprop(self, state: WorkerState, loss: Loss) -> Loss:
        pass

    def after_backprop(self, state: WorkerState, loss: Loss) -> Loss:
        pass
