from __future__ import annotations

import typing as t

import torch

from flight.strategies.base import DefaultTrainerStrategy, Strategy

from .fedavg import FedAvgWorker
from .fedsgd import FedSGDAggr, FedSGDCoord

if t.TYPE_CHECKING:
    NodeState: t.TypeAlias = t.Any
    from flight.strategies import Loss

DEVICE = "cpu"


class FedProxTrainer(DefaultTrainerStrategy):
    def __init__(self, mu: float = 0.3):
        self.mu = mu

    def before_backprop(self, state: NodeState, loss: Loss) -> Loss:
        global_model = state.global_model
        local_model = state.local_model
        assert global_model is not None
        assert local_model is not None

        global_model = global_model.to(DEVICE)
        local_model = local_model.to(DEVICE)

        params = list(local_model.state_dict().values())
        params0 = list(global_model.state_dict().values())

        proximal_diff = torch.tensor(
            [
                torch.sum(torch.pow(params[i] - params0[i], 2))
                for i in range(len(params))
            ]
        )
        proximal_term = torch.sum(proximal_diff)
        proximal_term = proximal_term * self.mu / 2

        proximal_term = proximal_term.to(DEVICE)

        loss += proximal_term
        return loss


class FedProx(Strategy):
    def __init__(
        self,
        mu: float = 0.3,
        participation: float = 1.0,
        probabilistic: bool = False,
        always_include_child_aggregators: bool = False,
    ):
        super().__init__(
            coord_strategy=FedSGDCoord(
                participation,
                probabilistic,
                always_include_child_aggregators,
            ),
            aggr_strategy=FedSGDAggr(),
            worker_strategy=FedAvgWorker(),
            trainer_strategy=FedProxTrainer(mu=mu),
        )
