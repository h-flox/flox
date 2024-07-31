from __future__ import annotations

import typing as t

import torch

from flight.strategies.base import DefaultTrainerStrategy, Strategy

from .fedavg import FedAvgWorker
from .fedsgd import FedSGDAggr, FedSGDCoord

if t.TYPE_CHECKING:
    from flight.strategies import Loss, NodeState

DEVICE = "cpu"


class FedProxTrainer(DefaultTrainerStrategy):
    """The coordinator and its respective methods for 'FedProx'.

    Extends:
        DefaultTrainerStrategy: The base class providing necessary
        methods for 'FedProxTrainer'.
    """

    def __init__(self, mu: float = 0.3):
        self.mu = mu

    def before_backprop(self, state: NodeState, loss: Loss) -> Loss:
        """Callback to run before backpropagation.

        Args:
            state (NodeState): The state of the current node.
            loss (Loss): The calculated loss associated with the current node.

        Returns:
            Loss: The updated loss associated with the current node.
        """
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
    """
    Implementation of FedAvg with Proximal Term.

    This strategy extends ``FedAvg`` and differs from it by computing a "proximal term"
    and adding it to the computed loss during the training step before doing
    backpropagation. This proximal term is the norm difference between the parameters
    of the global model and the worker's locally-updated model. This proximal term is
    used to make aggregation less sensitive to harshly heterogeneous (i.e., non-iid)
    data distributions across workers.

    More information on the proximal term and its definition can be found in the
    docstring for ``FedProx.wrk_after_train_step()`` and in the reference below.

    References:
        Li, Tian, et al. "Federated optimization in heterogeneous networks."
        Proceedings of Machine learning and systems 2 (2020): 429-450.
    """

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
