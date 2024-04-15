from __future__ import annotations

import typing as t

import torch

from flox.const import DEVICE
from flox.strategies import Strategy
from flox.strategies.impl.fedavg import FedAvgWorker
from flox.strategies.impl.fedsgd import FedSGDClient, FedSGDAggr
from flox.strategies.strategy import DefaultTrainerStrategy

if t.TYPE_CHECKING:
    from flox.flock import WorkerState
    from flox.nn.typing import Loss


class FedProxTrainer(DefaultTrainerStrategy):
    def __init__(self, mu: float = 0.3):
        self.mu = mu

    def before_backprop(self, state: WorkerState, loss: Loss) -> Loss:
        """
        Adds the proximal term before the optimization step during local training to minimize the following
        objective:

        $$
            \\min_{w} h_{k}(w; w^{t}) = F_{k}(w) +
            \\underbrace{\\frac{\\mu}{2} \\|w-w^{t}\\|^{2}}_{\\text{proximal term}}
        $$

        Args:
            state (WorkerState):
            loss (Loss):
            **kwargs ():

        Returns:
            Loss with the proximal term added to it.
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
            # , requires_grad=True
        )
        proximal_term = torch.sum(proximal_diff)
        proximal_term = proximal_term * self.mu / 2

        # proximal_term = sum([
        #     torch.sum(torch.pow(params[i] - params0[i], 2))
        #     for i in range(len(params))
        # ])

        # Ensure they're on the same device.
        proximal_term = proximal_term.to(DEVICE)

        loss += proximal_term
        return loss


class FedProx(Strategy):
    """Implementation of FedAvg with Proximal Term.

    This strategy extends ``FedAvg`` and differs from it by computing a "proximal term"
    and adding it to the computed loss during the training step before doing backpropagation.
    This proximal term is the norm difference between the parameters of the global model
    and the worker's locally-updated model. This proximal term is used to make aggregation
    less sensitive to harshly heterogeneous (i.e., non-iid) data distributions across workers.

    More information on the proximal term and its definition can be found in the docstring
    for ``FedProx.wrk_after_train_step()`` and in the reference below.

    References:
        Li, Tian, et al. "Federated optimization in heterogeneous networks." Proceedings of
        Machine learning and systems 2 (2020): 429-450.
    """

    def __init__(
        self,
        mu: float = 0.3,
        participation: float = 1.0,
        probabilistic: bool = False,
        always_include_child_aggregators: bool = False,
    ):
        super().__init__(
            client_strategy=FedSGDClient(
                participation,
                probabilistic,
                always_include_child_aggregators,
            ),
            aggr_strategy=FedSGDAggr(),
            worker_strategy=FedAvgWorker(),
            trainer_strategy=FedProxTrainer(mu=mu),
        )
