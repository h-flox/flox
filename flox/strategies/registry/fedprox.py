import torch

from flox.strategies import FedAvg
from flox.flock.states import FloxWorkerState, FloxAggregatorState


class FedProx(FedAvg):
    """Implementation of FedAvg with Proximal Term.

    This strategy extends ``FedAvg`` and differs from it by computing a "proximal term"
    and adding it to the computed loss during the training step before doing backpropagation.
    This proximal term is the norm difference between the parameters of the global model
    and the worker's locally-updated model. This proximal term is used to make aggregation
    less sensitive to harshly heterogeneous (i.e., non-iid) data distributions across workers.

    More information on the proximal term and its definition can be found in the docstring
    for ``FedProx.wrk_on_after_train_step()`` and in the reference below.

    References:
        Li, Tian, et al. "Federated optimization in heterogeneous networks." Proceedings of
        Machine learning and systems 2 (2020): 429-450.
    """

    def __init__(
        self,
        mu: float = 0.3,
        participation: float = 1.0,
        probabilistic: bool = False,
        always_include_child_aggregators: bool = True,
        seed: int = None,
    ):
        """

        Args:
            mu (float): Multiplier that weights the importance of the proximal term. If `mu=0` then
                ``FedProx`` reduces to ``FedAvg``.
            participation (float): Participation rate for random worker selection.
            probabilistic (bool): Probabilistically chooses workers if True; otherwise will always
                select `max(1, max_workers * participation)` workers.
            always_include_child_aggregators (bool): If True, Will always include child nodes that are
                aggregators; if False, then they are included at random.
            seed (int): Random seed.
        """
        super().__init__(
            participation,
            probabilistic,
            always_include_child_aggregators,
            seed,
        )
        self.mu = mu

    def wrk_on_after_train_step(
        self,
        state: FloxWorkerState,
        loss: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Adds the proximal term before the optimization step during local training to minimize the following
        objective:

        $$
            \\min_{w} h_{k}(w; w^{t}) = F_{k}(w) +
            \\underbrace{\\frac{\\mu}{2} \\|w-w^{t}\\|^{2}}_{\\text{proximal term}}
        $$

        Args:
            state (FloxWorkerState):
            loss (torch.Tensor):
            **kwargs ():

        Returns:

        """
        global_model = state.pre_local_train_model
        local_model = state.post_local_train_model

        params = list(local_model.state_dict().values())
        params0 = list(global_model.state_dict().values())

        norm = torch.sum(
            torch.Tensor(
                [torch.sum((params[i] - params0[i]) ** 2) for i in range(len(params))]
            )
        )

        proximal_term = (self.mu / 2) * norm
        loss += proximal_term
        return loss
