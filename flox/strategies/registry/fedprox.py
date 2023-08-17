import torch

from flox.aggregator.state import AggregatorState
from flox.strategies import Strategy, FedAvg
from flox.worker.state import FloxWorkerState


class FedProx(FedAvg):
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
            mu ():
            participation ():
            probabilistic ():
            always_include_child_aggregators ():
            seed ():
        """
        super().__init__(
            participation,
            probabilistic,
            always_include_child_aggregators,
            seed,
        )
        self.mu = mu

    def on_after_train_step(
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
        local_model = state.current_model
        global_model = state.global_model

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
