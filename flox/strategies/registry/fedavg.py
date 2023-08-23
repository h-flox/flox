from flox.flock import Flock, FlockNode
from flox.strategies.base import Loss, Strategy
from flox.strategies.commons import random_worker_selection
from flox.typing import StateDict


class FedAvg(Strategy):
    """
    One of the earliest Federated Learning algorithms.

    > **Reference:**
    >
    > McMahan, Brendan, et al. "Communication-efficient learning of deep networks
    > from decentralized data." Artificial intelligence and statistics. PMLR, 2017.
    """

    def __init__(
        self,
        participation: float = 1.0,
        probabilistic: bool = False,
        always_include_child_aggregators: bool = True,
        seed: int = None,
    ):
        """

        Args:
            participation ():
            probabilistic ():
            always_include_child_aggregators ():
            seed ():
        """
        super().__init__()
        self.participation = participation
        self.probabilistic = probabilistic
        self.always_include_child_aggregators = always_include_child_aggregators
        self.seed = seed

    def agg_on_worker_selection(
        self, children: list[FlockNode], **kwargs
    ) -> list[FlockNode]:
        """

        Args:
            children ():
            **kwargs ():

        Returns:
            list[FlockNode]
        """
        return random_worker_selection(
            children,
            participation=self.participation,
            probabilistic=self.probabilistic,
            always_include_child_aggregators=self.always_include_child_aggregators,
            seed=self.seed,  # TODO: Change this because it will always do the same thing as is.
        )

    def agg_on_param_aggregation(self, state_dicts, *args, **kwargs):
        pass
