from flox.flock import Flock, FlockNode
from flox.strategies.base import Loss, Strategy
from flox.strategies.building_blocks.worker_selection import random_worker_selection
from flox.strategies.registry.fedsgd import FedSGD
from flox.typing import StateDict


class FedAvg(FedSGD):
    """Implementation of the Federated Averaging algorithm.

    This algorithm extends ``FedSGD`` and differs from it by performing a weighted
    average based on the number of data samples each (sibling) worker has. Worker
    selection is done randomly, same as ``FedSGD``.

    > **Reference:**
    >
    > McMahan, Brendan, et al. "Communication-efficient learning of deep networks
    > from decentralized data." Artificial intelligence and statistics. PMLR, 2017.
    """

    def __init__(
        self,
        participation: float = 1.0,
        probabilistic: bool = True,
        always_include_child_aggregators: bool = True,
        seed: int = None,
    ):
        """

        Args:
            participation (float): Participation rate for random worker selection.
            probabilistic (bool): Probabilistically chooses workers if True; otherwise will always
                select `max(1, num_workers * participation)` workers.
            always_include_child_aggregators (bool): If True, Will always include child nodes that are
                aggregators; if False, then they are included at random.
            seed (int): Random seed.
        """
        super().__init__(
            participation, probabilistic, always_include_child_aggregators, seed
        )

    def agg_on_param_aggregation(self, state_dicts, *args, **kwargs):
        pass
