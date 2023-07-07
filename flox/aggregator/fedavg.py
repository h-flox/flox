import lightning as L
import torch

from flox.aggregator import SimpleAggregatorLogic
from flox.worker import WorkerLogicInterface
from numpy.random import RandomState
from typing import Iterable, Optional


class FedAvg(SimpleAggregatorLogic):

    def __init__(
            self,
            participation_frac: float,
            random_state: Optional[RandomState] = None,
            **kwargs
    ):
        if not 0.0 <= participation_frac <= 1.0:
            raise ValueError("Parameter `participation_frac` must be in range [0,1].")
        self.participation_frac = participation_frac
        if random_state is None:
            self.random_state = RandomState()

    def on_worker_select(self, workers: dict[str, WorkerLogicInterface]) -> Iterable[str]:
        size = int(len(workers) * self.participation_frac)
        size = max(1, size)
        choices = self.random_state.choice(list(workers), size=size, replace=False)
        return choices

    def on_module_aggregate(
            self,
            module: L.LightningModule,
            workers: dict[str, WorkerLogicInterface],
            updates: dict[str, L.LightningModule],
            **kwargs
    ) -> dict[str, torch.Tensor]:  # returns a `state_dict`
        # TODO: We need to change the code here such that it works with a "state" that is returned by the worker
        #       for that round. Assuming each worker logic overrides the `__len__` method is a bit inelegant.
        avg_weights = {}
        total_data_samples = sum(len(endp_data) for endp_data in workers.values())
        for w in workers:
            worker_module = updates[w] if w in updates else module
            for name, param in worker_module.state_dict().items():
                coef = len(workers[w]) / total_data_samples
                if name in avg_weights:
                    avg_weights[name] += coef * param.detach()
                else:
                    avg_weights[name] = coef * param.detach()

        return avg_weights
