import lightning as L
import torch

from flox._worker import WorkerLogic


class AggrLogic:
    def on_model_broadcast(self):
        pass

    def on_model_aggr(
            self,
            module: L.LightningModule,
            workers: dict[str, WorkerLogic],
            updates: dict[str, L.LightningModule],
            **kwargs
    ) -> dict[str, torch.Tensor]:  # returns a `state_dict`
        pass

    def on_module_eval(self, module: L.LightningModule):
        pass


class FedAvg(AggrLogic):
    def on_model_aggr(
            self,
            module: L.LightningModule,
            workers: dict[str, WorkerLogic],
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
