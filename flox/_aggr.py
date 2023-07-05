import lightning as L
import torch


class AggrLogic:
    def on_model_broadcast(self):
        pass

    def on_model_aggr(
            self,
            modules: dict[str, L.LightningModule],
            **kwargs
    ) -> torch.Tensor:
        pass

    def on_model_eval(self):
        pass
