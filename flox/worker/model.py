import torch


class FloxModel(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    #################################################################################

    def forward(self, x):
        raise NotImplementedError("`forward()` must be override.")

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        raise NotImplementedError("`training_step()` must be overriden.")

    def configure_optimizers(self) -> torch.optim.optimizer:
        raise NotImplementedError("`configure_optimizer()` must be overriden.")

    #################################################################################

    def validation_step(self):
        pass

    def test_step(self):
        pass

    def predict_step(self):
        pass
