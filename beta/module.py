from torch import nn
from typing import Any, Mapping


class FloxModule(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def setup(self):
        raise NotImplementedError()

    def training_step(self, batch, batch_nb) -> float:
        raise NotImplementedError()

    def validation_step(self) -> float:
        raise NotImplementedError()

    def test_step(self) -> float:
        raise NotImplementedError()

    def predict_step(self):
        raise NotImplementedError()

    def configure_optimizers(self):
        raise NotImplementedError()

    def log(self, name: str, value: Any, prog_bar: bool = False):
        pass

    def log_dict(self, dictionary: Mapping[str, Any]):
        for name, value in dictionary.items():
            self.log(name, value)


if __name__ == "__main__":
    import torch.nn.functional as F
    import torch.optim as optim

    class MyModule(FloxModule):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(28 * 28, 10)

        def forward(self, x):
            return F.relu(self.fc1(x))

        def training_step(self, batch, batch_nb) -> float:
            inputs, targets = batch
            loss = F.cross_entropy(inputs, targets)
            self.log("train/loss", loss)
            return loss

        def configure_optimizers(self):
            return optim.Adam(self.parameters(), lr=1e-3)

    module = MyModule()
