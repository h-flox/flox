import pytorch_lightning as pl
import torch.nn.functional as F
from flox import *

# Define the neural network module that you wish to train.
class LitClassifier(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

if __name__ == "__main__":
    module = LitClassifier()
    flx = FloxExec()
    flx.fit(
        endpoint_ids=[...],
        module=module,
        endpoint_logic=EndpointLogic(),
        controller_logic=ControllerLogic(),
        **kwargs
    )