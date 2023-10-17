import torch


class FloxModule(torch.nn.Module):
    """
    The ``FloxModule`` is a wrapper for the standard ``torch.nn.Module`` class from PyTorch, with
    a lot of inspiration from the ``lightning.LightningModule`` class from PyTorch Lightning.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        """

        Args:
            batch ():
            batch_idx ():

        Returns:

        """

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configures, initializes, and returns the optimizer used to train the model.

        Returns:
            The optimizer used to train the model.
        """

    def validation_step(self, batch, batch_idx):
        """

        Args:
            batch ():
            batch_idx ():

        Returns:

        """

    def test_step(self, batch, batch_idx):
        """

        Args:
            batch ():
            batch_idx ():

        Returns:

        """

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        """

        Args:
            batch ():
            batch_idx ():
            dataloader_idx ():

        Returns:

        """
