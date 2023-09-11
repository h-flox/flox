import torch


class FloxModule(torch.nn.Module):
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
        """

        Returns:

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

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """

        Args:
            batch ():
            batch_idx ():
            dataloader_idx ():

        Returns:

        """
