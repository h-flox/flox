import pytorch_lightning as pl


class FederatedDataModule(pl.LightningDataModule):

    def __init__(self, endpoint_ids: list[str]):
        super().__init__()

    def prepare_distribution(self):
        pass
