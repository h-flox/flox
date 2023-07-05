import pytorch_lightning as pl
import torch

from torch.utils.data import Dataset


class FederatedDataset(Dataset):
    def __init__(self, dataset: Dataset, indices: list[int]):
        pass


class FederatedDataModule(pl.LightningDataModule):

    def __init__(self, endpoint_ids: list[str]):
        super().__init__()

    def prepare_distribution(self):
        pass
