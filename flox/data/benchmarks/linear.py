import numpy as np
import torch
import pytorch_lightning as pl

from numpy.random import RandomState
from torch.utils.data import Dataset
from typing import Optional


class LinearDataset(Dataset):
    def __init__(
            self,
            n_samples: int,
            m_std,
            m_var,
            b_std,
            b_var,
            random_state: Optional[RandomState] = None
    ) -> None:
        if random_state is None:
            random_state = RandomState()

        avg = [0, 0, 0]
        cov = np.array([[6, -3, -2], [-3, 3.5, 2], [1, 2, 3]])
        points = random_state.multivariate_normal(avg, cov, size=n_samples)
        self.x = torch.from_numpy(points[:, 0])
        self.y = torch.from_numpy(points[:, 1])


class LinearDataModule(pl.LightningDataModule):
    def __init__(
            self,
            n_samples: int,
            m_std,
            m_var,
            b_std,
            b_var
    ) -> None:
        super().__init__()
