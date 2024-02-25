from pathlib import Path

import pandas as pd
import torch
from sklearn.datasets import make_classification

# TODO: Get rid of `sklearn` as a dependency.
from torch.utils.data import Dataset

from flox.data import LocalDataset
from flox.data import federated_split, FederatedSubsets
from flox.flock import Flock
from flox.flock.states import NodeState


##################################################################################################################


class MyDataDir(LocalDataset):
    def __init__(self, state: NodeState, csv_dir: Path):
        super().__init__(state)
        csv_path = csv_dir / f"{state.idx}" / "data.csv"
        self.data = pd.read_csv(csv_path)
        self.csv_path = csv_path

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        x = torch.tensor([[row.x1], [row.x2]])
        y = torch.tensor([row.y])
        return x, y

    def __len__(self):
        return len(self.data)


"""
def test_dir_datasets(tmpdir):
    data_dir = tmpdir
    flock = Flock.from_yaml("examples/flocks/2-tier.yaml")
    rand_state = np.random.RandomState(1)

    for worker in flock.workers:
        client_dir = (data_dir / f"{worker.idx}").mkdir()
        client_path = client_dir / "data.csv"
        with open(client_path, "w") as file:
            print("x1, x2, y", file=file)
            num_samples = rand_state.randint(low=1, high=1000)
            for _ in range(num_samples):
                a = rand_state.randint(low=-1000, high=1000)
                b = rand_state.randint(low=-1000, high=1000)
                c = a + b
                print(f"{a}, {b}, {c}", file=file)

    print("Files:")
    for dir_path, dir_names, filenames in os.walk(data_dir):
        if len(dir_names) == 0:
            assert (
                len(filenames) == 1
            ), f"Error generating data for test. Should only be 1 `data.csv` for worker {dir_path[-1]}."
            path = Path(dir_path) / filenames[0]
            data = pd.read_csv(path)
            print(data.head())

    for worker in flock.workers:
        state = FloxWorkerState(worker.idx, None, None)
        try:
            worker_data = MyDataDir(state, tmpdir)
            assert isinstance(worker_data, Dataset)
        except FileNotFoundError:
            assert False
"""


##################################################################################################################


class MyRandomDataset(Dataset):
    def __init__(self, n_classes: int):
        super().__init__()
        data_sk = make_classification(
            n_samples=100, n_features=20, n_classes=n_classes, random_state=1
        )
        x_sk, y_sk = data_sk
        x_torch = torch.from_numpy(x_sk)
        y_torch = torch.from_numpy(y_sk)
        y_torch = y_torch.unsqueeze(dim=-1)
        self.data_torch = torch.hstack((x_torch, y_torch))

    def __getitem__(self, idx: int):
        datum = self.data_torch[idx]
        x, y = datum[:-1], datum[-1]
        y = y.to(torch.int)
        return x, y

    def __len__(self):
        return len(self.data_torch)


def test_fed_subsets():
    flock = Flock.from_yaml("examples/flocks/2-tier.yaml")
    data = MyRandomDataset(n_classes=2)
    fed_data = federated_split(
        data, flock, num_classes=2, samples_alpha=1.0, labels_alpha=1.0
    )
    assert isinstance(fed_data, (dict, FederatedSubsets))
