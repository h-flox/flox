import numpy as np
import os

from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional

from flox.flock import Flock

TEST_DIR = ".tmp_test_dirs"


def populate(seed: Optional[int] = None):
    rand_state = np.random.RandomState(seed)
    for client in range(CLIENTS):
        client_path = Path(f"./{client}/data.txt")
        with open(client_path, "w") as file:
            num_samples = rand_state.randint(low=1, high=1000)
            for _ in range(num_samples):
                a = rand_state.randint(low=-1000, high=1000)
                b = rand_state.randint(low=-1000, high=1000)
                c = a + b
                print(f"{a} {b} {c}", file=file)
    print(">>> Done populating data")


def setup(flock: Flock):
    rand_state = np.random.RandomState(1)
    for worker in flock.workers:
        client_data_path = Path(f"./{TEST_DIR}/{worker.idx}/data.txt")
        with open(client_data_path, "w") as file:
            num_samples = rand_state.randint(low=1, high=1000)
            for _ in range(num_samples):
                a = rand_state.randint(low=-1000, high=1000)
                b = rand_state.randint(low=-1000, high=1000)
                c = a + b
                print(f"{a} {b} {c}", file=file)


def teardown():
    os.system(f"rm -r ./{TEST_DIR}")


def test_load():
    flock = Flock.from_yaml("examples/flocks/2-tier.yaml")
    setup(flock)

    class DataInDirs(Dataset):
        def __init__(self):
            super().__init__()
            pass

        def __getitem__(self, idx: int) -> tuple:
            pass

        def __len__(self) -> int:
            pass

    os.system("tree")
    # teardown(flock)
    assert False
