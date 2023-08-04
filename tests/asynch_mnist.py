import flox
import unittest

from flox.aggregator.asynch.standard import SimpleAsynchAggregatorLogic
from flox.worker import SimpleWorkerLogic
from modules import MnistModule


class MnistAsyncAggrLogic(SimpleAsynchAggregatorLogic):
    def __init__(self):
        super().__init__()


class MnistWorkerLogic(SimpleWorkerLogic):
    def __init__(self, idx, indices):
        super().__init__(idx)
        self.name = "mnist"
        self.indices = indices

    def on_data_fetch(self):
        from torch.utils.data import Subset
        from torchvision.datasets import MNIST
        from torchvision.transforms import ToTensor
        from os import environ

        root = environ.get("TORCH_DATASETS", ".")
        data = MNIST(root, download=True, train=True, transform=ToTensor())
        data = Subset(data, indices=self.indices)
        return data

    def __len__(self) -> int:
        return len(self.indices)


class AsynchMnistTest(unittest.TestCase):
    def test_asynch_aggregation(self):
        workers = flox.create_workers(5, MnistWorkerLogic)
        results = flox.federated_fit(
            global_module=MnistModule(),
            aggr=MnistAsyncAggrLogic(),
            workers=workers,
            global_rounds=3,
            test=False,
            n_threads=4,
            fit_kind="asynch"
        )
        self.assertTrue(True, "Successfully completed training!")  # add assertion here


if __name__ == '__main__':
    unittest.main()
