import os
from datetime import datetime

import torch
from torch import nn

from flox.federation.topologies import Node

# from flox.runtime.launcher import Launcher, LocalLauncher
# from flox.runtime.runtime import Runtime
from flox.learn import FloxModule
from flox.logger import CSVLogger, Logger, TensorBoardLogger

# from flox.runtime.federation.process_sync_v2 import SyncProcessV2


class MyModule(FloxModule):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        preds = self(inputs)
        loss = nn.functional.cross_entropy(preds, targets)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=1e-3)


def test_csv_protocol():
    logger = CSVLogger()
    assert isinstance(logger, CSVLogger)
    assert isinstance(logger, Logger)


def test_tensorboard_protocol():
    logger = TensorBoardLogger()
    assert isinstance(logger, TensorBoardLogger)
    assert isinstance(logger, Logger)


def test_csv_output_format():
    logger = CSVLogger()
    expect_records = []

    dt1 = datetime.now()
    logger.log("train/loss", 2.3, "node 1", 0, dt1)
    expect_records.append(
        {
            "name": "train/loss",
            "value": 2.3,
            "nodeid": "node 1",
            "epoch": 0,
            "datetime": dt1,
        }
    )
    dt2 = datetime.now()
    logger.log("train/loss", 2.4, "node 2", 0, dt2)
    expect_records.append(
        {
            "name": "train/loss",
            "value": 2.4,
            "nodeid": "node 2",
            "epoch": 0,
            "datetime": dt2,
        }
    )
    dt3 = datetime.now()
    logger.log("train/loss", 1.4, "node 1", 1, dt3)
    expect_records.append(
        {
            "name": "train/loss",
            "value": 1.4,
            "nodeid": "node 1",
            "epoch": 1,
            "datetime": dt3,
        }
    )
    dt4 = datetime.now()
    logger.log("train/loss", 1.2, "node 2", 1, dt4)
    expect_records.append(
        {
            "name": "train/loss",
            "value": 1.2,
            "nodeid": "node 2",
            "epoch": 1,
            "datetime": dt4,
        }
    )

    actual: str = logger.to_pandas(None)
    expected: str = (
        f"name,value,nodeid,epoch,datetime\ntrain/loss,2.3,node 1,0,{dt1}\ntrain/loss,2.4,"
        f"node 2,0,{dt2}\ntrain/loss,1.4,node 1,1,{dt3}\ntrain/loss,1.2,node 2,1,{dt4}"
    )

    assert expected.lower().splitlines() == actual.lower().splitlines()
    assert expect_records == logger.records


"""
def test_csv_flox():

    flock = Topology.from_yaml("./flox/tests/logger/examples/three-level.yaml")


    data = MNIST(
        root=os.environ["TORCH_DATASETS"],
        download=True,
        train=True,
        transform=ToTensor(),
    )

    subset_data = Subset(data, indices=range(len(data) // 1000))

    fs = federated_split(subset_data, flock, 10, samples_alpha=10.0, labels_alpha=10.0)

    logger = CSVLogger(filename="./flox/tests/logger/logs.csv")
    print(fs.dataset)
    print(fs.indices[1])

    module, train_history = federated_fit(
        flock,
        MyModule(),
        fs,
        1,
        strategy="fedavg",
        launcher_kind="thread",
        logger=logger,
    )

    assert True
"""


def test_tensorboard():
    node_worker1 = Node(idx="node1", kind="worker")
    node_worker2 = Node(idx="node2", kind="worker")

    exp_rec1 = []
    exp_rec2 = []

    logger_worker1 = TensorBoardLogger(node_worker1)
    logger_worker2 = TensorBoardLogger(node_worker2)

    import numpy as np

    for step in range(1, 101):
        rand1 = np.random.random()
        dt1 = datetime.now()
        logger_worker1.log("Train/Loss", rand1 / step, "node 1", step, dt1)
        exp_rec1.append(
            {
                "name": "Train/Loss",
                "value": rand1 / step,
                "nodeid": "node 1",
                "epoch": step,
                "datetime": dt1,
            }
        )

        rand2 = np.random.random()
        dt2 = datetime.now()
        logger_worker2.log("Train/Loss", rand2 / step, "node 2", step, dt2)
        exp_rec2.append(
            {
                "name": "Train/Loss",
                "value": rand2 / step,
                "nodeid": "node 2",
                "epoch": step,
                "datetime": dt2,
            }
        )

    assert exp_rec1 == logger_worker1.records
    assert exp_rec2 == logger_worker2.records

    check_dir_made: bool = os.path.exists("./runs")
    assert check_dir_made


def test_switch():
    logger_csv = CSVLogger()
    logger_tb = TensorBoardLogger()

    assert isinstance(logger_csv, Logger) and isinstance(logger_tb, Logger)

    expected = []

    dt1 = datetime.now()
    logger_csv.log("train/loss", 2.3, "node 1", 0, dt1)
    expected.append(
        {
            "name": "train/loss",
            "value": 2.3,
            "nodeid": "node 1",
            "epoch": 0,
            "datetime": dt1,
        }
    )

    new_logger: TensorBoardLogger = logger_csv

    assert expected == new_logger.records
