import argparse
import copy
import lightning as L
import os
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from numpy.random import RandomState
from concurrent.futures import ThreadPoolExecutor
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST

"""
Helpful thread for a environmental issue with OpenMP:
https://github.com/pytorch/pytorch/issues/44282
"""
PATH_DATASETS = os.environ.get("PATH_DATASETS", "..")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64


class FederatedSampler(torch.utils.data.Sampler):
    # TODO: This does not do anything but I think it is a way we can do the federated splitting we discussed earlier.
    def __iter__(self):
        pass

    def __len__(self):
        pass


class MnistModule(L.LightningModule):
    """The neural network used to be trained in this simple example. Of course, this is a VERY simple model."""

    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x: torch.Tensor):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb, **kwargs):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        print(kwargs.get("lol", "This didn't pass through..."))
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


def local_fit(
        endp_id: int,
        module: L.LightningModule,
        data_loader: DataLoader
):
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=3
    )
    trainer.fit(module, data_loader,
                lol="This made it!!")
    return endp_id, module


def fedavg(
        module: L.LightningModule,
        updates: dict[int, L.LightningModule],
        endpoints: dict[int, torch.utils.data.Sampler]
):
    avg_weights = {}
    total_data_samples = sum(len(endp_data) for endp_data in endpoints.values())
    for endp in endpoints:
        if endp in updates:
            endp_module = updates[endp]
        else:
            endp_module = module
        for name, param in endp_module.state_dict().items():
            coef = len(endpoints[endp]) / total_data_samples
            if name in avg_weights:
                avg_weights[name] += coef * param.detach()
            else:
                avg_weights[name] = coef * param.detach()

    return avg_weights


def main(args):
    # This header code simply sets up the FL process: loading in the
    # training/testing data, initializing the neural net, initializing
    # the random seed, and splitting up the across the "endpoints".
    random_state = RandomState(args.seed)
    module = MnistModule()
    global_rounds = 10
    mnist_train_data = MNIST(
        PATH_DATASETS,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    mnist_test_data = MNIST(
        PATH_DATASETS,
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    endpoints = {
        endp: torch.utils.data.RandomSampler(
            mnist_train_data,
            replacement=False,
            num_samples=random_state.randint(10, 250)
        )
        for endp in range(args.endpoints)
    }

    # Below is the execution of the Global Aggregation Rounds. Each round consists of the following steps:
    #   (1) clients are selected to do local training
    #   (2) selected clients do local training and send back their locally-trained model udpates
    #   (3) the aggregator then aggregates the model updates using FedAvg
    #   (4) the aggregator tests/evaluates the new global model
    #   (5) the loop repeats until all global rounds have been done.
    for gr in range(global_rounds):
        print(f">> Starting global round ({gr + 1}/{global_rounds}).")

        # Perform random client selection and submit "local" fitting tasks.
        size = max(1, int(args.participation_frac * len(endpoints)))
        selected_endps = random_state.choice(list(endpoints), size=size, replace=False)
        futures = []
        with ThreadPoolExecutor(max_workers=size) as exc:
            for endp in selected_endps:
                fut = exc.submit(
                    local_fit,
                    endp,
                    copy.deepcopy(module),
                    DataLoader(mnist_train_data, sampler=endpoints[endp], batch_size=args.batch_size)
                )
                print("Job submitted!")
                futures.append(fut)

        # Retrieve the "locally" updated the models and do aggregation.
        updates = [fut.result() for fut in futures]
        updates = {endp: module for (endp, module) in updates}
        avg_weights = fedavg(module, updates, endpoints)
        module.load_state_dict(avg_weights)

        # Evaluate the global model performance.
        trainer = L.Trainer()
        # metrics = trainer.test(module, DataLoader(mnist_test_data))
        # print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", default=123, type=int)
    parser.add_argument("-e", "--endpoints", default=10, type=int)
    parser.add_argument("-c", "--participation_frac", default=0.25, type=float)
    parser.add_argument("-b", "--batch_size", default=BATCH_SIZE, type=int)
    main(parser.parse_args())
