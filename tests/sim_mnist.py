import lightning as L
import os
import random
import torch
import torch.nn.functional as F
import unittest

from flox.executor.local import LocalExec
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")


class MnistModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x: torch.Tensor):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
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


class MnistSimulationTest(unittest.TestCase):

    def test_local_fl(self):

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

        random.seed(1)
        executor = LocalExec()
        # aggr_logic
        worker_logic = None
        aggr_rounds = 3
        n_workers = 5
        mnist_train_data = MNIST(
            PATH_DATASETS,
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
        workers = {
            endp: torch.utils.data.RandomSampler(
                mnist_train_data,
                replacement=False,
                num_samples=random.randint(10, 250)
            )
            for endp in range(n_workers)
        }

        global_module = MnistModule()  # aggr_logic.on_model_init()
        for ar in range(aggr_rounds):
            ar_workers = workers  # aggr_logic.on_worker_selection(workers)
            futures = executor.submit_jobs(workers=ar_workers, logic=worker_logic, module=global_module,
                                           dataset=mnist_train_data)
            results = [fut.result() for fut in futures]
            module_updates = {worker: module for (worker, module) in results}

            aggr_weights = fedavg(global_module, module_updates, workers)  # aggr_logic(global_module, results)
            global_module.load_state_dict(aggr_weights)
            test_metrics = aggr_logic.on_model_eval(global_module)

        self.assertTrue(isinstance(global_module, L.LightningModule))
