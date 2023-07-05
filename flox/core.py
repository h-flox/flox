import os
import torch
import lightning as L

from numpy.random import RandomState
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from typing import Optional

from flox.aggregator.base import AggregatorLogic
from flox.worker.base import WorkerLogic

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64


def _local_fit(
        logic: WorkerLogic,
        module: L.LightningModule
):
    data_loader = DataLoader(logic.on_data_fetch(), batch_size=32, shuffle=True)
    module = logic.on_module_fit(module, data_loader)
    return module


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
    module = trainer.fit(module, data_loader)
    return endp_id, module


def fit(
        endpoints,
        module: L.LightningModule,
        global_rounds: int,
        aggr_logic: AggregatorLogic,
        trainer_logic: WorkerLogic,
        random_state: Optional[RandomState] = None,
        **kwargs
):
    if random_state is None:
        random_state = RandomState()
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
        metrics = trainer.test(module, DataLoader(mnist_test_data))
        print(metrics)
