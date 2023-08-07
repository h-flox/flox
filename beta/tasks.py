import torch

from collections import defaultdict
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import Any, Mapping, Optional


def local_training_task(
    model: nn.Module,
    train_dataset: Dataset,
    num_epochs: int,
    validation_dataset: Optional[Dataset] = None,
    lr: float = 1e-3,
    batch_size: int = 32,
    shuffle: bool = False,
    record_every: int = 1000,
) -> tuple[nn.Module, Mapping[str, Any]]:
    results = defaultdict(list)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    training_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    for epoch in range(num_epochs):
        model.train(True)
        for i, data in enumerate(training_loader):
            running_loss = 0
            inputs, targets = data

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_fn(outputs, targets)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

            if i % record_every == (record_every - 1):
                last_loss = running_loss / record_every
                results_x = epoch * len(training_loader) + i + 1
                print("\tbatch {} loss: {}".format(i + 1, last_loss))
                results["loss/train"].append(last_loss)
                results["epoch"].append(epoch)
                running_loss = 0.0

        if validation_dataset:
            running_loss = 0.0
            model.eval()
            validation_loader = DataLoader(
                validation_dataset, batch_size=batch_size, shuffle=False
            )

            i = 0
            with torch.no_grad():
                for i, vdata in enumerate(validation_loader):
                    vinputs, vtargets = vdata
                    voutputs = model(vinputs)
                    vloss = loss_fn(voutputs, vtargets)
                    running_loss += vloss.item()

            avg_vloss = running_loss / (i + 1)
            results["loss/validation"].append(avg_vloss)
            results["epoch"].append(epoch)

    return model, results


def aggregation_task(
    node_models: dict[Any, nn.Module], nodel_weights: dict[Any, float]
):
    pass
