import datetime
from collections import defaultdict
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def local_fitting_task(
    worker,
    worker_dataset: Dataset,
    module_cls: type[nn.Module],
    module_state_dict: dict[str, torch.Tensor],
    batch_size: int = 4,
    max_epochs: int = 2,
    device: Optional[torch.device] = None,
):
    module = module_cls()
    module.to(device)
    module.load_state_dict(module_state_dict)

    history = defaultdict(list)
    optimizer = torch.optim.SGD(module.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = DataLoader(worker_dataset, batch_size=batch_size)

    for epoch in range(max_epochs):
        running_loss, last_loss = 0.0, 0.0
        for batch_nb, batch in enumerate(train_loader):
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = module(inputs)

            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        history["worker"].append(worker)
        history["train/loss"].append(running_loss / len(train_loader))
        history["epoch"].append(epoch)
        history["time"].append(str(datetime.datetime.now()))

    return {
        "worker": worker,
        "module_state_dict": module.state_dict(),
        "local_history": history,
    }


def evaluate(
    module: nn.Module,
    test_dataloader: DataLoader,
    device: Optional[torch.device] = None,
):
    if device is None:
        device = torch.device("cpu")

    loss_fn = nn.CrossEntropyLoss()
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    module.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for batch in test_dataloader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = module(inputs)
            test_loss += loss_fn(outputs, targets).item()
            correct += (outputs.argmax(1) == targets).sum().item()

    test_loss /= num_batches
    correct /= size

    return {"test_loss": test_loss, "test_acc": correct}
