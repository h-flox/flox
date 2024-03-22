from flox.nn import FloxModule


def test_model(module: FloxModule) -> tuple[float, float]:
    import os
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as transforms

    from torch.utils.data import DataLoader
    from torchvision.datasets import FashionMNIST

    data = FashionMNIST(
        root=os.environ["TORCH_DATASETS"],
        download=False,
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
            ]
        ),
    )
    dataloader = DataLoader(data)
    # accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)

    with torch.no_grad():
        num_samples = 0
        running_loss = 0.0
        running_acc = 0.0
        for batch_idx, batch in enumerate(dataloader):
            inputs, targets = batch
            preds = module(inputs)
            assert isinstance(preds, torch.Tensor) and isinstance(targets, torch.Tensor)
            loss = F.cross_entropy(preds, targets)
            pred_label = preds.topk(1).indices[0]
            acc = torch.sum(pred_label == targets)
            running_loss += loss.item()
            running_acc += acc.item()
            num_samples += len(targets)

    final_acc = running_acc / num_samples
    final_loss = running_loss / num_samples
    # print(f"{running_acc=}, {running_loss=}  |  {num_samples=}")
    # exit(0)
    # print(f"{final_acc=}, {final_loss=}")
    return final_acc, final_loss
