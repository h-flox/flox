from torch import nn
from torch.utils.data import DataLoader


class FloxTrainer:
    def __init__(self, max_epochs: int):
        self.max_epochs = max_epochs

    def fit(self, module: nn.Module, train_loader: DataLoader):
        for i, data in enumerate(train_loader):
            inputs, targets = data

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_fn()

    def _training_epoch(self):
        running_loss = 0
        last_loss = 0

        model.train(True)
        for i, data in enumerate(training_loader):
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
