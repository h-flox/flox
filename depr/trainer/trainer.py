import torch
from torch import nn


class Trainer:
    def fit(self, model: nn.Module):
        model.train()
        torch.set_grad_enabled(True)

        losses = []
        for batch in train_dataloader:
            # calls hooks like this one
            on_train_batch_start()

            # train step
            loss = training_step(batch)

            # clear gradients
            optimizer.zero_grad()

            # backward
            loss.backward()

            # update parameters
            optimizer.step()

            losses.append(loss)
