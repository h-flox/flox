# Deep Learning in Flight

Here we discuss the process of deep learning is implemented in Flight.
Deep Learning in Flight is done using PyTorch and PyTorch-Ignite.

## PyTorch-Ignite

<img src="../../graphics/ignite_logo.png" alt="PyTorch-Ignite Logo" width="400"/>

[PyTorch-Ignite](https://pytorch.org/ignite/) is a high-level library to
help with training neural networks in PyTorch.

### Why Ignite?

There are a few reasons:

- **Maintainers**: Ignite is maintained by the PyTorch team.
- **Simplicity**: Ignite provides a simple API for training and evaluating neural
  networks.
- **Flexibility**: Ignite is highly customizable and can be used to implement
  a wide variety of training loops. Ignites' `Events` system allows you to
  directly program your own logic into the training loop with minimal effort.
- **Performance**: Ignite is designed to be fast and efficient. Unlike
  the widely-used PyTorch Lightning framework, Ignite is light and makes fewer
  assumptions about *how* you are training your model (i.e., `lightning` is well-catered
  to training a single model on big machines, rather than several models in FL).

### Ignite Basics

Here, we provide a few of the fundamental basics of getting familiar with
PyTorch-Ignite. For a more thorough introduction, we defer to the
[official documentation](https://pytorch.org/ignite/).

First, PyTorch-Ignite is built around the concept of an `Engine`. An `Engine` is a
simple object that runs a function over a dataset. The function is called the
`update_fn` and is passed the `Engine` object and a batch of data.

Without loss of generality, there are **two** ways to instantiate an
`Engine` with an `update_fn`:

1. **Manually**: write it your own `update_fn` from scratch and pass it to the `Engine`
   initializer. _Note: this is required for unsupervised learning_.
2. **Automatically**: use the `create_supervised_trainer` function (provided by Ignite)
   to create an `Engine`. _Note: this only works for supervised learning_.

Below is a very simple (skeletal) program that shows the key components necessary
to train a model with Ignite.

```py title="Training with manual, user-defined 'update_fn'." linenums="1" hl_lines="5 6 7 15 16 17 18 19 20 21 25 26"
import typing as t
import torch
from ignite.engine import Engine

model: torch.nn.Module = ...
optimizer: torch.optim.Optimizer = ...
loss_fn: t.Callable | torch.nn.Module = ...


def train_step(engine: Engine, batch: t.Any):
    """
    Our 'update_fn' that trains a single batch of data with our
    model, optimizer, and loss function (defined above).
    """
    x, y = batch
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


train_dataloader = ...
trainer = Engine(train_step)
trainer.run(train_dataloader, max_epochs=10)  # train for 10 epochs
```

With this approach, you have full control over the training loop and can
customize it to your heart's content. However, it is more verbose and requires
more boilerplate code—though much less than with standard PyTorch.

```py title="Training with automatically-generated 'Engine' via 'create_supervised_trainer'." linenums="1" hl_lines="9"
from ignite.engine import create_supervised_trainer

model = ...
optimizer = ...
loss_fn = ...
train_dataloader = ...

trainer = create_supervised_trainer(model, optimizer, loss_fn)
trainer.run(train_dataloader, max_epochs=10)
```

The advantage of the second approach is that it is more concise and less error-prone.
However, it is limited to supervised learning tasks. Also, if you wish to do more highly
specific training procedures, then this may not create an `update_fn` that does what
you wish. But, for most general training purposes, it suffices.

In Flight, you can use _either_ approach.

***

## Using Ignite in Flight

