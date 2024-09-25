"""
This module provides the standard interfaces, protocols, and classes for defining and
training neural networks in Flight. This module also provides classes and utility
functions for loading (and setting up) datasets for Flight federations.

Flight currently provides support for the following deep learning frameworks:

- [PyTorch](https://pytorch.org)
- [PyTorch Lightning](https://lightning.ai/pytorch-lightning)
- [Scikit-Learn](https://scikit-learn.org/stable/) (specifically, its `MLPRegressor`
  and `MLPClassifier` models).

## Trainable Models
...

## Trainer
...

| Trainer | Module |
| :------ | :----- |
| TorchTrainer | ... |
| ScitkitTrainer | ... |
| LightningTrainer | ... |
| CustomTrainer | ... |



```mermaid
flowchart TB

AbstractModule-->LightningModule
AbstractModule-->ScikitModule
AbstractModule-->TorchModule

AbstractDataModule-->LightningDataModule
AbstractDataModule-->ScikitDataModule
AbstractDataModule-->TorchDataModule

```
"""

from .base import AbstractDataModule, AbstractModule, AbstractTrainer

__all__ = ["AbstractModule", "AbstractDataModule", "AbstractTrainer"]
