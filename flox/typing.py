import torch

from typing import NewType

StateDict = dict[str, torch.Tensor]
"""The state dict of PyTorch ``torch.nn.Module`` (see ``torch.nn.Module.state_dict()``)."""
