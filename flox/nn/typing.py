from typing import TypeAlias, Literal

import torch

Kind: TypeAlias = Literal["async", "sync"]
Where: TypeAlias = Literal["local", "globus_compute"]
StateDict = dict[str, torch.Tensor]
"""The state dict of PyTorch ``torch.nn.Module`` (see ``torch.nn.Module.state_dict()``)."""
