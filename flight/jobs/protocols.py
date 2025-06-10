from __future__ import annotations

import typing as t
from dataclasses import dataclass, field
from uuid import UUID

import torch
from ignite.engine import Engine
from torch.optim import Optimizer

from flight.learning.module import TorchModule
from flight.learning.parameters import Params
from flight.state import AbstractNodeState
from flight.system.node import Node

if t.TYPE_CHECKING:
    pass


@dataclass
class Result:
    """
    The result of a job.
    """

    node: Node
    state: AbstractNodeState = field(repr=False)
    module: TorchModule = field(repr=False)
    params: Params = field(repr=False)
    uuid: UUID | str | None = field(default=None, repr=True)
    extra: dict[str, t.Any] | None = field(default_factory=dict, repr=False)


ProcessFn: t.TypeAlias = t.Callable[[Engine, t.Any], t.Any]
"""
As defined by Ignite (see [Engine][ignite.engine.Engine]), this is a function
that receives a handle to the engine and the current batch in each iteration,
and returns data to be stored in the engine's state.

Simply put, this function defines how an `Engine` processes a batch of data during
training, testing, and evaluation.
"""

UnwrappedProcessFn: t.TypeAlias = t.Callable[
    [
        tuple[torch.nn.Module, Optimizer, torch.nn.Module],
        Engine,
        t.Any,
    ],
    t.Any,
]
"""
A function that has access to the model, optimizer, and loss function, and processes
in a broader scope than a `ProcessFn`. This function type is used to wrap a
`ProcessFn` with these dependencies in context.
"""
