from __future__ import annotations

import enum
import typing as t
from dataclasses import dataclass, field
from uuid import UUID

import torch
from ignite.engine import Engine
from torch.optim import Optimizer

from flight.learning.module import TorchModule
from flight.learning.parameters import Params
from flight.state import AbstractNodeState
from flight.system.node import Node, NodeKind

if t.TYPE_CHECKING:
    pass


class JobStatus(str, enum.Enum):
    SUCCESS = "success"


@dataclass
class Result:
    """
    The result of a job.
    """

    node: Node
    module: TorchModule | None = field(repr=False)
    params: Params | None = field(default=None, repr=False)
    state: AbstractNodeState | None = field(default=None, repr=False)
    uuid: UUID | str | None = field(default=None, repr=True)
    status: JobStatus | None = field(default=None, repr=True)
    errors: list[BaseException] = field(default_factory=list)
    extra: dict[str, t.Any] | None = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self.params is None:
            self.params = self.module.get_params()

    def usable(self, to: NodeKind = NodeKind.AGGREGATOR) -> bool:
        """
        Check if the result is usable, i.e., if it has a valid state and module.

        Specifically, the `Result` is used as a standard format for responses from
        worker and aggregator jobs. For rapid prototyping, users might want to use
        the worker job to ensure their models are being trained as expected
        (i.e., outside of a federation's execution).

        Thus, we allow some attributes of this class to be `None` to allow for
        easier use for rapid prototyping. However, in actual runs of federations, we
        expect the `Result` to have a valid state and module.

        Args:
            to (NodeKind): The kind of node to check if the result is usable for.
                Defaults to `NodeKind.AGGREGATOR`.

        Returns:
            This function returns `True` if the `Result` has a valid state and
                module and can be used within a federation; otherwise this function
                returns `False`.

        Throws:
            - `ValueError`: If the `to` argument is not a valid `NodeKind` value.
        """
        if to == NodeKind.COORDINATOR or to == NodeKind.AGGREGATOR:
            return True  # TODO

        elif to == NodeKind.WORKER:
            return True  # TODO

        raise ValueError(
            f"Unsupported NodeKind value for argument {to=} for usability check."
        )


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
