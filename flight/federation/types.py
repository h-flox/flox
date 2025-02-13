from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

import torch
from proxystore.proxy import Proxy

from flight.federation.topologies.node import Node, NodeState, WorkerState
from flight.learning.base import AbstractDataModule, AbstractModule

if t.TYPE_CHECKING:
    from ignite.engine import Engine, Events
    from torch import nn

    from flight.engine.transporters import AbstractTransporter
    from flight.learning.params import Params
    from flight.strategies import AggrStrategy, TrainerStrategy, WorkerStrategy
    from flight.types import Record

    WorkerLocalState: t.TypeAlias = dict[str, t.Any]
    EventHandler: t.TypeAlias = t.Callable[[Engine, WorkerLocalState], None]


def _default_prepare_batch(
    batch: t.Sequence[torch.Tensor],
    device: t.Optional[str | torch.device] = None,
    non_blocking: bool = False,
) -> tuple[torch.Tensor | t.Sequence | t.Mapping | str | bytes, ...]:
    """Prepare batch for training or evaluation: pass to a device with options."""
    from ignite.utils import convert_tensor

    x, y = batch
    return (
        convert_tensor(x, device=device, non_blocking=non_blocking),
        convert_tensor(y, device=device, non_blocking=non_blocking),
    )


@dataclass
class Result:
    node: Node
    """
    The node that produced this result during a federation.
    """

    node_state: NodeState
    """
    The current state of the node that returned a given result during a federation.
    """

    params: Params
    """
    Parameters returned as part of a result from a single Node in a federation.
    """

    module: AbstractModule | None = field(default=None)
    """
    The model module that was returned as part of this result. For *Workers*, these
    are locally-trained model updates. For *Aggregators*, these are the aggregated
    modules.
    """

    records: list[Record] = field(default_factory=list)
    """
    List of records for model training/aggregation metrics.
    """

    extra: dict[str, t.Any] = field(default_factory=dict)
    """
    Extra data recorded by a node during the runtime of its job.
    """


@dataclass(slots=True, frozen=True)
class AggrJobArgs:
    # fut: Future
    round_num: int
    node: Node
    children: t.Collection[Node]
    child_results: t.Collection[Result]
    aggr_strategy: AggrStrategy
    transfer: AbstractTransporter


@dataclass
class IgniteConfig:
    # loss_fn: nn.Module
    # optimizer_cls: type[optim.Optimizer]
    #
    # optimizer_args: dict[str, t.Any] = field(default_factory=dict)
    # TODO: Is there a way make this dynamic? I.e., where strategies
    #       can create these per-endpoint?
    supervised: bool = True

    # supervised_training_step_args with defaults
    device: str | torch.DeviceObjType = "mps"  # TODO: Create `auto` default
    non_blocking: bool = True
    prepare_batch: t.Callable = _default_prepare_batch
    model_transform: t.Callable[[t.Any], t.Any] = lambda output: output
    output_transform: t.Callable[
        [t.Any, t.Any, t.Any, torch.Tensor], t.Any
    ] = lambda x, y, y_pred, loss: loss.item()
    gradient_accumulation_steps: int = 1
    model_fn: t.Callable[[nn.Module, t.Any], t.Any] = lambda model, x: model(x)


@dataclass(slots=True)
class TrainJobArgs:
    """
    Arguments for the local training job run on worker nodes in a federation.

    The default implementation for a local training job is given by
    [`default_training_job`][flight.federation.aggr.work.default_training_job].
    """

    node: Node
    parent: Node
    node_state: WorkerState
    model: AbstractModule | None
    data: AbstractDataModule
    worker_strategy: WorkerStrategy

    # New fields
    train_step: t.Callable | None = None
    valid_step: t.Callable | None = None
    test_step: t.Callable | None = None

    # ignite_config: IgniteConfig | None = None
    ignite_config: IgniteConfig = field(default_factory=lambda: IgniteConfig())
    supervised: bool = True  # Ignite-specific

    train_handlers: list[tuple[Events, EventHandler]] = field(default_factory=list)
    valid_handlers: list[tuple[Events, EventHandler]] = field(default_factory=list)
    test_handlers: list[tuple[Events, EventHandler]] = field(default_factory=list)

    # TODO: Remove entirely.
    trainer_strategy_depr: TrainerStrategy = None


AbstractResult: t.TypeAlias = Result | Proxy[Result]
"""
Helper type alias for a `Result` or a proxy to a `Result`.
"""

AggrJob: t.TypeAlias = t.Callable[[AggrJobArgs], Result]
"""
Function signature for aggregation aggr.
"""

TrainJob: t.TypeAlias = t.Callable[[TrainJobArgs], Result]
"""
Function signature for loca training aggr.
"""
