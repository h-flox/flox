from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

if t.TYPE_CHECKING:
    import torch

    from flight.events import EventHandler
    from flight.learning.module import TorchDataModule, TorchModule, Params
    from flight.strategy import Strategy

    from ignite.engine import Engine

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
            tuple[torch.nn.Module, torch.optim.Optimizer, torch.nn.Module],
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


@dataclass
class WorkerJobArgs:
    strategy: Strategy

    model: TorchModule
    data: TorchDataModule
    params: Params

    train_step: t.Any
    supervised: bool = field(default=True)  # TODO: Move to `TorchModule`?

    dataset_cfg: t.Mapping[str, t.Any] = field(
        default_factory=lambda: {"batch_size": 64}
    )


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
    model_fn: t.Callable[[torch.nn.Module, t.Any], t.Any] = lambda model, x: model(x)


def worker_job(args: WorkerJobArgs):
    import functools

    from ignite.engine import Engine, create_supervised_trainer
    from torch.utils.data import DataLoader, Dataset

    from flight.events import WorkerEvents
    from flight.learning.module import TorchDataModule, TorchModule

    context: dict[str, t.Any] = {}

    ####################################################################################

    args.strategy.fire_event_handler(WorkerEvents.STARTED, context)

    assert isinstance(args.model, TorchModule)

    optimizer = args.model.configure_optimizers()
    loss_fn = args.model.configure_criterion()
    ignite_cfg = IgniteConfig()

    ####################################################################################
    # Setup the PyTorch-Ignite trainer `Engine`.
    ####################################################################################

    if args.train_step is not None:
        wrapped_train_step: ProcessFn = functools.partial(
            args.train_step,
            (args.model, optimizer, loss_fn),
        )
        trainer = Engine(wrapped_train_step)

    elif args.supervised:
        trainer = create_supervised_trainer(
            args.model,
            optimizer,
            loss_fn,
            device=ignite_cfg.device,
            non_blocking=ignite_cfg.non_blocking,
            prepare_batch=ignite_cfg.prepare_batch,
            output_transform=ignite_cfg.output_transform,
            gradient_accumulation_steps=ignite_cfg.gradient_accumulation_steps,
        )

    else:
        raise ValueError(
            "Unsupervised training can only be done by providing a `train_step` "
            "function that implements the `UnwrappedProcessFn` interface. "
            "If the user does not provide one, then the `create_supervised_trainer` "
            "function from PyTorch-Ignite is used. This function does not support "
            "unsupervised training."
        )

    ####################################################################################

    local_state = context

    train_handlers: list[tuple[str, EventHandler]] = []  # TODO
    valid_handlers: list[tuple[str, EventHandler]] = []  # TODO
    test_handlers: list[tuple[str, EventHandler]] = []  # TODO

    validator: Engine | None = None  # TODO: Implement this.
    tester: Engine | None = None  # TODO: Implement this.

    for event, handler in train_handlers:
        # handler_with_state = functools.partial(handler, local_state)
        # trainer.add_event_handler(event, handler_with_state)
        trainer.add_event_handler(event, handler, local_state)

    if validator:
        for event, handler in valid_handlers:
            handler_with_state = functools.partial(handler, local_state)
            validator.add_event_handler(event, handler_with_state)

    if tester:
        for event, handler in test_handlers:
            handler_with_state = functools.partial(handler, local_state)
            tester.add_event_handler(event, handler_with_state)

    ####################################################################################

    if isinstance(args.data, TorchDataModule):
        train_loader = args.data.train_data()
    elif isinstance(args.data, DataLoader):
        train_loader = args.data
    elif isinstance(args.data, Dataset):
        train_loader = DataLoader(args.data, **args.dataset_cfg)
    else:
        raise ValueError

    ####################################################################################

    args.strategy.fire_event_handler(WorkerEvents.COMPLETED, context)

    return None
