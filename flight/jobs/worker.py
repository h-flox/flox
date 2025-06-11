from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

if t.TYPE_CHECKING:
    import torch
    from torch.optim import Optimizer  # noqa
    from torch.utils.data import DataLoader, Dataset

    from flight.events import EventHandler
    from flight.learning.module import TorchDataModule, TorchModule
    from flight.strategies.strategy import Strategy
    from flight.system.node import Node

    from ..learning.parameters import Params
    from .protocols import ProcessFn, Result


def _default_dataset_cfg() -> dict[str, t.Any]:
    """
    Default dataset configuration for the DataLoader.

    Returns:
        Default configuration for the DataLoader.
    """
    return {
        "batch_size": 64,
    }


class WorkerJobProto(t.Protocol):
    @staticmethod
    def __call__(args: WorkerJobArgs) -> Result:
        """
        This method is called when the WORKER job is launched.
        """


@dataclass
class WorkerJobArgs:
    strategy: Strategy

    model: TorchModule
    data: TorchDataModule | DataLoader | Dataset
    params: Params

    node: Node | None = None

    train_step: t.Any | None = None
    supervised: bool = field(default=True)  # TODO: Move to `TorchModule`?

    dataset_cfg: dict[str, t.Any] = field(default_factory=_default_dataset_cfg)


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
    device: str | torch.device | None = "cpu"  # TODO: Create `auto` default
    non_blocking: bool = True
    prepare_batch: t.Callable = _default_prepare_batch
    model_transform: t.Callable[[t.Any], t.Any] = lambda output: output
    output_transform: t.Callable[
        [t.Any, t.Any, t.Any, torch.Tensor], t.Any
    ] = lambda x, y_true, y_pred, loss: (
        # x,
        y_true,
        y_pred,
        # loss.item(),
    )
    gradient_accumulation_steps: int = 1
    model_fn: t.Callable[[torch.nn.Module, t.Any], t.Any] = lambda model, x: model(x)


def worker_job(args: WorkerJobArgs):
    import functools

    from ignite.engine import Engine, create_supervised_trainer
    from torch.utils.data import DataLoader, Dataset

    from flight.events import IgniteEvents, WorkerEvents
    from flight.jobs.protocols import Result
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
            args.model,
            optimizer,
            loss_fn,
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

    validator: Engine | None = None  # TODO: Implement this.
    tester: Engine | None = None  # TODO: Implement this.

    ####################################################################################

    train_handlers = args.strategy.get_event_handlers_by_genre(IgniteEvents)
    for event, handler in train_handlers:
        print(f">>> Adding train_handler `{handler.__name__}` to `trainer` Engine.")
        handler_with_state = functools.partial(handler, trainer, context)
        trainer.add_event_handler(event, handler_with_state)

    if validator:
        # NOTE: We need to figure out how we will discern between the handlers
        #       meant for the trainer versus the validator and the test.
        valid_handlers: list[tuple[str, EventHandler]] = []  # TODO
        for event, handler in valid_handlers:
            handler_with_state = functools.partial(handler, trainer, context)
            validator.add_event_handler(event, handler_with_state)

    if tester:
        test_handlers: list[tuple[str, EventHandler]] = []  # TODO
        for event, handler in test_handlers:
            handler_with_state = functools.partial(handler, context)
            tester.add_event_handler(event, handler_with_state)

    ####################################################################################

    if isinstance(args.data, TorchDataModule):
        train_loader = args.data.train_data()
    elif isinstance(args.data, DataLoader):
        train_loader = args.data
    elif isinstance(args.data, Dataset):
        train_loader = DataLoader(args.data, **args.dataset_cfg)
    else:
        raise ValueError("`data` must be a TorchDataModule, DataLoader, or Dataset.")

    context = locals()
    args.strategy.fire_event_handler(WorkerEvents.BEFORE_TRAINING, context)

    max_epochs = 3  # TODO: Parameterize
    epoch_length = None  # TODO: Parameterize
    trainer_state = trainer.run(
        train_loader,
        max_epochs=max_epochs,
        epoch_length=epoch_length,
    )

    context = locals()
    args.strategy.fire_event_handler(WorkerEvents.AFTER_TRAINING, context)

    ####################################################################################

    context = locals()
    args.strategy.fire_event_handler(WorkerEvents.COMPLETED, context)

    return Result(
        node=args.node,
        state=None,
        params=args.model.get_params(),
        module=args.model,
        extra={},
    )
