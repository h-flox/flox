from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

if t.TYPE_CHECKING:
    import torch
    from torch.optim import Optimizer  # noqa
    from torch.utils.data import DataLoader, Dataset

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
class TrainingConfig:
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

    # TODO: Parametrize these.
    max_epochs = 3
    epoch_length = None


def worker_job(args: WorkerJobArgs):
    import functools

    from ignite.engine import Engine, create_supervised_trainer
    from torch.utils.data import DataLoader, Dataset

    from flight.events import IgniteEvents, WorkerEvents
    from flight.jobs.protocols import JobStatus, Result
    from flight.learning.module import TorchDataModule, TorchModule

    result = Result(
        node=args.node,
        # status=...,
        state=None,
        module=args.model,
        params=args.model.get_params(),
        # uuid=args.node.globus_compute_id,
    )
    extra: dict[str, t.Any] = {}
    args.strategy.fire_event_handler(WorkerEvents.STARTED, locals())

    ####################################################################################

    if not isinstance(args.model, TorchModule):
        raise TypeError("The `model` argument must be a subclass of `TorchModule`.")

    optimizer = args.model.configure_optimizers()
    criterion = args.model.configure_criterion()
    train_config = TrainingConfig()

    ####################################################################################
    # Setup the PyTorch-Ignite trainer `Engine`.
    ####################################################################################

    if args.train_step is not None:
        wrapped_train_step: ProcessFn = functools.partial(
            args.train_step,
            args.model,
            optimizer,
            criterion,
        )
        trainer = Engine(wrapped_train_step)

    elif args.supervised:
        trainer = create_supervised_trainer(  # TODO: Convert to our hooked function.
            args.model,
            optimizer,
            criterion,
            device=train_config.device,
            non_blocking=train_config.non_blocking,
            prepare_batch=train_config.prepare_batch,
            output_transform=train_config.output_transform,
            gradient_accumulation_steps=train_config.gradient_accumulation_steps,
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

    for event, handler in args.strategy.get_event_handlers_by_genre(
        IgniteEvents, when="train"
    ):
        print(f">>> Adding train_handler `{handler.__name__}` to `trainer` Engine.")
        handler_with_state = functools.partial(handler, trainer, locals())
        trainer.add_event_handler(event, handler_with_state)

    if validator:
        for event, handler in args.strategy.get_event_handlers_by_genre(
            IgniteEvents, when="validation"
        ):
            handler_with_state = functools.partial(handler, trainer, locals())
            validator.add_event_handler(event, handler_with_state)

    if tester:
        for event, handler in args.strategy.get_event_handlers_by_genre(
            IgniteEvents, when="test"
        ):
            handler_with_state = functools.partial(handler, locals())
            tester.add_event_handler(event, handler_with_state)

    ####################################################################################

    if isinstance(args.data, TorchDataModule):
        train_loader = args.data.train_data()
    elif isinstance(args.data, DataLoader):
        train_loader = args.data
    elif isinstance(args.data, Dataset):
        train_loader = DataLoader(args.data, **args.dataset_cfg)
    else:
        err = ValueError("`data` must be a TorchDataModule, DataLoader, or Dataset.")
        result.status = JobStatus.SUCCESS
        result.errors.append(err)
        return result

    args.strategy.fire_event_handler(WorkerEvents.BEFORE_TRAINING, locals())

    trainer_state = trainer.run(
        train_loader,
        max_epochs=train_config.max_epochs,
        epoch_length=train_config.epoch_length,
    )

    context = locals()
    args.strategy.fire_event_handler(WorkerEvents.AFTER_TRAINING, locals())

    ####################################################################################

    context = locals()
    args.strategy.fire_event_handler(WorkerEvents.COMPLETED, context)

    return Result(
        node=args.node,
        # status=...,
        state=None,
        params=args.model.get_params(),
        module=args.model,
        extra=extra,
    )
