from __future__ import annotations

import functools
import typing as t

import torch
from ignite.utils import convert_tensor

from flight.events import TrainProcessFnEvents
from flight.strategies import Strategy

if t.TYPE_CHECKING:
    from ignite.engine import Engine

    Loss: t.TypeAlias = torch.Tensor
    ModelOutput: t.TypeAlias = torch.Tensor


class ProcessFn(t.Protocol):
    def __call__(self) -> Loss:
        pass


class HookedProcessFn(t.Protocol):
    def __call__(self) -> ModelOutput:
        pass


def hooked_process_fn(model, batch, criterion, strategy, optimizer) -> Loss:
    x, y_true = batch
    y_pred = model(x)
    loss = criterion(y_pred, y_true)

    strategy.fire_event("BACKWARD_STARTED")
    loss.backward()
    strategy.fire_event("BACKWARD_COMPLETED")

    optimizer.step()
    return loss.item()


def _prepare_batch(
    batch: t.Sequence[torch.Tensor],
    device: t.Optional[str | torch.device] = None,
    non_blocking: bool = False,
) -> tuple[t.Union[torch.Tensor, t.Sequence, t.Mapping, str, bytes], ...]:
    """Prepare batch for training or evaluation: pass to a device with options."""
    x, y = batch
    return (
        convert_tensor(x, device=device, non_blocking=non_blocking),
        convert_tensor(y, device=device, non_blocking=non_blocking),
    )


def fire_event_handler_if_strategy_exists(
    strategy: Strategy | None,
    event: TrainProcessFnEvents,
) -> None:
    """Fire an event handler if a strategy is provided.

    Args:
        strategy (Strategy): The strategy to check.
        event (TrainProcessFnEvents): The event to fire.
    """
    if strategy:
        strategy.fire_event_handler(event, locals())


########################################################################################


def hooked_supervised_training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: t.Callable[[t.Any, t.Any], torch.Tensor] | torch.nn.Module,
    device: str | torch.device | None = None,
    non_blocking: bool = False,
    prepare_batch: t.Callable = _prepare_batch,
    model_transform: t.Callable[[t.Any], t.Any] = lambda output: output,
    output_transform: t.Callable[
        [t.Any, t.Any, t.Any, torch.Tensor], t.Any
    ] = lambda x, y, y_pred, loss: loss.item(),
    gradient_accumulation_steps: int = 1,
    model_fn: t.Callable[[torch.nn.Module, t.Any], t.Any] = lambda model, x: model(x),
    strategy: Strategy | None = None,
) -> t.Callable:
    """
    This factory function for supervised training.

    This function mimics the implementation of the
    [`supervised_training_step`](
    https://docs.pytorch.org/
    ignite/_modules/ignite/engine.html#supervised_training_step
    )
    function provided by Ignite, but with the addition of Flight's own custom
    event handling system via [`ProcessFnEvents`][flight.events.ProcessFnEvents].

    These added events allow for more granular control over the training process,
    using Flight's event system. An example [`Strategy`][flight.strategies.Strategy]
    that uses this function is Flight's implementation of the
    [`FedProx`][flight.strategies.contrib.fedprox.FedProx] algorithm.

    ```mermaid
    %%{init: { 'noteFontFamily': 'Courier New, monospace' } }%%
    sequenceDiagram
        participant I as Ignite
        participant S as Strategy

        par
            I->>S: `ProcessFnEvents.BATCH_PREPARE_STARTED`
            Note over I: Get and prepare the current batch `(x, y)`
            I->>S: `ProcessFnEvents.BATCH_PREPARE_COMPLETED`
        end

        Note over I: Forward pass and calculate loss

        par
            I->>S: `ProcessFnEvents.BACKWARD_STARTED`
            Note over I: Backpropagate loss (via `loss.backward()`)
            I->>S: `ProcessFnEvents.BACKWARD_COMPLETED`
        end

        par
            I->>S: `ProcessFnEvents.OPTIM_STEP_STARTED`
            Note over I: Update model parameters (via `optimizer.step()`)
            I->>S: `ProcessFnEvents.OPTIM_STEP_COMPLETED`
        end
    ```

    Args:
        model (torch.nn.Module):
            The model to train.
        optimizer (torch.optim.Optimizer):
            The optimizer to use.
        loss_fn (t.Callable[[t.Any, t.Any], torch.Tensor] | torch.nn.Module):
            The loss function that receives `y_pred` and `y`, and returns the
            loss as a tensor.
        device (str | torch.device | None):
            Device type specification (default: None). Applies to batches after
            starting the engine. Model *will not* be moved. Device can be CPU, GPU.
        non_blocking (bool):
            If `True` and this copy is between CPU and GPU, the copy may
            occur asynchronously with respect to the host. For other cases, this
            argument has no effect. Defaults to `False`.
        prepare_batch (t.Callable):
            Function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        model_transform (t.Callable[[t.Any], t.Any]):
            Function that receives the output from the model and convert it into the
            form as required by the loss function
        output_transform (t.Callable[[t.Any, t.Any, t.Any, torch.Tensor], t.Any]):
            Function that receives 'x', 'y', 'y_pred', 'loss' and returns value
            to be assigned to engine's state.output after each iteration.
            Default is returning `loss.item()`.
        gradient_accumulation_steps (int):
            Number of steps the gradients should be accumulated across.
            (default: 1 (means no gradient accumulation))
        model_fn (t.Callable[[torch.nn.Module, t.Any], t.Any]):
            The model function that receives `model` and `x`, and returns `y_pred`.
        strategy (Strategy | None):
            The strategy that will handle the events and process the training step.
            Defaults to `None`. If `None`, no events will be fired. Note: in
            federations, this will never be `None`.

    Returns:
        A function that can be used as a process function for an Ignite `Engine`.
            It processes a batch of data, computes the loss, performs backpropagation,
            and updates the model parameters.
    """
    if gradient_accumulation_steps <= 0:
        raise ValueError(
            "Gradient_accumulation_steps must be strictly positive. "
            "No gradient accumulation if the value set to one (default)."
        )

    _fire_event_handler_if_strategy_exists = functools.partial(
        fire_event_handler_if_strategy_exists,
        strategy=strategy,
    )

    def update(
        engine: Engine, batch: t.Sequence[torch.Tensor]
    ) -> t.Union[t.Any, tuple[torch.Tensor]]:
        if (engine.state.iteration - 1) % gradient_accumulation_steps == 0:
            optimizer.zero_grad()
        model.train()

        _fire_event_handler_if_strategy_exists(
            event=TrainProcessFnEvents.BATCH_PREPARE_STARTED
        )
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        _fire_event_handler_if_strategy_exists(
            event=TrainProcessFnEvents.BATCH_PREPARE_COMPLETED
        )

        output = model_fn(model, x)
        y_pred = model_transform(output)
        loss = loss_fn(y_pred, y)

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        _fire_event_handler_if_strategy_exists(
            event=TrainProcessFnEvents.BACKWARD_STARTED
        )
        loss.backward()
        _fire_event_handler_if_strategy_exists(
            event=TrainProcessFnEvents.BACKWARD_COMPLETED
        )

        if engine.state.iteration % gradient_accumulation_steps == 0:
            _fire_event_handler_if_strategy_exists(
                event=TrainProcessFnEvents.OPTIM_STEP_COMPLETED
            )
            optimizer.step()
            _fire_event_handler_if_strategy_exists(
                event=TrainProcessFnEvents.OPTIM_STEP_COMPLETED
            )

        return output_transform(
            x,
            y,
            y_pred,
            loss * gradient_accumulation_steps,
        )

    return update


########################################################################################


def hooked_supervised_evaluation_step(
    model: torch.nn.Module,
    device: str | torch.device | None = None,
    non_blocking: bool = False,
    prepare_batch: t.Callable = _prepare_batch,
    model_transform: t.Callable[[t.Any], t.Any] = lambda output: output,
    output_transform: t.Callable[[t.Any, t.Any, t.Any], t.Any] = lambda x, y, y_pred: (
        y_pred,
        y,
    ),
    model_fn: t.Callable[[torch.nn.Module, t.Any], t.Any] = lambda model, x: model(x),
    strategy: Strategy | None = None,
) -> t.Callable:
    """
    Args:
        model (torch.nn.Module):
            The model to evaluate.
        device (str | torch.device | None):
            Device type specification (default: None).
            Applies to batches after starting the engine.
            Model *will not* be moved.
        non_blocking (bool):
            If True and this copy is between CPU and GPU, the copy may occur
            asynchronously with respect to the host. For other cases, this argument
            has no effect.
        prepare_batch (t.Callable):
            Function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        model_transform (t.Callable[[t.Any], t.Any]):
            Function that receives the output from the model and convert it into
            the predictions: ``y_pred = model_transform(model(x))``.
        output_transform (t.Callable[[t.Any, t.Any, t.Any], t.Any]):
            Function that receives 'x', 'y', 'y_pred' and returns value
            to be assigned to engine's state.output after each iteration.
            Default is returning `(y_pred, y,)` which fits output expected by metrics.
            If you change it you should use `output_transform` in metrics.
        model_fn (t.Callable[[torch.nn.Module, t.Any], t.Any]):
            The model function that receives `model` and `x`, and returns `y_pred`.
        strategy (Strategy | None):
            The strategy that will handle the events and process the evaluation step.
            Defaults to `None`. If `None`, no events will be fired.
            Note: in federations, this will never be `None`.

    Returns:
        Inference function to be used for evaluation.
    """

    # _fire_event_handler_if_strategy_exists = functools.partial(
    #     fire_event_handler_if_strategy_exists,
    #     strategy=strategy,
    # )

    def evaluate_step(
        engine: Engine, batch: t.Sequence[torch.Tensor]
    ) -> t.Any | tuple[torch.Tensor]:
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            output = model_fn(model, x)
            y_pred = model_transform(output)
            # TODO: Add new event handlers (with new events) here!
            return output_transform(x, y, y_pred)

    return evaluate_step


"""
if __name__ == "__main__":
    global_model = ...
    model = ...
    criterion = ...
    batch = ...
    loss = hooked_process_fn(model, batch, criterion)

    fire_event("BACKWARD_STARTED")
    proximal_term = 0.0
    for w, w_t in zip(model.get_params(), global_model.get_params()):
        proximal_term += (w - w_t).norm(2)

    loss = criterion(y_pred, label) + (args.mu / 2) * proximal_term

    loss.backward()
"""
