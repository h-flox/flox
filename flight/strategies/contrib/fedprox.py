"""
## Code for FedProx (\
[Reference](https://github.com/ki-ljl/FedProx-PyTorch/blob/main/client.py))

```python
for epoch in tqdm(range(args.E)):
    train_loss = []
    for (seq, label) in Dtr:
        seq = seq.to(args.device)
        label = label.to(args.device)
        y_pred = model(seq)
        optimizer.zero_grad()
        # compute proximal_term
        proximal_term = 0.0
        for w, w_t in zip(model.parameters(), global_model.parameters()):
            proximal_term += (w - w_t).norm(2)

        loss = loss_function(y_pred, label) + (args.mu / 2) * proximal_term
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
```
"""
from __future__ import annotations

import typing as t

from ignite.engine import EventEnum

from flight.events import TrainProcessFnEvents, on
from flight.strategies.strategy import Strategy

if t.TYPE_CHECKING:
    from flight.events import Context


class FedProxEvents(EventEnum):
    BACKWARD_STARTED = "BACKWARD_STARTED"
    BACKWARD_COMPLETED = "BACKWARD_COMPLETED"
    OPTIM_STEP_COMPLETED = "OPTIM_STEP_COMPLETED"


class FedProx(Strategy):
    """
    Implementation of the *FedProx* algorithm [(ref)](
    https://proceedings.mlsys.org/paper_files/paper/
    2020/file/1f5fe83998a09396ebe6477d9475ba0c-Paper.pdf
    ).

    The optimization performed in this algorithm can be defined by:

    $$
    \\min_{w} h_{k}(w, w^{t}) = F_{k}(w) + \frac{\\mu}{2} \\|w - w^{t}\\|^{2}
    $$

    Specifically, FedProx relies on the addition of a proximal term to the loss
    objective before the optimization step occurs.
    """

    _requires_hooked_process_fn: bool = True

    def __init__(self, mu: float = 0.3):
        super().__init__()
        self.mu = mu

    @on(TrainProcessFnEvents.BACKWARD_STARTED)  # TODO: Fix `on` decorator typing.
    def add_proximal_term(self, context: Context):
        local_model = context["model"]
        global_model = context["global_model"]

        proximal_term = 0.0
        for (_, local_weights), (_, global_weights) in zip(
            local_model.get_params(), global_model.get_params()
        ):
            proximal_term += (local_weights - global_weights).norm(2)

        context["loss"] = context["loss"] + (self.mu / 2) * proximal_term
