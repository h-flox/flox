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

from flight.strategy import Strategy

if t.TYPE_CHECKING:
    from ignite.engine import Engine


class FedProx(Strategy):
    def __init__(self, mu: float = 0.3):
        super().__init__()
        self.mu = mu

    # TODO: This has to be a custom event, see:
    # https://pytorch-ignite.ai/how-to-guides/08-custom-events/
    # @on(IgniteEvents.BACKWARD_STARTED)
    def compute_proximal_term(self, engine: Engine, context):
        proximal_term = 0.0
        local_model = context["model"]
        global_model = context["global_model"]
        for (_, w), (_, w_t) in zip(
            local_model.get_params(),
            global_model.get_params(),
        ):
            proximal_term += (w - w_t).norm(2)

        criterion = context["criterion"]
        x, y_true = engine.state.batch
        y_pred = engine.state.output
        loss = criterion(y_pred, y_true) + (self.mu / 2) * proximal_term
        return loss
