# Defining Your Own Custom Strategies

FLoX was designed with customizability in mind. FL is a new research area that invites countless questions about how to
best perform FL. Additionally, the best FL approach will vary depending on the data, network connectivity, other
requirements, etc. As such, we aimed to make defining original Strategies to be as pain-free as possible.

Implementing a custom ``Strategy`` simply requires defining a new class that extends/subclasses the ``Strategy`` protocol
(as seen above). The ``Strategy`` protocol provides a handful of callbacks for you to inject custom logic to adjust how the
FL process runs.

As an example, let's use our source code for the implementation of ``FedProx`` as an example.

```python
class FedProx(FedAvg):
    """..."""

    def __init__(
            self,
            mu: float = 0.3,
            participation: float = 1.0,
            probabilistic: bool = False,
            always_include_child_aggregators: bool = True,
            seed: int = None,
    ):
        """..."""
        super().__init__(
            participation,
            probabilistic,
            always_include_child_aggregators,
            seed,
        )
        self.mu = mu

    def wrk_after_train_step(
            self,
            state: FloxWorkerState,
            loss: torch.Tensor,
            **kwargs,
    ) -> torch.Tensor:
        """..."""
        global_model = state.global_model
        local_model = state.local_model

        params = list(local_model.params().values())
        params0 = list(global_model.params().values())

        norm = torch.sum(
            torch.Tensor(
                [torch.sum((params[i] - params0[i]) ** 2) for i in range(len(params))]
            )
        )

        proximal_term = (self.mu / 2) * norm
        loss += proximal_term
        return loss
```