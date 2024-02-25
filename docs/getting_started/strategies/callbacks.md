# Strategy Callbacks

## How are Strategies defined in FLoX?
FLoX was designed to support modularity to enable creative and novel solutions for FL research. Therefore, in FLoX, we define a base ``Strategy`` class which serves as a class of callbacks. Classes that extend this base class (e.g., `FedAvg` extends `Strategy`) can implement their own unique logic which is seamlessly incorporated into the FL process.

```python
from flox.strategies import Strategy


class MyStrategy(Strategy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ...

    def wrk_before_train_step(self):
        pass

    def wrk_after_train_step(self, loss) -> 'loss':
        pass

    def agg_worker_selection(self):
        pass

    def agg_param_aggregation(self, state_dicts, **kwargs):
        pass

    def agg_before_share_params(self):
        pass

    def agg_after_collect_params(self) -> 'state_dict':
        pass

    def wrk_before_submit_params(self) -> 'state_dict':
        pass

    def wrk_on_recv_params(self):
        pass

```