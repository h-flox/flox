## What are Strategies?
There are many different ways to perform Federated Learning. More specifically, there are many different ways to optimize the _global model_ being trained in an FL process with respect to many intricate challenges (e.g., system heterogeneity or statistical heterogeneity). In FLoX, a **Strategy** is an abstraction that implements one of these ways of optimizing the global model.

Some prominent examples from the literature of what we consider "Strategies" include (but are, of course, not limited to):

* Federated Stochastic Gradient Descent (`FedSGD`)[^fedavg]
* Federated Averaging (`FedAvg`)[^fedavg]
* `FedProx`[^fedprox]

## What _exactly_ do Strategies do?
In a nutshell, a lot. Federated Learning is a complex process with tasks being done on the worker nodes and the aggregator node(s). Thus, Strategies can touch a lot of different parts of the entire logic of an FL process. 

### Model Parameter Aggregation
...

### Model Parameter Communication
...

#### Parameter Quantization/Compression
...

#### Parameter Encryption/Decryption
...

### Client Selection
...

### Local Training
...


## How are Strategies defined in FLoX?
In FLoX, we define...

```python
from flox.strategies import Strategy


class MyStrategy(Strategy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ...

    def on_before_train_step(self):
        pass

    def on_after_train_step(self, loss) -> 'loss':
        pass

    def on_worker_selection(self):
        pass

    def on_param_aggregation(self, state_dicts, **kwargs):
        pass

    def on_before_aggr_send_params(self):
        pass

    def on_after_aggr_recv_params(self) -> 'state_dict':
        pass

    def on_before_worker_send_params(self) -> 'state_dict':
        pass

    def on_worker_recv_params(self):
        pass

```

[^fedavg]: McMahan, Brendan, et al. "Communication-efficient learning of deep networks from decentralized data." Artificial intelligence and statistics. PMLR, 2017. [(Link)](http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf)
[^fedprox]: Li, Tian, et al. "Federated optimization in heterogeneous networks." Proceedings of Machine learning and systems 2 (2020): 429-450. [(Link)](https://proceedings.mlsys.org/paper_files/paper/2020/file/1f5fe83998a09396ebe6477d9475ba0c-Paper.pdf)