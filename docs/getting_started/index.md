Here, we provide a brief high-level overview of how to get started with FLoX.

```python
from flox.run import federated_fit
from flox.federation.topologies import Topology

flock = Topology.from_yaml("sample-topologies.yml")
results = federated_fit(
    flock, module_cls, datasets, num_global_rounds=10, strategy="fedavg"
)
```