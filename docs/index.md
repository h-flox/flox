# Welcome to FLoX

![FLoX Graphic](graphics/flox_fig.png)

### Getting Started
FLoX (**F**ederated **L**earning **o**n func**X**) is a simple, highly customizable, and easy-to-deploy framework for launching Federated Learning processes across a decentralized network. Built on top of _Globus Compute_ (formerly known as _funcX_), FLoX is designed to run on anything that can be started as a Globus Compute Endpoint.  

#### Installation

The package can be found on pypi:

```bash
pip install flox
```

#### Usage

FLoX is a simple, highly-customizable, and easy-to-deploy framework for hierarchical, multi-tier federated learning
systems built on top of the Globus Compute platform.

```python title="Basic FLoX Example" linenums="1"
from flox import Flock, federated_fit
from torch import nn


class MyModule(nn.Module):
    ...


flock = Flock.from_yaml("my_flock.yaml")
federated_fit(
    module=MyModule(),
    flock=flock,
    strategy="fedavg",
    strategy_params={"participation_frac": 0.5},
    where="local",
    logger="csv",
    log_out="my_results.csv"
)
```
