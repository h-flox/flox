## What is a Flock?
Federated Learning is done over a scattered collection of devices that collectively train a machine learning model. In FLoX, we refer to the topology that defines these devices and their connectivity as a ***Flock***, coined after a flock of birds.

## Creating Flock Networks
The ``flox.flock`` module contains the code needed to define your own ``Flock`` networks. They are built on top of the ``NetworkX``  library. Generally speaking, to create Flock instances in FLoX, we provide two interfaces:
  1. interactive mode
  2. file mode

Interactive mode involves creating a ``NetworkX.DiGraph()`` object directly and then passing that into the ``Flock`` constructor. This is **not** recommended.

The recommended approach is ***file mode***. In this mode, you define the Flock network using a supported file type (e.g., `*.yaml`) and simply use it to create the Flock instance.

```python
from flox.flock import Flock

f = "my_flock.yaml"
flock = Flock.from_yaml(f)
```

***

# Endpoint YAML Configuration

```yaml
rpi-0:
  globus-compute-endpoint: ... # required
  proxystore-endpoint: ...     # required
  children: [rpi-1]            # required
  resources:
    num_cpus: 2
    num_gpus: 0.5

rpi-1:
  ...

...
```