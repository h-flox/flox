<img src="docs/graphics/flight-logo.png" style="max-width: 400px;">

# Flight
[![CI/CD](https://github.com/h-flox/flox/actions/workflows/ci.yaml/badge.svg?branch=numpy-params)](https://github.com/h-flox/flox/actions/workflows/ci.yaml)
[![tests](https://github.com/h-flox/flox/actions/workflows/tests.yaml/badge.svg?branch=numpy-params)](https://github.com/h-flox/flox/actions/workflows/tests.yaml)

**Flight** (**F**ederated **L**earning **I**n **G**eneralized **H**ierarchical **T**opologies)
is a modular, easy-to-use federated learning framework built on top of Globus Compute,
a federated Function-as-a-Service platform.

## Installation

At this time, FLoX is not available on `pypi`, but can be installed with [`pip`](https://pip.pypa.io/en/stable)
using the following command:

```bash
pip install git+https://www.github.com/h-flox/flox/
```

## Documentation

Documentation is currently being written and we hope to have them available online here soon.

## Citing FLoX

If you use FLoX or any of this code in your work, please cite the following paper.
> Nikita Kotsehub, Matt Baughman, Ryan Chard, Nathaniel Hudson, Panos Patros, Omer Rana,
> Ian Foster, and Kyle Chard.
> ["FLoX: Federated learning with FaaS at the edge."](https://ieeexplore.ieee.org/document/9973578)
> In 2022 IEEE 18th International Conference on e-Science (e-Science), pp. 11-20. IEEE, 2022.

```bibtex
@inproceedings{kotsehub2022flox,
  title={{FLoX}: Federated learning with {FaaS} at the edge},
  author={Kotsehub, Nikita and Baughman, Matt and Chard, Ryan and Hudson, Nathaniel and Patros, Panos and Rana, Omer and Foster, Ian and Chard, Kyle},
  booktitle={2022 IEEE 18th International Conference on e-Science (e-Science)},
  pages={11--20},
  year={2022},
  organization={IEEE}
}
```

```
flox/
├── federation/
│   ├── jobs/
│   ├── topologies/
│   ├── config.py
│   └── app.py
├── planes/
│   ├── control/
│   │   ├── globus.py
│   │   ├── parsl.py
│   │   ├── local.py
│   │   └── ...
│   ├── data/
│   │   ├── redis.py
│   │   └── proxystore.py
│   └── runtime.py
└── strategies/
    ├── commons/
    ├── impl/
    └── protocols.py
```
