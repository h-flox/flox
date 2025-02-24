<img src="graphics/flight-logo.png" style="max-width: 400px;">

# Welcome to Flight

### Getting Started

**Flight** (**F**ederated **L**earning **I**n **G**eneralized **H**ierarchical **T**
opologies)
is a simple, highly customizable, and easy-to-deploy framework for launching Federated
Learning processes across
a decentralized network. It is designed to simulate FL workflows while also making it
trivially easy to deploy them on
real-world devices (e.g., Internet-of-Things and edge devices). Built on top of _Globus
Compute_ (formerly known as
_funcX_), Flight is designed to run on anything that can be started as a Globus Compute
Endpoint.

### What can Flight do?

Flight is supports several state-of-the-art approaches for FL processes, including
hierarchical and asynchronous FL.

|                  |      2-tier      |  Hierarhchical   |
|-----------------:|:----------------:|:----------------:|
|  **Synchronous** | :material-check: | :material-check: |
| **Asynchronous** | :material-check: | :material-close: |

#### Installation

The package can be found on pypi:

```bash
pip install py-flight
```

#### Usage

Flight is a simple, highly-customizable, and easy-to-deploy framework for hierarchical,
multi-tier federated learning
systems built on top of the Globus Compute platform.

```python title="Basic Flight Example" linenums="1"
from v1.flight import Topology, federated_fit
from torch import nn


class MyModule(nn.Module):
    ...


topo = Topology.from_yaml("my-topo.yaml")
federated_fit(
    module=MyModule(),
    topo=topo,
    strategy="fedavg",
    strategy_params={"participation_frac": 0.5},
    where="local",
    logger="csv",
    log_out="my_results.csv"
)
```

```mermaid
flowchart TD
%% Nodes
    A("fab:fa-youtube Starter Guide")
    B("fab:fa-youtube Make Flowchart")
    n1@{ icon: "fa:gem", pos: "b", h: 24}
    C("fa:fa-book-open Learn More")
    D{"Use the editor"}
    n2(Many shapes)@{ shape: delay}
    E(fa:fa-shapes Visual Editor)
    F("fa:fa-chevron-up Add node in toolbar")
    G("fa:fa-comment-dots AI chat")
    H("fa:fa-arrow-left Open AI in side menu")
    I("fa:fa-code Text")
    J(fa:fa-arrow-left Type Mermaid syntax)

%% Edge connections between nodes
    A --> B --> C --> n1 & D & n2
    D -- Build and Design --> E --> F
    D -- Use AI --> G --> H
    D -- Mermaid js --> I --> J

%% Individual node styling. Try the visual editor toolbar for easier styling!
    style E color:#FFFFFF, fill:#AA00FF, stroke:#AA00FF
    style G color:#FFFFFF, stroke:#00C853, fill:#00C853
    style I color:#FFFFFF, stroke:#2962FF, fill:#2962FF

%% You can add notes with two "%" signs in a row!
```
