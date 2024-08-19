"""
```mermaid
flowchart LR

    subgraph federation_round
        direction LR
        federation_step-->traverse_step

        subgraph round
            direction BT
            traverse_step-.->coordinator_task
            traverse_step-.->aggr_task
            traverse_step-.->worker_task
        end
    end
```
"""

from .topologies.topo import Topology

__all__ = ["Topology"]
