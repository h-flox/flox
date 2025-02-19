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

from v1.flight.topologies import Topology

from .fed_async import AsyncFederation
from .fed_sync import SyncFederation

__all__ = ["AsyncFederation", "SyncFederation", "Topology"]
