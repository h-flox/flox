"""
This module defines _Federated Learning **Processes**_. FL Processes come in two flavors:

- [Asynchronous][flox.federation.process_async.AsyncProcess]
- [Synchronous][flox.federation.process_sync.SyncProcess]

The former breed of FL Process (namely, synchronous) is the most widely-studied in the literature and is the only
one of the two that (currently) supports hierarchical execution.
"""

from flox.federation.fed import Federation
from flox.federation.fed_async import AsyncFederation
from flox.federation.fed_sync import SyncFederation

__all__ = [
    "Federation",
    "AsyncFederation",
    "SyncFederation",
]
