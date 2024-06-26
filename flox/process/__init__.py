"""
This module defines _Federated Learning **Processes**_. FL Processes come in two flavors:

+ Synchronous
+ Asynchronous

The former breed of FL Process (namely, synchronous) is the most widely-studied in the literature and is the only
one of the two that (currently) supports hierarchical execution.
"""

from flox.process.process_async import AsyncProcess
from flox.process.process import Process
from flox.process.process_sync import SyncProcess

__all__ = ["AsyncProcess", "Process", "SyncProcess"]
