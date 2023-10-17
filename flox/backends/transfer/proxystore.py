import proxystore

from proxystore.connectors.endpoint import EndpointConnector
from proxystore.store import Store

from flox.flock import Flock


class ProxyStoreWrapper:
    def __init__(self, flock: Flock, store: str = "endpoint", name: str = "default"):
        if not flock.proxystore_ready:
            raise ValueError(
                "Flock is not ready to use ProxyStore (i.e., `flock.proxystore_ready` "
                "returns `False`). You need each node should have a valid ProxyStore "
                "Endpoint UUID."
            )

        self.connector = EndpointConnector(
            endpoints=[node.proxystore_endpoint for node in flock.nodes()]
        )
        self.store = Store(name=name, connector=self.connector)
