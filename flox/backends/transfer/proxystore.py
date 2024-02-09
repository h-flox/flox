from proxystore.connectors.endpoint import EndpointConnector
from proxystore.proxy import Proxy
from proxystore.store import Store

from flox.backends.transfer.base import BaseTransfer
from flox.flock import Flock
from flox.reporting.job import JobResult


class ProxyStoreTransfer(BaseTransfer):
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
        store = Store(name=name, connector=self.connector)
        self.config = store.config()

    def report(
        self, node_state, node_idx, node_kind, state_dict, history
    ) -> Proxy[JobResult]:
        jr = JobResult(
            node_state=node_state,
            node_idx=node_idx,
            node_kind=node_kind,
            state_dict=state_dict,
            history=history,
        )
        return Store.from_config(self.config).proxy(jr)

    def proxy(self, data) -> Proxy:
        return Store.from_config(self.config).proxy(data)
