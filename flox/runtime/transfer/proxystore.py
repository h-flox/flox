from typing import Any

from pandas import DataFrame
from proxystore.connectors.endpoint import EndpointConnector
from proxystore.proxy import Proxy
from proxystore.store import Store

from flox.flock import Flock, FlockNodeID, FlockNodeKind
from flox.flock.states import NodeState
from flox.runtime.result import JobResult
from flox.runtime.transfer.base import BaseTransfer
from flox.typing import StateDict


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
        self,
        node_state: NodeState | dict[str, Any] | None,
        node_idx: FlockNodeID | None,
        node_kind: FlockNodeKind | None,
        state_dict: StateDict | None,
        history: DataFrame | None,
    ) -> Proxy[JobResult]:
        result = JobResult(
            node_state=node_state,
            node_idx=node_idx,
            node_kind=node_kind,
            state_dict=state_dict,
            history=history,
        )
        return Store.from_config(self.config).proxy(result)

    def proxy(self, data: Any) -> Proxy[Any]:
        return Store.from_config(self.config).proxy(data)
