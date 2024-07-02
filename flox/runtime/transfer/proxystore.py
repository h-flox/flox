from __future__ import annotations

import typing as t

import cloudpickle
from proxystore.connectors.endpoint import EndpointConnector
from proxystore.store import Store

if t.TYPE_CHECKING:
    from uuid import UUID

    from proxystore.proxy import Proxy

    from flox.federation.topologies import Topology


class ProxyStoreTransfer:
    def __init__(self, flock: Topology, name: str = "default") -> None:
        if not flock.proxystore_ready:
            raise ValueError(
                "Flock is not ready to use ProxyStore (i.e., `topologies.proxystore_ready` "
                "returns `False`). You need each node should have a valid ProxyStore "
                "Endpoint UUID."
            )

        endpoints: list[str | UUID] = []
        for node in flock.nodes():
            assert node.proxystore_endpoint is not None
            endpoints.append(node.proxystore_endpoint)

        self.connector = EndpointConnector(endpoints=endpoints)
        store = Store(
            name=name,
            connector=self.connector,
            serializer=cloudpickle.dumps,
            deserializer=cloudpickle.loads,
        )
        self.config = store.config()

    def transfer(self, data: t.Any) -> Proxy[t.Any]:
        return Store.from_config(self.config).proxy(data)
