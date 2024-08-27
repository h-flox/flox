from __future__ import annotations

import typing as t
from uuid import UUID

import cloudpickle
from proxystore.connectors.endpoint import EndpointConnector
from proxystore.store import Store

from ...federation.topologies import Topology
from .base import AbstractTransfer

if t.TYPE_CHECKING:
    from proxystore.proxy import Proxy


class ProxystoreTransfer(AbstractTransfer):
    def __init__(self, topo: Topology, name: str = "default") -> None:
        if not topo.proxystore_ready:
            raise ValueError(
                "Flock is not ready to use ProxyStore (i.e., "
                "`topologies.proxystore_ready` returns `False`). You need each node "
                "should have a valid ProxyStore Endpoint UUID."
            )

        endpoints: list[UUID | str] = []
        for node in topo.nodes():
            if not isinstance(node.proxystore_id, UUID):
                raise ValueError(f"`{node.proxystore_id=} is not a UUID.")
            else:
                endpoints.append(node.proxystore_id)

        self.name = name
        self.connector = EndpointConnector(endpoints=endpoints)
        store = Store(
            name=name,
            connector=self.connector,
            serializer=cloudpickle.dumps,
            deserializer=cloudpickle.loads,
        )
        self.config = store.config()

    def __call__(self, data: t.Any) -> Proxy[t.Any]:
        return Store.from_config(self.config).proxy(data)
