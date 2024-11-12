from __future__ import annotations

import typing as t
from uuid import UUID

import cloudpickle
from proxystore.connectors.endpoint import EndpointConnector
from proxystore.store import Store, get_or_create_store, get_store

from ...federation.topologies import Topology
from .base import AbstractTransporter

if t.TYPE_CHECKING:
    from proxystore.proxy import Proxy

T = t.TypeVar("T")


class ProxystoreTransfer(AbstractTransfer):
    def __init__(
        self,
        topo: Topology,
        evict: bool = False,
        name: str = "default",
    ) -> None:
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

        store = get_store(name)
        if store is None:
            store = Store(
                name=name,
                connector=EndpointConnector(endpoints=endpoints),
                # In the future, these could be customized (de)serializers
                # that are optimized for Flight/PyTorch models.
                serializer=cloudpickle.dumps,
                deserializer=cloudpickle.loads,
                register=True,
            )
        self.evict = evict
        self.store = store

    def __call__(self, data: T) -> Proxy[T]:
        # evict=True is only safe when it is guarenteed that the proxy will
        # only be used by a single consumer.
        return self.store.proxy(data, evict=self.evict)

    def __getstate__(self) -> dict[str, t.Any]:
        # Customize pickle behavior so that stateful objects are not pickled.
        return {"config": self.store.config(), "evict": self.evict}

    def __setstate__(self, state: dict[str, t.Any]) -> None:
        # Initialized an object from its pickled state.
        self.evict = state["evict"]
        self.store = get_or_create_store(state["config"], register=True)
