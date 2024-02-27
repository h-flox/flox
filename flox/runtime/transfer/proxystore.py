from __future__ import annotations

import typing

from proxystore.connectors.endpoint import EndpointConnector
from proxystore.store import Store

from flox.runtime.transfer.base import BaseTransfer

if typing.TYPE_CHECKING:
    from uuid import UUID
    from proxystore.proxy import Proxy
    from flox.flock import Flock


class ProxyStoreTransfer(BaseTransfer):
    def __init__(
        self, flock: Flock, name: str = "default"
    ) -> None:  # , store: str = "endpoint",):
        if not flock.proxystore_ready:
            raise ValueError(
                "Flock is not ready to use ProxyStore (i.e., `flock.proxystore_ready` "
                "returns `False`). You need each node should have a valid ProxyStore "
                "Endpoint UUID."
            )

        endpoints: list[str | UUID] = []
        for node in flock.nodes():
            assert node.proxystore_endpoint is not None
            endpoints.append(node.proxystore_endpoint)

        self.connector = EndpointConnector(endpoints=endpoints)
        store = Store(name=name, connector=self.connector)
        self.config = store.config()

    # TODO: Revisit this design to see if we need separate methods for `report` and `proxy`.
    def report(self, data: typing.Any) -> Proxy[typing.Any]:
        return Store.from_config(self.config).proxy(data)

    def proxy(self, data: typing.Any) -> Proxy[typing.Any]:
        return Store.from_config(self.config).proxy(data)
