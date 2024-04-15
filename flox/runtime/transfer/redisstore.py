from __future__ import annotations

import typing

from proxystore.connectors.endpoint import EndpointConnector
from proxystore.connectors.redis import RedisConnector
from proxystore.store import Store

from flox.runtime.transfer.base import BaseTransfer

if typing.TYPE_CHECKING:
    from uuid import UUID

    from proxystore.proxy import Proxy

    from flox.flock import Flock


class RedisTransfer(BaseTransfer):
    def __init__(
        self, ip_address, name: str = "default"
    ) -> None:  # , store: str = "endpoint",):
        self.connector = RedisConnector(hostname=ip_address, port=6379)
        store = Store(name=name, connector=self.connector)
        self.config = store.config()

    # TODO: Revisit this design to see if we need separate methods for `report` and `proxy`.
    def report(self, data: typing.Any) -> Proxy[typing.Any]:
        return Store.from_config(self.config).proxy(data)

    def proxy(self, data: typing.Any) -> Proxy[typing.Any]:
        return Store.from_config(self.config).proxy(
            data, populate_target=True, evict=True
        )
