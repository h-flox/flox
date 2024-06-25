from __future__ import annotations

import typing

from proxystore.connectors.redis import RedisConnector
from proxystore.store import Store

from flox.runtime.transfer.base import TransferProtocol

if typing.TYPE_CHECKING:
    from proxystore.proxy import Proxy


class RedisTransfer(TransferProtocol):
    def __init__(
        self, ip_address, name: str = "default"
    ) -> None:  # , store: str = "endpoint",):
        self.connector = RedisConnector(hostname=ip_address, port=6379)
        store = Store(name=name, connector=self.connector)
        self.config = store.config()

    def transfer(self, data: typing.Any) -> Proxy[typing.Any]:
        return Store.from_config(self.config).proxy(data)
