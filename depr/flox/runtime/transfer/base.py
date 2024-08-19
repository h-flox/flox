import typing as t


class TransferProtocol(t.Protocol):
    def transfer(self, data: t.Any) -> t.Any:
        """
        Transfer data method.

        Args:
            data (t.Any): Data to transfer across the network.

        Returns:
            Transferred object.
        """


class Transfer:
    def transfer(self, data: t.Any) -> t.Any:  # noqa
        return data
