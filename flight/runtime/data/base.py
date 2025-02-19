from __future__ import annotations

import typing as t


class DataPlane(t.Protocol):
    def transfer(self, data: t.Any) -> t.Any:
        """
        Abstract method to facilitate data transfer.

        Args:
            data (typing.Any): The data to be transferred.

        Returns:
            Reference to data after transfer protocol.
        """
