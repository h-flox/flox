from __future__ import annotations

import abc
import typing as t


class AbstractTransporter(abc.ABC):
    @abc.abstractmethod
    def transfer(self, data: t.Any) -> t.Any:
        """
        Abstract method to facilitate data transfer.
        """


class InMemoryTransporter(AbstractTransporter):
    """
    An in-memory transporter that simply returns the data as-is.

    This class does nothing fancy, it simply returns the data as-is. The need
    for this class is that it adheres to the `AbstractTransporter` standard.
    """

    def transfer(self, data: t.Any) -> t.Any:
        return data
