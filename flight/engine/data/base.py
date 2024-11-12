from __future__ import annotations

import abc
import typing as t

if t.TYPE_CHECKING:
    pass


class AbstractTransfer(abc.ABC):
    @abc.abstractmethod
    def transfer(self, data: t.Any) -> t.Any:
        """
        Abstract method to facilitate data transfer.
        """


class BaseTransfer(AbstractTransfer):
    def transfer(self, data: t.Any) -> t.Any:
        return data
