from __future__ import annotations

import abc
import typing as t

if t.TYPE_CHECKING:
    pass


class AbstractTransfer(abc.ABC):
    @abc.abstractmethod
    def __call__(self, data: t.Any) -> t.Any:
        pass


class BaseTransfer(AbstractTransfer):
    def __call__(self, data: t.Any) -> t.Any:
        return data
