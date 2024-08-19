from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    pass


class TransferProto(t.Protocol):
    def __call__(self, data: t.Any) -> t.Any:
        pass
