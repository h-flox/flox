from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from concurrent.futures import Future


class ControlPlane(t.Protocol):
    def submit(self, fn: t.Callable, /, *args, **kwargs) -> Future:
        """
        ...

        Args:
            fn:
            *args:
            **kwargs:

        Returns:

        """
