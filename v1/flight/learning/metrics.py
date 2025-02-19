from __future__ import annotations

import datetime as dt
import sys
import typing as t

DATE_RECORD_KEY = "date"

# if sys.version_info >= (3, 10):
#     from typing import TypeAlias
# else:
#     from typing_extensions import TypeAlias


if t.TYPE_CHECKING:
    from types import TracebackType

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self


class RecordLogger(t.Protocol):
    def log(self, **kwargs: t.Mapping[str, t.Any]) -> None:
        pass


class NullLogger:
    def __init__(self):
        pass

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None,
    ):
        return

    def log(self, **kwargs: t.Mapping[str, t.Any]) -> None:
        """Logs and records nothing."""
        return


class InMemoryRecordLogger:
    def __init__(self):
        self.records: list[t.Mapping[str, t.Any]] = []

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None,
    ):
        pass

    def log(self, **kwargs: t.Mapping[str, t.Any]) -> None:
        records: dict[str, t.Any] = {name: value for name, value in kwargs.items()}
        records.update({DATE_RECORD_KEY: dt.datetime.now()})
        self.records.append(records)


class JsonRecordLogger:
    pass
