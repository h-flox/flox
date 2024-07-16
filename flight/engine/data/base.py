import typing as t


class BaseTransfer:
    def __call__(self, data: t.Any) -> t.Any:
        return data
