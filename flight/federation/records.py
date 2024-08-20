from flight.types import Record

# record: Record = []


def join_records(a: list[Record], b: list[Record]) -> list[Record]:
    pass


def broadcast_records(records: list[Record], **kwargs) -> None:
    if len(kwargs) == 0:
        return None

    for rec in records:
        for key, value in kwargs.items():
            rec[key] = value
