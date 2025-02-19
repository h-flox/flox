from v1.flight.types import Record


def broadcast_records(records: list[Record], **kwargs) -> None:
    """
    Broadcasts the key-value pairs in `kwargs` to all the records in the `records` list.

    Args:
        records (list[Record]): List of records to broadcast the key-value pairs to.
        **kwargs: Key-value pairs to broadcast to the records.

    Notes:
        If a key already exists in a record, the value will be overwritten.

    Examples:
        >>> records = [
        >>>     {"name": "Alice", "age": 20},
        >>>     {"name": "Bob", "age": 19},
        >>> ]
        >>> records
        [{'name': 'Alice', 'age': 20}, {'name': 'Bob', 'age': 19}]
        >>> broadcast_records(records, foo="bar")
        >>> records
        [{'name': 'Alice', 'age': 20, 'foo': 'bar'}, {'name': 'Bob', 'age': 19,
        'foo': 'bar'}]
    """
    if len(kwargs) == 0:
        return None

    for rec in records:
        for key, value in kwargs.items():
            rec[key] = value
