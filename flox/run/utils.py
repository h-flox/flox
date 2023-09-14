from __future__ import annotations

from concurrent.futures import Future
from typing import Any


def set_parent_future(parent_future: Future, child_future: Future) -> Any:
    """Sets the result of the `parent_future` to the result of its `child_future` and returns it.

    Args:
        parent_future (Future): The parent Future.
        child_future (Future): The child Future.

    Returns:
        The result of `child_future` which is now set to be the result of `parent_future`.
    """
    assert child_future.done()
    if child_future.exception():
        parent_future.set_exception(child_future.exception())
    else:
        result = child_future.result()
        try:
            parent_future.set_result(result)
        except Exception as ex:
            print(ex)  # TODO: Log this better.
        return result
