from typing import NewType, Union
from uuid import UUID

WorkerID = NewType("WorkerID", Union[int, str, UUID])

Indices = NewType("Indices", list[int])
