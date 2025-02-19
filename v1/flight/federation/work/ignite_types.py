from __future__ import annotations

import typing as t

from v1.flight.learning import AbstractDataModule, AbstractModule
from v1.flight.topologies import Node, NodeState

"""
Here, we have a challenge of defining the arguments needed for a worker job function.

If we want the function to be minimally runnable, we need data, model, and some setup
for which optimizer and loss function to configure (this might be done with the
`TorchModule` or we might incorporate it into the worker's strategy). By
"minimally runnable" I mean that this function can be run locally for testing
to ensure that the model is able to be trained on data. For this, we do not need
to have information about the node, parent, or other hierarchical setups. It's also
ideal for end users of Flight to have this functionality such that they can more
rapidly prototype with the event handlers within Ignite to ensure it works as expected.

However, if we need to run this on a remote worker, we need to have information about
the node, parent, and other hierarchical setups.

So, we need to have some type of definition of the arguments and their types to make
this more configurable and easier to follow for knowing which type of execution to do.

Also, it will be helpful in the event that we extend the customizability of Flight
to include complete user-provided local work jobs that can be run on remote workers.
So long as users adhere to a given set of parameters, they're good to go!
"""


class TrainingJobSignature(t.TypedDict):
    model: t.NotRequired[AbstractModule]
    data: t.NotRequired[AbstractDataModule]
    state: t.NotRequired[NodeState]
    node: t.NotRequired[Node]
    parent: t.NotRequired[Node]
    work_event_handlers: t.NotRequired[t.Any]
    learning_event_handlers: t.NotRequired[t.Any]


"""
Below is a simple example of how we might do validation of dictionaries to infer
which type of execution to run (i.e., no training, launch job on remote workers
with/without training, etc.).

```python
import typing as t
from typing_extensions import TypedDict

from pydantic import TypeAdapter, ValidationError


class Foo(TypedDict):
	name: t.Required[str]
	age: t.NotRequired[int]


Validator = TypeAdapter(Foo)

d = {}
Validator.validate_python(d)
Validator.validate_python({"ssn": 1})
```
"""
