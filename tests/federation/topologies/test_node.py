from pydantic import ValidationError

from flight.federation.topologies.node import Node, NodeKind


def test_node_inits():
    def mini_test(should_work: bool, **kwargs):
        try:
            Node(**kwargs)
            assert should_work
        except ValidationError:
            assert not should_work

    mini_test(
        True,
        idx=123,
        kind=NodeKind.WORKER,
        extra={"battery_cap": 10},
    )

    mini_test(
        False,
        extra={"battery_cap": 10},
    )

    mini_test(
        False,
        idx=10,
        kind="hello",
        extra={"battery_cap": 10},
    )
