import pytest
from pydantic import ValidationError

from v1.flight.topologies.node import (
    AggrState,
    Node,
    NodeKind,
    NodeState,
    WorkerState,
)


class TestNodeInits:
    @staticmethod
    def mini_test(should_work: bool, **kwargs):
        if should_work:
            node = Node(**kwargs)
            assert isinstance(node, Node)
        else:
            with pytest.raises(ValidationError):
                Node(**kwargs)
        # try:
        #     Node(**kwargs)
        #     assert should_work
        # except ValidationError:
        #     assert not should_work

    def test_valid_inits(self):
        should_init = True
        TestNodeInits.mini_test(
            should_init,
            idx=123,
            kind=NodeKind.WORKER,
            extra={"battery_cap": 10},
        )

        TestNodeInits.mini_test(
            should_init,
            idx=123,
            kind="worker",
            extra={"battery_cap": 10},
        )

    def test_invalid_inits(self):
        should_init = False
        TestNodeInits.mini_test(
            should_init,
            extra={"battery_cap": 10},
        )

        TestNodeInits.mini_test(
            should_init,
            idx=10,
            kind="hello",
            extra={"battery_cap": 10},
        )


class TestNodeUse:
    def test_node_extra_dicts(self):
        key, val = "foo", "bar"
        no_extra_node = Node(idx=0, kind="worker")

        assert no_extra_node.extra is None
        with pytest.raises(KeyError):
            _ = no_extra_node.get(key, val) == val
        with pytest.raises(KeyError):
            _ = no_extra_node[key]

        no_extra_node[key] = val
        assert no_extra_node[key] == "bar"


class TestNodeState:
    def test_state_init(self):
        with pytest.raises(TypeError):
            NodeState(1)

        children = [Node(idx=1, kind=NodeKind.WORKER), Node(idx=2, kind=NodeKind.AGGR)]
        state = AggrState(1, children)
        assert isinstance(state, NodeState)
        assert isinstance(state, AggrState)

        state = WorkerState(1)
        assert isinstance(state, NodeState)
        assert isinstance(state, WorkerState)
