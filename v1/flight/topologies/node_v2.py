import copy
import typing as t

from v1.flight.topologies.node import NodeKind, _node_kind_or_str

if t.TYPE_CHECKING:
    from uuid import UUID

    from ignite.engine import Engine, Events

    from v1.flight.topologies.node import NodeID, NodeKindType

    WorkerLocalState: t.TypeAlias = dict[str, t.Any]
    EventHandler: t.TypeAlias = t.Callable[[Engine, WorkerLocalState], None]


class JobArgs:
    pass


class AggregatorJobArgs(JobArgs):
    pass


class WorkerJobArgs(JobArgs):
    train_handlers: t.Iterable[tuple[Events, EventHandler]] | None = None
    validate_handlers: t.Iterable[tuple[Events, EventHandler]] | None = None
    eval_handlers: t.Iterable[tuple[Events, EventHandler]] | None = None


class NodeV2:
    idx: NodeID
    kind: NodeKindType
    globus_compute_id: UUID | None = None
    proxystore_id: UUID | None = None
    cache: dict[str, t.Any] | None = None

    def __init__(
        self,
        idx: NodeID,
        kind: NodeKindType,
        globus_compute_id: UUID | None = None,
        proxystore_id: UUID | None = None,
        deep_copy_kwargs: bool = False,
        **kwargs: dict[str, t.Any],
    ) -> None:
        """

        Args:
            idx (NodeID): The ID of the node. This value must be distinct across all
                nodes in a topology.
            kind (NodeKindType): The kind of node instantiated. This can be a string or
                an instance of the `NodeKind` enum. String values will be converted to
                the corresponding enum value. The kind of node determines its role in
                the federation.
            globus_compute_id (UUID | None): The UUID of the Globus Compute that is
                used for remote execution. Default is `None`.
            proxystore_id (UUID | None): The UUID of the ProxyStore that is used for
                data transfer for remote execution with Globus Compute. Default is
                `None`.
            deep_copy_kwargs (bool): Whether to deep copy the values passed into the
                cache. Default is `False`.
            **kwargs: Keyword arguments that are passed into the node's `cache`. These
                can be node-specific parameters that are used in user-defined logic
                for FL tasks.

        Notes:
            - The values of passed into the cache directly through `**kwargs` are
                copied into the cache. By default, this is done in a shallow way. Some
                data (e.g., lists) will have the reference copied and this could have
                side effects since they could be shared across multiple nodes. To avoid
                this issue, set `deep_copy_kwargs=True` to ensure that the data is
                copied in a deep way.
        """
        self.idx = idx
        self.kind = _node_kind_or_str(kind)
        self.globus_compute_id = globus_compute_id
        self.proxystore_id = proxystore_id
        self.cache = {
            key: copy.deepcopy(val) if deep_copy_kwargs else val
            for key, val in kwargs.items()
        }

    def prepare_job_args(self) -> JobArgs:
        match self.kind:
            case NodeKind.COORD:
                raise ValueError("Cannot prepare job args for a coordinator node.")
            case NodeKind.AGGR:
                return AggregatorJobArgs(...)
            case NodeKind.WORKER:
                return WorkerJobArgs(...)
            case _:
                raise ValueError("Illegal value for node kind.")

    def get(self, key: str, default: t.Any | None = None) -> t.Any:
        if default:
            return self.cache[key]
        else:
            return self.cache.get(key, default)

    def set(self, key: str, value: t.Any) -> None:
        self.cache[key] = value
