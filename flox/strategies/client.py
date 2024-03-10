import typing as t


class ClientStrategy(t.Protocol):
    # def get_node_statuses(self):
    #     pass

    def select_worker_nodes(self, state, children, seed):
        return children

    # def before_share_params(self):
    #     pass
