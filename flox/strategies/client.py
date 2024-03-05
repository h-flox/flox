import typing as t


class ClientStrategy(t.Protocol):
    def get_node_statuses(self):
        pass

    def select_worker_nodes(self):
        pass

    def before_share_params(self):
        pass
