class Params:
    pass


class BaseModel:
    def __init__(self):
        pass

    def get_params(self) -> Params:
        pass

    def set_params(self, params: Params) -> None:
        pass
