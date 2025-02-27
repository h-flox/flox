import pytest

from flight.strategy import _EnforceSuperMeta


def test_enforce_super_meta():
    class Invalid(metaclass=_EnforceSuperMeta):
        def __init__(self):
            self.name = "invalid"

    with pytest.raises(RuntimeError):
        Invalid()

    class InvalidChild(Invalid):
        def __init__(self):
            self.name = "invalid_child"

    with pytest.raises(RuntimeError):
        InvalidChild()

    class Valid(metaclass=_EnforceSuperMeta):
        def __init__(self):
            super().__init__()
            self._initialized = True
            self.name = "valid"

    assert isinstance(Valid(), Valid)

    class ValidWorkingChild(Valid):
        def __init__(self):
            super().__init__()
            self.name = "valid_working_child"

    inst = ValidWorkingChild()
    assert isinstance(inst, ValidWorkingChild)
    assert isinstance(inst, Valid)

    class ValidFailingChild(Valid):
        """This class should *fail* on construction."""
        def __init__(self):
            self.name = "valid_failing_child"

    with pytest.raises(RuntimeError):
        ValidFailingChild()

            
def test_strategy_event():
    pass
