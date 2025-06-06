import pytest

from flight.events import on, CoordinatorEvents
from flight.strategy import (
    _EnforceSuperMeta,
    _SUPER_META_FLAG,
    Strategy,
    DefaultStrategy,
)


def test_enforce_super_meta():
    class Invalid(metaclass=_EnforceSuperMeta):
        def __init__(self):
            self.name = "invalid"

    with pytest.raises(AttributeError):
        Invalid()

    class InvalidChild(Invalid):
        def __init__(self):  # noqa
            self.name = "invalid_child"

    with pytest.raises(AttributeError):
        InvalidChild()

    class Valid(metaclass=_EnforceSuperMeta):
        def __init__(self):
            super().__init__()
            setattr(self, _SUPER_META_FLAG, True)
            self.name = "valid"

    assert isinstance(Valid(), Valid)

    ########################################################################

    class ValidWorkingChild(Valid):
        def __init__(self):
            super().__init__()
            self.name = "valid_working_child"

    inst = ValidWorkingChild()
    assert isinstance(inst, ValidWorkingChild)
    assert isinstance(inst, Valid)

    ########################################################################

    class ValidFailingChild(Valid):
        """This class should *fail* on construction."""

        def __init__(self):  # noqa
            self.name = "valid_failing_child"

    with pytest.raises(AttributeError):
        ValidFailingChild()


def test_strategy_def_with_explicit_policies():
    class TestStrategy(Strategy):
        def __init__(self):
            super().__init__()

        def aggregation_policy(self, *args, **kwargs):
            return  # TODO: Change this when we have this working in `fitter.py`.

        def selection_policy(self, *args, **kwargs):
            return  # TODO: Change this when we have this working in `fitter.py`.

    assert isinstance(TestStrategy(), Strategy)


def test_strategy_def_with_arg_policies():
    def aggregation_policy(self, *args, **kwargs):
        return  # TODO: Change this when we have this working in `fitter.py`.

    def selection_policy(self, *args, **kwargs):
        return  # TODO: Change this when we have this working in `fitter.py`.

    assert isinstance(Strategy(aggregation_policy, selection_policy), Strategy)


def test_strategy_event():
    class TestStrategy(DefaultStrategy):
        def __init__(self):
            super().__init__()

        @on(CoordinatorEvents.STARTED)
        def test_event_handler(self, context):
            context["event_handled"] = True

    strategy = TestStrategy()
    with pytest.raises(ValueError):
        strategy.add_event_handler(
            CoordinatorEvents.STARTED, strategy.test_event_handler
        )
