"""`personalos.models.structured`: the bounded repair step every model boundary shares."""

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from personalos.models.structured import (
    REPAIR_PROMPT,
    ainvoke_structured,
    invoke_structured,
)
from tests.fixtures.node_contracts import ReplayChatModel


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: int


TURNS = [("system", "answer with a number"), ("human", "how many?")]


def _runnable(*replies):
    model = ReplayChatModel(*replies)
    return model.with_structured_output(Answer), model


def test_a_valid_first_answer_is_returned_without_a_second_call():
    runnable, model = _runnable({"value": 3})

    assert invoke_structured(runnable, TURNS, Answer, span="t", model=None) == Answer(value=3)
    assert len(model.calls) == 1


def test_an_instance_of_the_schema_is_returned_as_is():
    runnable, _model = _runnable(Answer(value=4))

    assert invoke_structured(runnable, TURNS, Answer, span="t", model=None) == Answer(value=4)


def test_an_invalid_answer_is_repaired_with_the_original_turns_and_the_error_kind():
    runnable, model = _runnable({"value": "many"}, {"value": 5})

    assert invoke_structured(runnable, TURNS, Answer, span="t", model=None) == Answer(value=5)
    assert model.calls[1] == [*TURNS, ("human", REPAIR_PROMPT.format(error="ValidationError"))]


def test_a_parser_error_is_a_schema_error_too():
    runnable, model = _runnable(ValueError("could not parse"), {"value": 6})

    assert invoke_structured(runnable, TURNS, Answer, span="t", model=None) == Answer(value=6)
    assert model.calls[1][-1] == ("human", REPAIR_PROMPT.format(error="ValueError"))


def test_after_max_attempts_the_last_error_is_raised_for_the_callers_fallback():
    runnable, model = _runnable({"value": "a"}, {"value": "b"}, {"value": 1})

    with pytest.raises(ValidationError):
        invoke_structured(runnable, TURNS, Answer, span="t", model=None, max_attempts=2)
    assert len(model.calls) == 2


def test_a_provider_error_is_raised_at_once():
    runnable, model = _runnable(RuntimeError("503"), {"value": 1})

    with pytest.raises(RuntimeError):
        invoke_structured(runnable, TURNS, Answer, span="t", model=None)
    assert len(model.calls) == 1


@pytest.mark.parametrize("attempts", [0, -1])
def test_at_least_one_attempt_is_required(attempts):
    runnable, _model = _runnable({"value": 1})

    with pytest.raises(ValueError):
        invoke_structured(runnable, TURNS, Answer, span="t", model=None, max_attempts=attempts)


async def test_the_async_variant_repairs_the_same_way():
    runnable, model = _runnable({"value": None}, {"value": 7})

    result = await ainvoke_structured(runnable, TURNS, Answer, span="t", model=None)

    assert result == Answer(value=7)
    assert len(model.calls) == 2


async def test_the_async_variant_raises_after_max_attempts():
    runnable, _model = _runnable({"value": None}, {"value": None})

    with pytest.raises(ValidationError):
        await ainvoke_structured(runnable, TURNS, Answer, span="t", model=None)
