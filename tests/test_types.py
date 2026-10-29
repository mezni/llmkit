import pytest
from pydantic import ValidationError

from llmkit import Message, Response, Usage


def test_message_is_immutable() -> None:
    msg = Message(role="user", content="hello")
    with pytest.raises(ValidationError):
        msg.content = "changed"  # type: ignore[misc]


def test_message_rejects_empty_content() -> None:
    with pytest.raises(ValidationError):
        Message(role="user", content="")


def test_message_rejects_unknown_role() -> None:
    with pytest.raises(ValidationError):
        Message.model_validate({"role": "robot", "content": "hi"})


def test_message_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Message.model_validate({"role": "user", "content": "hi", "tone": "calm"})


def test_usage_rejects_negative_tokens() -> None:
    with pytest.raises(ValidationError):
        Usage(input_tokens=-1)


def test_usage_total() -> None:
    assert Usage(input_tokens=2, output_tokens=3).total_tokens == 5


def test_response_defaults_to_zero_usage() -> None:
    assert Response(text="hi", model="m").usage == Usage()


def test_message_round_trips_through_dict() -> None:
    msg = Message.model_validate({"role": "user", "content": "hi"})
    assert msg.model_dump() == {"role": "user", "content": "hi"}


def test_json_schema_lists_allowed_roles() -> None:
    schema = Message.model_json_schema()
    assert schema["properties"]["role"]["enum"] == ["system", "user", "assistant"]
