from dataclasses import FrozenInstanceError

import pytest

from llmkit import Message, Response, Usage


def test_message_is_immutable() -> None:
    msg = Message("user", "hello")
    with pytest.raises(FrozenInstanceError):
        msg.content = "changed"  # type: ignore[misc]


def test_response_defaults_to_zero_usage() -> None:
    resp = Response(text="hi", model="m")
    assert resp.usage == Usage(0, 0)
