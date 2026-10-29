import pytest

from llmkit import FakeClient, FakeClientExhaustedError, Message


def test_returns_scripted_replies_in_order() -> None:
    client = FakeClient(["one", "two"])
    assert client.generate([Message(role="user", content="a")]).text == "one"
    assert client.generate([Message(role="user", content="b")]).text == "two"


def test_records_what_was_sent() -> None:
    client = FakeClient(["ok"])
    msgs = [Message(role="system", content="be brief"), Message(role="user", content="hi there")]
    client.generate(msgs)
    assert client.calls == [msgs]


def test_reports_approximate_usage() -> None:
    client = FakeClient(["three word reply"])
    resp = client.generate([Message(role="user", content="two words")])
    assert resp.usage.input_tokens == 2
    assert resp.usage.output_tokens == 3


def test_raises_when_script_is_exhausted() -> None:
    client = FakeClient([])
    with pytest.raises(FakeClientExhaustedError):
        client.generate([Message(role="user", content="hi")])
