from collections.abc import Sequence

from llmkit import FakeClient, Message, Response, Summarizer


def test_summarizer_returns_model_text() -> None:
    summarizer = Summarizer(FakeClient(["A short summary."]))
    assert summarizer.summarize("long text...") == "A short summary."


def test_summarizer_sends_instruction_and_text() -> None:
    fake = FakeClient(["ok"])
    Summarizer(fake).summarize("the document")
    sent = fake.calls[0]
    assert sent[0].role == "system"
    assert sent[1] == Message(role="user", content="the document")


class UppercaseClient:
    """A different implementation that does NOT inherit from anything."""

    def generate(self, messages: Sequence[Message]) -> Response:
        return Response(text=messages[-1].content.upper(), model="upper")


def test_any_matching_class_can_be_injected() -> None:
    # mypy verifies UppercaseClient satisfies LLMClient structurally.
    assert Summarizer(UppercaseClient()).summarize("hello") == "HELLO"
