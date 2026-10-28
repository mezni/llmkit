from collections import deque
from collections.abc import Sequence

from llmkit.types import Message, Response, Usage


class FakeClientExhaustedError(RuntimeError):
    """Raised when a test asks for more replies than it scripted."""


class FakeClient:
    """Deterministic stand-in for a model. Returns scripted replies in order."""

    def __init__(self, replies: Sequence[str], model: str = "fake-model") -> None:
        self._replies = deque(replies)
        self._model = model
        self.calls: list[list[Message]] = []

    def generate(self, messages: Sequence[Message]) -> Response:
        self.calls.append(list(messages))
        if not self._replies:
            raise FakeClientExhaustedError("FakeClient has no scripted replies left")
        text = self._replies.popleft()
        usage = Usage(
            input_tokens=sum(len(m.content.split()) for m in messages),
            output_tokens=len(text.split()),
        )
        return Response(text=text, model=self._model, usage=usage)
