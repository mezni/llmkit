from collections.abc import Sequence
from typing import Protocol

from llmkit.types import Message, Response


class LLMClient(Protocol):
    """Anything that can turn a conversation into a model response."""

    def generate(self, messages: Sequence[Message]) -> Response: ...
