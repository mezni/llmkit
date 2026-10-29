from llmkit.client import LLMClient
from llmkit.types import Message


class Summarizer:
    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def summarize(self, text: str) -> str:
        messages = [
            Message(role="system", content="Summarize the text in one sentence."),
            Message(role="user", content=text),
        ]
        return self._client.generate(messages).text
