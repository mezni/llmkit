from llmkit.client import LLMClient
from llmkit.types import Message


class Summarizer:
    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def summarize(self, text: str) -> str:
        messages = [
            Message("system", "Summarize the text in one sentence."),
            Message("user", text),
        ]
        return self._client.generate(messages).text
