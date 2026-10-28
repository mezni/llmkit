"""llmkit: a small async LLM client library with swappable providers."""

from importlib.metadata import version

from llmkit.client import LLMClient
from llmkit.fake import FakeClient, FakeClientExhaustedError
from llmkit.summarizer import Summarizer
from llmkit.types import Message, Response, Role, Usage

__version__ = version("llmkit")

__all__ = [
    "FakeClient",
    "FakeClientExhaustedError",
    "LLMClient",
    "Message",
    "Response",
    "Role",
    "Summarizer",
    "Usage",
    "__version__",
]
