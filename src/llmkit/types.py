from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant"]


class _Frozen(BaseModel):
    """Immutable, strict-about-extras base for all boundary types."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Message(_Frozen):
    role: Role
    content: str = Field(min_length=1)


class Usage(_Frozen):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class Response(_Frozen):
    text: str
    model: str
    usage: Usage = Field(default_factory=Usage)
