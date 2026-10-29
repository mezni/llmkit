from typing import Literal, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """llmkit configuration, loaded from LLMKIT_* env vars or a .env file."""

    model_config = SettingsConfigDict(
        env_prefix="LLMKIT_",
        env_file=".env",
        extra="ignore",  # a shared .env may hold unrelated keys
        frozen=True,
    )

    provider: Literal["fake", "anthropic"] = "fake"
    model: str = "claude-sonnet-5-5"
    timeout_seconds: float = Field(default=30.0, gt=0, le=600)
    max_retries: int = Field(default=3, ge=0, le=10)
    api_key: SecretStr | None = None

    @field_validator("model")
    @classmethod
    def _model_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("model must not be blank")
        return value

    @model_validator(mode="after")
    def _anthropic_requires_key(self) -> Self:
        if self.provider == "anthropic" and self.api_key is None:
            raise ValueError("api_key is required when provider is 'anthropic'")
        return self
