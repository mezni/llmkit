import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from llmkit import Settings


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)  # no stray .env gets picked up
    for key in [k for k in os.environ if k.startswith("LLMKIT_")]:
        monkeypatch.delenv(key)


def test_defaults() -> None:
    s = Settings()
    assert s.provider == "fake"
    assert s.timeout_seconds == 30.0
    assert s.max_retries == 3
    assert s.api_key is None


def test_reads_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMKIT_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("LLMKIT_MAX_RETRIES", "5")
    s = Settings()
    assert s.timeout_seconds == 12.5
    assert s.max_retries == 5


def test_reads_dotenv_and_env_var_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("LLMKIT_MAX_RETRIES=7\nLLMKIT_MODEL=from-file\n")
    assert Settings().max_retries == 7
    monkeypatch.setenv("LLMKIT_MAX_RETRIES", "2")
    s = Settings()
    assert s.max_retries == 2  # env var beats .env
    assert s.model == "from-file"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("LLMKIT_TIMEOUT_SECONDS", "0"),
        ("LLMKIT_TIMEOUT_SECONDS", "abc"),
        ("LLMKIT_MAX_RETRIES", "-1"),
        ("LLMKIT_MAX_RETRIES", "99"),
        ("LLMKIT_PROVIDER", "openai"),
        ("LLMKIT_MODEL", "   "),
    ],
)
def test_rejects_invalid_values(monkeypatch: pytest.MonkeyPatch, name: str, value: str) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        Settings()


def test_anthropic_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMKIT_PROVIDER", "anthropic")
    with pytest.raises(ValidationError, match="api_key is required"):
        Settings()


def test_api_key_is_hidden_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMKIT_PROVIDER", "anthropic")
    monkeypatch.setenv("LLMKIT_API_KEY", "sk-secret-value")
    s = Settings()
    assert "sk-secret-value" not in repr(s)
    assert s.api_key is not None
    assert s.api_key.get_secret_value() == "sk-secret-value"


def test_settings_are_frozen() -> None:
    s = Settings()
    with pytest.raises(ValidationError):
        s.max_retries = 9  # type: ignore[misc]
