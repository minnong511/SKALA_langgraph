import pytest
from pydantic import ValidationError

from src.config import Settings


def test_environment_overrides_dotenv_without_exposing_secrets(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_API_KEY=synthetic-secret\nMAX_SEARCH_RETRIES=1\nLLM_MODEL=chosen-model\n")
    monkeypatch.setenv("MAX_SEARCH_RETRIES", "0")
    settings = Settings.from_env(env_file)
    assert settings.limits.max_search_retries == 0
    assert settings.llm_model == "chosen-model"
    assert "synthetic-secret" not in str(settings.public_config())
    assert settings.secrets == ["synthetic-secret"]


def test_live_configuration_requires_exact_model_and_credentials():
    with pytest.raises(ValueError, match="LLM_MODEL"):
        Settings().validate_live()
    Settings(llm_model="chosen-model", llm_api_key="test", tavily_api_key="test").validate_live()
    with pytest.raises(ValidationError):
        Settings(max_search_retries=-1)
