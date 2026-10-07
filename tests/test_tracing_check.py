from unittest.mock import Mock

from kv_cache_agent import tracing_check


def test_missing_key_is_reported_without_network(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.setattr(tracing_check, "tracing_is_enabled", lambda: True)
    client = Mock()
    monkeypatch.setattr(tracing_check, "Client", client)
    result = tracing_check.check_langsmith()
    assert result["status"] == "failed"
    assert result["error_type"] == "TracingConfigurationMissing"
    client.assert_not_called()


def test_provider_failure_does_not_expose_key_or_error_body(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "SECRET")
    monkeypatch.setattr(tracing_check, "tracing_is_enabled", lambda: True)
    monkeypatch.setattr(tracing_check, "Client", Mock())
    monkeypatch.setattr(
        tracing_check, "trace", Mock(side_effect=RuntimeError("SECRET auth response"))
    )
    result = tracing_check.check_langsmith()
    assert result["error_type"] == "RuntimeError"
    assert "SECRET" not in str(result)
