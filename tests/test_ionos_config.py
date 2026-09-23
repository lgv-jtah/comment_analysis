import httpx
import pytest

from ionos_config import (
    IonosVerificationError,
    ModelRouteInfo,
    discover_models,
    verify_ionos_only,
)


class FakeModelInfoResponse:
    """Stand-in for httpx.Response returned by GET /v1/model/info."""

    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", "http://testserver/v1/model/info")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("error", request=request, response=response)


def _payload(entries):
    return {"data": entries}


def _entry(model_name, model, api_base):
    return {
        "model_name": model_name,
        "litellm_params": {"model": model, "api_base": api_base},
    }


# Real shape confirmed live against the LiteLLM proxy this session.
REALISTIC_PAYLOAD = _payload(
    [
        _entry(
            "mistral-nemo",
            "openai/mistralai/Mistral-Nemo-Instruct-2407",
            "https://openai.inference.de-txl.ionos.com/v1/",
        ),
        _entry(
            "llama-31-405b-fp8",
            "openai/meta-llama/Meta-Llama-3.1-405B-Instruct-FP8",
            "https://openai.inference.de-txl.ionos.com/v1/",
        ),
        _entry("claude-sonnet-4-6", "anthropic/claude-sonnet-4-6", None),
        _entry(
            "gpt-4o",
            "azure/my_azure_deployment",
            "https://NLP-2.services.ai.azure.com/openai/v1",
        ),
    ]
)


def test_discover_models_classifies_ionos_and_non_ionos_entries(monkeypatch):
    monkeypatch.setattr(
        "ionos_config.httpx.get", lambda *a, **k: FakeModelInfoResponse(REALISTIC_PAYLOAD)
    )

    routes = discover_models("http://localhost:4000", api_key="sk-test")

    assert routes["mistral-nemo"].is_ionos is True
    assert routes["mistral-nemo"].api_base == "https://openai.inference.de-txl.ionos.com/v1/"
    assert routes["mistral-nemo"].upstream_model == "openai/mistralai/Mistral-Nemo-Instruct-2407"
    assert routes["llama-31-405b-fp8"].is_ionos is True
    assert routes["claude-sonnet-4-6"].is_ionos is False
    assert routes["gpt-4o"].is_ionos is False


def test_discover_models_sends_bearer_auth_and_correct_url(monkeypatch):
    captured = {}

    def fake_get(url, headers=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        return FakeModelInfoResponse(_payload([]))

    monkeypatch.setattr("ionos_config.httpx.get", fake_get)

    discover_models("http://localhost:4000/", api_key="sk-test")

    assert captured["url"] == "http://localhost:4000/v1/model/info"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"


def test_discover_models_skips_entries_without_model_name(monkeypatch):
    payload = _payload([{"litellm_params": {"api_base": "https://x.ionos.com/v1/"}}])
    monkeypatch.setattr("ionos_config.httpx.get", lambda *a, **k: FakeModelInfoResponse(payload))

    routes = discover_models("http://localhost:4000", api_key="sk-test")

    assert routes == {}


def test_discover_models_propagates_http_errors(monkeypatch):
    monkeypatch.setattr(
        "ionos_config.httpx.get",
        lambda *a, **k: FakeModelInfoResponse({}, status_code=401),
    )

    with pytest.raises(httpx.HTTPStatusError):
        discover_models("http://localhost:4000", api_key="wrong-key")


def test_verify_ionos_only_passes_for_ionos_routed_model(monkeypatch):
    monkeypatch.setattr(
        "ionos_config.httpx.get", lambda *a, **k: FakeModelInfoResponse(REALISTIC_PAYLOAD)
    )

    info = verify_ionos_only("mistral-nemo", "http://localhost:4000", api_key="sk-test")

    assert isinstance(info, ModelRouteInfo)
    assert info.is_ionos is True


def test_verify_ionos_only_raises_for_non_ionos_model(monkeypatch):
    monkeypatch.setattr(
        "ionos_config.httpx.get", lambda *a, **k: FakeModelInfoResponse(REALISTIC_PAYLOAD)
    )

    with pytest.raises(IonosVerificationError, match="not IONOS"):
        verify_ionos_only("claude-sonnet-4-6", "http://localhost:4000", api_key="sk-test")


def test_verify_ionos_only_raises_for_unknown_model(monkeypatch):
    monkeypatch.setattr(
        "ionos_config.httpx.get", lambda *a, **k: FakeModelInfoResponse(REALISTIC_PAYLOAD)
    )

    with pytest.raises(IonosVerificationError, match="not present"):
        verify_ionos_only("some-model-not-configured", "http://localhost:4000", api_key="sk-test")
