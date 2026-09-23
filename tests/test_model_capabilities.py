from model_capabilities import DEFAULT_CAPABILITIES, ModelCapabilities, get_capabilities


def test_known_model_returns_its_registered_capabilities():
    caps = get_capabilities("mistral-nemo")

    assert isinstance(caps, ModelCapabilities)
    assert caps.supports_structured_output is True
    assert caps.supports_thinking_mode is False


def test_unknown_model_falls_back_to_conservative_default():
    caps = get_capabilities("some-brand-new-model-not-registered-yet")

    assert caps == DEFAULT_CAPABILITIES
    assert caps.supports_thinking_mode is False


def test_all_registered_models_default_thinking_mode_off():
    # Every model this project has actually used is accessed through the
    # LiteLLM->IONOS OpenAI-compatible surface, not a raw vLLM/Qwen endpoint.
    for name in ("mistral-nemo", "llama-31-405b-fp8", "gptoss-120b"):
        assert get_capabilities(name).supports_thinking_mode is False
