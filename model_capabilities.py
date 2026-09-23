"""Per-model structured-output/thinking-mode capability registry.

STANCE_PIPELINE_IMPLEMENTATION_BRIEF.md section 8: "Read the provider's
current structured-output capabilities. Use schema-constrained generation
where supported and always validate locally with Pydantic. If unsupported,
use a documented JSON mode/prompt fallback with bounded repair; report which
mode each run used. [...] Remove assumptions that every endpoint accepts
vLLM-specific thinking fields."

llm_handler.call_model already always requests schema-constrained JSON
output (response_format={"type": "json_schema", ...}) and always validates
locally with Pydantic downstream (policy.py, StanceDetector callers) -- that
part of the brief's requirement is structural, not per-model. What varies by
model is whether the *thinking-mode* extra_body field
(chat_template_kwargs.enable_thinking, a vLLM/Qwen-specific extension) is
safe to send: sending it to a model/backend that doesn't understand it can
be rejected outright by some OpenAI-compatible proxies. This registry is the
single place that answers "should enable_thinking be sent for this model,"
so callers don't have to hardcode it (as main.py's STANCE_LLM_ENABLE_THINKING
env var and its docstring already do today, informally) and it can gain
entries as new IONOS-hosted models are added to the comparison matrix
(brief section 8's Mistral Small 24B / Llama 3.3 70B / GPT-OSS 120B).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelCapabilities:
    """What is known/assumed about a model's request-shape support.

    Attributes:
        supports_structured_output: Whether json_schema-constrained output is
            known to work for this model. True for every model verified this
            session (all requests already use it); conservatively True by
            default too, since llm_handler always requests it regardless and
            always validates the result locally with Pydantic -- a model
            that silently ignores the schema still gets caught by
            validation, it doesn't get a free pass.
        supports_thinking_mode: Whether the vLLM/Qwen-specific
            chat_template_kwargs.enable_thinking extra_body field is safe to
            send. False by default for any model not explicitly verified --
            sending an unsupported extension can cause some OpenAI-compatible
            proxies/backends to reject the request outright, so the
            conservative default is to omit it.
        notes: Free-form context for why this entry has these values.
    """

    supports_structured_output: bool = True
    supports_thinking_mode: bool = False
    notes: str = ""


# Verified via this session's own testing: all three models are accessed
# through the LiteLLM proxy's OpenAI-compatible surface (openai/... litellm
# model prefix), routed to IONOS, with STANCE_LLM_ENABLE_THINKING=false in
# the working .env -- none of them are raw vLLM/Qwen endpoints that the
# original thinking-mode code was written for.
_KNOWN_MODELS: dict[str, ModelCapabilities] = {
    "mistral-nemo": ModelCapabilities(
        supports_structured_output=True,
        supports_thinking_mode=False,
        notes="IONOS-routed via LiteLLM proxy; used throughout this project's "
        "prompt experiments and the policy v2 smoke tests with thinking mode off.",
    ),
    "llama-31-405b-fp8": ModelCapabilities(
        supports_structured_output=True,
        supports_thinking_mode=False,
        notes="IONOS-routed via LiteLLM proxy (confirmed in the proxy's live "
        "/v1/model/info); not independently exercised by this project yet.",
    ),
    "gptoss-120b": ModelCapabilities(
        supports_structured_output=True,
        supports_thinking_mode=False,
        notes="IONOS-routed via LiteLLM proxy (confirmed in the proxy's live "
        "/v1/model/info); not independently exercised by this project yet.",
    ),
}

# Conservative fallback for any model not in _KNOWN_MODELS -- e.g. a newly
# added IONOS catalog entry (brief section 8's Mistral Small 24B / Llama 3.3
# 70B) that hasn't been added here yet.
DEFAULT_CAPABILITIES = ModelCapabilities(
    supports_structured_output=True,
    supports_thinking_mode=False,
    notes="Unverified model: defaulting to schema-constrained output (always "
    "validated locally regardless) and thinking mode disabled.",
)


def get_capabilities(model_name: str) -> ModelCapabilities:
    """Look up a model's known capabilities, falling back to the conservative default.

    Args:
        model_name: The model identifier as configured (e.g.
            STANCE_LLM_MODEL_NAME), matching the proxy's model_name.

    Returns:
        The registered ModelCapabilities, or DEFAULT_CAPABILITIES if the
        model isn't registered.
    """
    return _KNOWN_MODELS.get(model_name, DEFAULT_CAPABILITIES)
