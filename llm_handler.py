"""Handles the communication with the llm"""
from pydantic import BaseModel
import httpx
import json
import asyncio

# ── Configuration ──────────────────────────────────────────────────

MAX_TOKENS = 8000
THINKING_BUDGET = 6000
RATE_LIMIT_DELAY = 0.5

STANCE_LABEL_MAP = {
    "IN FAVOR": "in favor",
    "AGAINST": "ablehnung",
    "NEUTRAL": "neutral",
}


def normalize_stance_label(raw):
    """Map the LLM's stance vocabulary to the gold-label convention.

    The model is prompted to answer with "IN FAVOR" | "AGAINST" | "NEUTRAL",
    but gold_stance in gold_labels_corrected.json uses "in favor" | "ablehnung"
    | "neutral". Note "AGAINST" maps to "ablehnung" (German for "rejection") --
    this is a deliberate translation in the gold vocabulary, not a typo or a
    simple lowercasing. Matching is case-insensitive and whitespace-trimmed since LLM
    output casing/spacing isn't perfectly deterministic. Anything that doesn't
    match one of the three known categories is returned unchanged (not coerced
    or guessed), so anomalies stay visible instead of being silently masked.

    Args:
        raw: The stance value as returned by the model. May be any type;
            non-string values are returned unchanged.

    Returns:
        The canonical gold-vocabulary label if recognized, otherwise the
        original value unchanged.
    """
    if not isinstance(raw, str):
        return raw
    return STANCE_LABEL_MAP.get(raw.strip().upper(), raw)


class Stance(BaseModel):
    """Schema for a single stance-detection result returned by the model."""

    stance: str
    explanation: str

class KeyMessage(BaseModel):
    """Schema for a key-message extraction result returned by the model."""

    key_message: str
    key_message_type: str
    key_message_explanation: str

class LLMHandler:
    """Wraps calls to an OpenAI-compatible (vLLM) chat completions endpoint."""

    def __init__(self):
        """Initialize the handler with a default model and a single-slot semaphore."""
        self.model_name = "Qwen/Qwen3.5-27B-FP8"
        self.semaphore = asyncio.Semaphore(1)
        pass

    async def call_model(
            self,
    client: httpx.AsyncClient,
    base_url: str,
    model_name: str,
    system: str,
    messages: list[dict],
    semaphore: asyncio.Semaphore,
    enable_thinking: bool = False,
    api_key: str = "not-needed",
    response_schema: type[BaseModel] | None = None,
    ) -> dict | None:
        """Call a vLLM/OpenAI-compatible chat completions endpoint.

        Args:
            client: Async HTTP client used to issue the request.
            base_url: Endpoint base URL; a trailing slash is tolerated.
            model_name: Model identifier passed to the server.
            system: System prompt prepended to the message list.
            messages: Conversation messages in OpenAI chat format.
            semaphore: Concurrency guard; the call holds it for its duration.
            enable_thinking: If True, request the model's thinking mode and
                strip any ``<think>`` block from the returned content.
            api_key: Bearer token; defaults to a placeholder for local servers.
            response_schema: Optional Pydantic model whose JSON schema constrains
                the model's structured output. Defaults to ``Stance`` when None,
                preserving backward compatibility with existing callers.

        Returns:
            The parsed JSON object from the model's reply, or None if the
            request fails or the reply cannot be parsed as JSON.
        """
        async with semaphore:
            try:
                # Build request body
                full_messages = [{"role": "system", "content": system}] + messages

                body = {
                    "model": model_name,
                    "messages": full_messages,
                    "max_tokens": MAX_TOKENS,
                    "temperature": 0.1,  # Low temperature for consistency
                }

                # `response_schema or Stance` (instead of `if response_schema is
                # None`) intentionally also falls back to Stance for any falsy
                # value. In practice callers pass either a BaseModel subclass or
                # None, so this keeps the default behaviour for legacy callers
                # while staying a single concise expression.
                schema_model = response_schema or Stance
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_model.__name__,
                        "schema": schema_model.model_json_schema()
                    }
                }
                # Must stay a native dict: httpx's json= kwarg serializes the
                # whole body once, so a pre-encoded JSON string here would be
                # double-encoded and rejected with HTTP 400.
                body["response_format"] = response_format

                # Enable thinking for models that support it (e.g. Qwen3.5)
                if enable_thinking:
                    body["extra_body"] = {
                        "chat_template_kwargs": {"enable_thinking": True},
                        #"thinking": {"type": "enabled", "budget_tokens": THINKING_BUDGET},
                    }

                url = f"{base_url.rstrip('/')}/chat/completions"
                response = await client.post(
                    url,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    },
                    json=body,
                    timeout=120.0,
                )
                response.raise_for_status()

                data = response.json()
                content = data["choices"][0]["message"]["content"]
                # print(content)

                # If thinking mode, the actual response may be after </think> tags
                if enable_thinking and "<think>" in content:
                    # Extract content after thinking
                    parts = content.split("</think>")
                    if len(parts) > 1:
                        content = parts[-1].strip()

                # Parse JSON from response
                # Strip markdown fences if present
                clean = content.strip()
                clean = (
                    clean.removeprefix("```json")
                    .removeprefix("```")
                    .removesuffix("```")
                    .strip()
                )

                # Try to find JSON object in the response
                start = clean.find("{")
                end = clean.rfind("}") + 1
                if start >= 0 and end > start:
                    clean = clean[start:end]

                parsed = json.loads(clean)
                # Only "stance" is remapped to the gold vocabulary; "explanation"
                # is free-form natural language and must be returned verbatim.
                if isinstance(parsed, dict) and "stance" in parsed:
                    parsed["stance"] = normalize_stance_label(parsed["stance"])
                return parsed

            except httpx.HTTPStatusError as e:
                print(f"  HTTP error: {e.response.status_code} - {e.response.text[:200]}")
                return None
            except json.JSONDecodeError as e:
                print(f"  JSON parse error: {e}")
                print(f"  Raw response: {content[:300]}")
                return None
            except Exception as e:
                print(f"  Unexpected error: {e}")
                return None
            finally:
                await asyncio.sleep(RATE_LIMIT_DELAY)
