"""Handles the communication with the llm"""
from pydantic import BaseModel
import httpx
import json
import asyncio

# ── Configuration ──────────────────────────────────────────────────

MAX_TOKENS = 20000
THINKING_BUDGET = 8000
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
    """Schema for a single key-message extraction result returned by the model.

    Attributes:
        key_message: The distilled key message extracted from the contribution.
        key_message_type: The category/type of the key message (e.g. a wish or
            proposal).
        related_sentences: The verbatim source sentences the key message was
            derived from.
        explanation: Free-form natural-language justification for the extraction.
    """

    key_message: str
    key_message_type: str
    related_sentences: list[str]
    explanation: str

class KeyMessageList(BaseModel):
    """Schema wrapping the list of key messages extracted from one contribution.

    A single contribution can yield several key messages, so the model returns a
    list rather than a single object. This wrapper exists because the structured-
    output JSON schema must have an object at its root, not a bare array.

    Attributes:
        key_messages: All key messages extracted from one contribution.
    """

    key_messages: list[KeyMessage]

class JudgeIssue(BaseModel):
    """Schema for one issue the judge raises against a specific key message.

    Attributes:
        key_message_index: 0-based index of the offending key message, matching
            the numbering produced by ``KeyMessageJudge.format_key_messages``.
        problem: Description of what is wrong with that key message.
        suggested_improvement: The judge's proposed fix for the problem.
    """

    key_message_index: int
    problem: str
    suggested_improvement: str

class CompletenessScore(BaseModel):
    """Schema for the completeness dimension of a key-message judgment.

    Completeness scores the whole set of key messages together (rather than any
    single one), so it carries missing topics instead of per-message issues.

    Attributes:
        score: The judge's completeness rating for the key-message set.
        justification: Free-form rationale for the score.
        missing_topics: Topics present in the contribution but not captured by
            any key message.
    """

    score: int
    justification: str
    missing_topics: list[str]

class IssueDimensionScore(BaseModel):
    """Schema for a per-key-message dimension (precision or viability) score.

    Used for the dimensions that fault individual key messages — semantic
    precision and downstream viability — hence the list of per-message issues.

    Attributes:
        score: The judge's rating for this dimension.
        justification: Free-form rationale for the score.
        issues: Per-key-message problems found for this dimension; empty when
            the judge raises none.
    """

    score: int
    justification: str
    issues: list[JudgeIssue]

class KeyMessageJudgment(BaseModel):
    """Schema for the full quality judgment of a contribution's key messages.

    This is the top-level structured-output schema passed to the LLM by
    ``KeyMessageJudge.judge`` and the shape returned to callers.

    Attributes:
        completeness: Whether the key messages cover the contribution's content.
        semantic_precision: How accurately each key message captures its meaning.
        downstream_viability: How usable the key messages are for downstream
            stance detection.
        overall_score: The judge's single aggregate score across all dimensions.
        overall_assessment: Free-form summary of the overall judgment.
    """

    completeness: CompletenessScore
    semantic_precision: IssueDimensionScore
    downstream_viability: IssueDimensionScore
    overall_score: int
    overall_assessment: str

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
                    # Structured split timeout (replaces the former flat 120.0s):
                    # a short connect (10s) fails fast on an unreachable server,
                    # while a long read (600s) tolerates slow LLM generation; write
                    # (30s) covers uploading the request body and pool (10s) bounds
                    # the wait for a free connection from the pool.
                    timeout=httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=10.0),
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
