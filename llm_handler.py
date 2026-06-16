"""Handles the communication with the llm"""
from pydantic import BaseModel
import httpx
import json
import asyncio

# ── Configuration ──────────────────────────────────────────────────

MAX_TOKENS = 8000
THINKING_BUDGET = 6000
RATE_LIMIT_DELAY = 0.5

class Stance(BaseModel):
    """Schema for a single stance-detection result returned by the model."""

    stance: str
    explanation: str

class LLMHandler:
    """Wraps calls to an OpenAI-compatible (vLLM) chat completions endpoint."""

    def __init__(self):
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

                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "stance-detection",
                        "schema": Stance.model_json_schema()
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
                        "thinking": {"type": "enabled", "budget_tokens": THINKING_BUDGET},
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

                return json.loads(clean)

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
