from __future__ import annotations

import asyncio
from pathlib import Path
from string import Template

import httpx

from llm_handler import LLMHandler, KeyMessageList
from stance_detector import StanceDetector

DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


class KeyMessageExtraction:
    """Extract a structured key message from a citizen participation contribution."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        prompts_dir: Path = DEFAULT_PROMPTS_DIR,
    ) -> None:
        """Initialize and load prompt config once.

        Args:
            base_url: vLLM/OpenAI-compatible endpoint base URL.
            model_name: Model identifier passed to the server.
            prompts_dir: Directory containing config.json and prompt text files.
                Defaults to the prompts/ subdirectory next to this module.
        """
        self.base_url = base_url
        self.model_name = model_name
        self._llm_handler = LLMHandler()
        config_path = str(Path(prompts_dir) / "config.json")
        # Load prompt files once at init to avoid repeated file I/O on every extract() call.
        self._prompt_config = StanceDetector().load_prompt_config(config_path, "key_message")

    async def extract(
        self,
        client: httpx.AsyncClient,
        contribution: str,
        proceeding_description: str,
        semaphore: asyncio.Semaphore,
        enable_thinking: bool = False,
    ) -> list[dict] | None:
        """Extract the key messages from a contribution.

        Args:
            client: Async HTTP client for the LLM request.
            contribution: The citizen contribution text.
            proceeding_description: Description of the planning proceeding for context.
            semaphore: Concurrency guard shared across concurrent extract calls.
            enable_thinking: If True, request the model's thinking mode.

        Returns:
            A list of key-message dicts, each with keys ``key_message``,
            ``key_message_type``, ``related_sentences``, and ``explanation``; or
            ``None`` if the LLM call fails or the response cannot be parsed.
            Returned strings are verbatim model output — do not treat as safe HTML/SQL.
        """
        # safe_substitute (not substitute) is used so that any stray "$" in the
        # prompt template or in the substituted values doesn't raise KeyError /
        # ValueError. Unknown placeholders are left intact rather than crashing.
        user_content = Template(self._prompt_config["user_template"]).safe_substitute(
            contribution=contribution,
            proceeding_description=proceeding_description,
        )
        system = self._prompt_config["system"]
        messages = [{"role": "user", "content": user_content}]

        raw = await self._llm_handler.call_model(
            client=client,
            base_url=self.base_url,
            model_name=self.model_name,
            system=system,
            messages=messages,
            semaphore=semaphore,
            enable_thinking=enable_thinking,
            response_schema=KeyMessageList,
        )
        if raw is None:
            return None
        return raw.get("key_messages")
