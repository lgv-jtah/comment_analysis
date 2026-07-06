"""LLM-as-a-judge evaluation of already-extracted key messages.

Given a citizen contribution and the key messages previously extracted from it,
:class:`KeyMessageJudge` asks the LLM to score their completeness, semantic
precision, and downstream stance-ability, returning a structured judgment.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from string import Template

import httpx

from llm_handler import LLMHandler, KeyMessageJudgment
from stance_detector import StanceDetector

DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


class KeyMessageJudge:
    """Judge the quality of key messages extracted from a citizen contribution."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        prompts_dir: Path = DEFAULT_PROMPTS_DIR,
        api_key: str = "",
    ) -> None:
        """Initialize and load the judge prompt config once.

        Args:
            base_url: vLLM/OpenAI-compatible endpoint base URL.
            model_name: Model identifier passed to the server.
            prompts_dir: Directory containing config.json and prompt text files.
                Defaults to the prompts/ subdirectory next to this module.
            api_key: API key forwarded to every LLM call. Defaults to an empty
                string so the judge can be constructed without a secret (e.g. in
                tests); callers thread the real key in from the environment.
        """
        self.base_url = base_url
        self.model_name = model_name
        self._api_key = api_key
        self._llm_handler = LLMHandler()
        config_path = str(Path(prompts_dir) / "config.json")
        # Load prompt files once at init to avoid repeated file I/O per judge() call.
        self._prompt_config = StanceDetector().load_prompt_config(
            config_path, "key_message_judge"
        )

    @staticmethod
    def format_key_messages(key_messages: list[dict]) -> str:
        """Render key messages as a numbered, 0-indexed list for the prompt.

        Args:
            key_messages: The key-message dicts extracted from one contribution.

        Returns:
            One line per key message in the form ``[i] (type) text``, joined by
            newlines. Missing ``key_message`` / ``key_message_type`` fields render
            as empty strings rather than raising.
        """
        lines = []
        for i, msg in enumerate(key_messages):
            msg_type = msg.get("key_message_type", "")
            text = msg.get("key_message", "")
            lines.append(f"[{i}] ({msg_type}) {text}")
        return "\n".join(lines)

    async def judge(
        self,
        client: httpx.AsyncClient,
        contribution: str,
        category: str,
        key_messages: list[dict],
        semaphore: asyncio.Semaphore,
        enable_thinking: bool = False,
    ) -> dict | None:
        """Judge the quality of one contribution's key messages.

        Args:
            client: Async HTTP client for the LLM request.
            contribution: The citizen contribution text.
            category: Category/context of the contribution; may be empty.
            key_messages: The key-message dicts to evaluate.
            semaphore: Concurrency guard shared across concurrent judge calls.
            enable_thinking: If True, request the model's thinking mode.

        Returns:
            The parsed judgment dict matching :class:`KeyMessageJudgment` (keys
            ``completeness``, ``semantic_precision``, ``downstream_viability``,
            ``overall_score``, ``overall_assessment``), or ``None`` if the LLM
            call fails or the response cannot be parsed.
        """
        key_messages_formatted = self.format_key_messages(key_messages)
        # safe_substitute leaves any stray "$" in the template or values intact
        # rather than raising KeyError / ValueError.
        user_content = Template(self._prompt_config["user_template"]).safe_substitute(
            contribution=contribution,
            category=category,
            key_messages_formatted=key_messages_formatted,
        )
        messages = [{"role": "user", "content": user_content}]

        raw = await self._llm_handler.call_model(
            client=client,
            base_url=self.base_url,
            model_name=self.model_name,
            system=self._prompt_config["system"],
            messages=messages,
            semaphore=semaphore,
            enable_thinking=enable_thinking,
            api_key=self._api_key,
            response_schema=KeyMessageJudgment,
        )
        if raw is None:
            return None
        return raw
