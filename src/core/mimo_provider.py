import os
import time
from typing import Dict, Any, Optional, Generator
from openai import OpenAI, RateLimitError, APIError
from src.core.llm_provider import LLMProvider
from src.core.key_manager import KeyManager

# Default OpenAI-compatible endpoint for Xiaomi MiMo (token-plan SGP).
DEFAULT_MIMO_BASE_URL = "https://token-plan-sgp.xiaomimimo.com/v1"


class MiMoProvider(LLMProvider):
    """
    Provider for Xiaomi MiMo models via its OpenAI-compatible endpoint.

    Reuses the OpenAI SDK by pointing base_url at the MiMo gateway.
    Supports rotating across multiple MIMO_API_KEYS to avoid rate limits.
    """

    def __init__(self, model_name: str = "mimo-v2.5-pro", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        self.base_url = os.getenv("MIMO_BASE_URL", DEFAULT_MIMO_BASE_URL)
        self.key_manager = KeyManager("MIMO")
        active_key = self.key_manager.current() or api_key
        self.client = OpenAI(api_key=active_key, base_url=self.base_url)
        self.max_retries = max(self.key_manager.count, 1)

    def _rebuild_client(self):
        """Switch the client to the next key in the pool."""
        next_key = self.key_manager.rotate()
        if next_key:
            self.client = OpenAI(api_key=next_key, base_url=self.base_url)

    def _build_messages(self, prompt: str, system_prompt: Optional[str]):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        messages = self._build_messages(prompt, system_prompt)
        start_time = time.time()

        last_error = None
        response = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                )
                break
            except (RateLimitError, APIError) as exc:
                last_error = exc
                self._rebuild_client()
                time.sleep(1)
        if response is None:
            raise RuntimeError(f"All MiMo API keys exhausted. Last error: {last_error}")

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        content = response.choices[0].message.content
        usage = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
            "completion_tokens": getattr(response.usage, "completion_tokens", 0),
            "total_tokens": getattr(response.usage, "total_tokens", 0),
        }

        return {
            "content": content,
            "usage": usage,
            "latency_ms": latency_ms,
            "provider": "mimo",
        }

    def stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        messages = self._build_messages(prompt, system_prompt)
        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
