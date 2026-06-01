import time
from typing import Dict, Any, Optional, Generator
from openai import OpenAI, RateLimitError, APIError
from src.core.llm_provider import LLMProvider
from src.core.key_manager import KeyManager


class OpenAIProvider(LLMProvider):
    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        # Pool of keys for rotation. Falls back to the single api_key if no pool.
        self.key_manager = KeyManager("OPENAI")
        active_key = self.key_manager.current() or api_key
        self.client = OpenAI(api_key=active_key)
        self.max_retries = max(self.key_manager.count, 1)

    def _rebuild_client(self):
        """Switch the client to the next key in the pool."""
        next_key = self.key_manager.rotate()
        if next_key:
            self.client = OpenAI(api_key=next_key)

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
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                )
                break
            except (RateLimitError, APIError) as exc:
                last_error = exc
                # Rotate to the next key and retry
                self._rebuild_client()
                time.sleep(1)
        else:
            raise RuntimeError(f"All API keys exhausted. Last error: {last_error}")

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        content = response.choices[0].message.content
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        return {
            "content": content,
            "usage": usage,
            "latency_ms": latency_ms,
            "provider": "openai",
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
