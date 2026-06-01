import time
import google.generativeai as genai
from typing import Dict, Any, Optional, Generator
from src.core.llm_provider import LLMProvider
from src.core.key_manager import KeyManager


class GeminiProvider(LLMProvider):
    def __init__(self, model_name: str = "gemini-1.5-flash", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        self.key_manager = KeyManager("GEMINI")
        self._active_key = self.key_manager.current() or api_key
        genai.configure(api_key=self._active_key)
        self.model = genai.GenerativeModel(model_name)
        self.max_retries = max(self.key_manager.count, 1)

    def _rebuild_model(self):
        """Switch to the next key in the pool."""
        next_key = self.key_manager.rotate()
        if next_key:
            self._active_key = next_key
            genai.configure(api_key=self._active_key)
            self.model = genai.GenerativeModel(self.model_name)

    def _build_prompt(self, prompt: str, system_prompt: Optional[str]) -> str:
        if system_prompt:
            return f"System: {system_prompt}\n\nUser: {prompt}"
        return prompt

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        full_prompt = self._build_prompt(prompt, system_prompt)
        start_time = time.time()

        last_error = None
        response = None
        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(full_prompt)
                break
            except Exception as exc:
                # Gemini raises ResourceExhausted on quota/rate limits.
                last_error = exc
                self._rebuild_model()
                time.sleep(1)
        if response is None:
            raise RuntimeError(f"All Gemini keys exhausted. Last error: {last_error}")

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        content = response.text
        usage = {
            "prompt_tokens": response.usage_metadata.prompt_token_count,
            "completion_tokens": response.usage_metadata.candidates_token_count,
            "total_tokens": response.usage_metadata.total_token_count,
        }

        return {
            "content": content,
            "usage": usage,
            "latency_ms": latency_ms,
            "provider": "google",
        }

    def stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        full_prompt = self._build_prompt(prompt, system_prompt)
        response = self.model.generate_content(full_prompt, stream=True)
        for chunk in response:
            yield chunk.text
