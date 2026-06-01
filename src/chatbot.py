from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker


class Chatbot:
    """
    Baseline chatbot: a single LLM call with NO tools and NO reasoning loop.

    This is intentionally "dumb" so we can compare it against the ReAct agent.
    It will tend to hallucinate or miscalculate on multi-step questions because
    it has no way to look up real prices or run real calculations.
    """

    SYSTEM_PROMPT = (
        "You are a helpful shopping assistant. "
        "Answer the user's question directly and concisely."
    )

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def run(self, user_input: str) -> str:
        logger.log_event("CHATBOT_START", {"input": user_input, "model": self.llm.model_name})

        result = self.llm.generate(user_input, system_prompt=self.SYSTEM_PROMPT)

        tracker.track_request(
            provider=result.get("provider", "unknown"),
            model=self.llm.model_name,
            usage=result.get("usage", {}),
            latency_ms=result.get("latency_ms", 0),
        )

        answer = result["content"]
        logger.log_event("CHATBOT_END", {"answer": answer})
        return answer
