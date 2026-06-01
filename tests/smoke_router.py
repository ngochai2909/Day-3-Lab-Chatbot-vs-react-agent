"""Smoke test: run chatbot + agent through the factory using the router provider."""
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from src.core.factory import create_provider
from src.chatbot import Chatbot
from src.agent.agent import ReActAgent
from src.tools import TOOL_REGISTRY


def main():
    provider = create_provider()
    print("PROVIDER:", type(provider).__name__, "| MODEL:", provider.model_name)

    q = "Buy 2 iphones and apply coupon WINNER (10% off). Total?"
    print("\nQUESTION:", q)

    print("\n--- CHATBOT ---")
    bot = Chatbot(provider)
    print(bot.run(q))

    print("\n--- AGENT (live steps) ---")
    agent = ReActAgent(provider, TOOL_REGISTRY, max_steps=8, verbose=True)
    answer = agent.run(q)
    print("FINAL:", answer)


if __name__ == "__main__":
    main()
