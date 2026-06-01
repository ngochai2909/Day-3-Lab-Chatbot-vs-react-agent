"""
Lab 3 runner: compare the Chatbot baseline vs the ReAct Agent.

Usage:
    python main.py                # run both on all test cases
    python main.py chatbot        # run only the chatbot
    python main.py agent          # run only the agent
    python main.py "your question"  # run both on a single custom question
"""
import sys
from dotenv import load_dotenv

from src.core.factory import create_provider
from src.chatbot import Chatbot
from src.agent.agent import ReActAgent
from src.tools import TOOL_REGISTRY


# Test cases with known correct answers (ground truth) for grading.
TEST_CASES = [
    {"q": "What is the price of an iphone?", "expected": "$999"},
    {"q": "How much discount does coupon WINNER give?", "expected": "10%"},
    {"q": "I want to buy 2 iphones. What is the total price?", "expected": "$1998"},
    {"q": "Buy 2 iphones and apply coupon WINNER (10% off). Total?", "expected": "$1798.2"},
    {
        "q": "Buy 1 laptop and 1 headphones, apply coupon SALE20, ship to hanoi. Final total?",
        "expected": "$1085",
    },
]


def run_chatbot(provider, question):
    bot = Chatbot(provider)
    print("\n[CHATBOT]")
    answer = bot.run(question)
    print(f"  Q: {question}")
    print(f"  A: {answer}")
    return answer


def run_agent(provider, question):
    agent = ReActAgent(provider, TOOL_REGISTRY, max_steps=8)
    print("\n[REACT AGENT]")
    answer = agent.run(question)
    print(f"  Q: {question}")
    print(f"  A: {answer}")
    return answer


def main():
    load_dotenv()
    provider = create_provider()

    arg = sys.argv[1] if len(sys.argv) > 1 else "both"

    # Custom single question (anything that isn't a known mode keyword).
    if arg not in ("both", "chatbot", "agent"):
        question = arg
        run_chatbot(provider, question)
        run_agent(provider, question)
        return

    for case in TEST_CASES:
        q = case["q"]
        print("\n" + "=" * 70)
        print(f"TEST: {q}")
        print(f"EXPECTED: {case['expected']}")
        if arg in ("both", "chatbot"):
            run_chatbot(provider, q)
        if arg in ("both", "agent"):
            run_agent(provider, q)


if __name__ == "__main__":
    main()
