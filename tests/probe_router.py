"""Quick probe for the local 9router gateway with a given key + model."""
import sys
import time
from openai import OpenAI

API_KEY = "sk-3c6c3c4b59d8d9e4-24v3zk-02878526"
BASE_URL = "http://localhost:20128/v1"
MODEL = "cx/gpt-5.5"


def main():
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=60.0)

    # 1) List models (best-effort).
    try:
        models = client.models.list()
        names = [m.id for m in models.data]
        print(f"MODELS ({len(names)}):")
        for n in names:
            print("  -", n)
    except Exception as exc:
        print(f"models.list failed: {type(exc).__name__}: {str(exc)[:200]}")

    # 2) Chat completion.
    print(f"\nCalling chat.completions with model={MODEL} ...")
    start = time.time()
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Viet cho toi ham Python tinh tong 2 so"}],
            stream=False,
        )
        elapsed = int((time.time() - start) * 1000)
        print("REPLY:\n", resp.choices[0].message.content)
        if resp.usage:
            print("USAGE:", resp.usage)
        print("LATENCY_ms:", elapsed)
    except Exception as exc:
        print(f"chat FAILED: {type(exc).__name__}: {str(exc)[:300]}")


if __name__ == "__main__":
    main()
