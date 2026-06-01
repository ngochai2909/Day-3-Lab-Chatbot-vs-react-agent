"""
ChatGPT-style Streamlit UI for Lab 3: Chatbot vs ReAct Agent (Layout A: clean single column).

Run it with:
    streamlit run app.py

- Sidebar: provider + model (dropdowns), mode (BUTTONS), max steps, samples.
- Single-column chat. For each question:
    * Chatbot answer (if mode includes it)
    * Agent: Final Answer shown, reasoning steps folded inside an expander.
- Small metrics row (tokens / latency) under each answer.
"""
import os
import time
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

from src.core.factory import create_provider
from src.core.openai_provider import OpenAIProvider
from src.core.gemini_provider import GeminiProvider
from src.chatbot import Chatbot
from src.agent.agent import ReActAgent
from src.tools import TOOL_REGISTRY


st.set_page_config(page_title="Chatbot vs ReAct Agent", page_icon="🤖", layout="centered")

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.5rem; max-width: 50rem; }
      [data-testid="stChatMessage"] { padding: 0.2rem 0.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Provider builder
# --------------------------------------------------------------------------
PROVIDER_MODELS = {
    "router": ["cx/gpt-5.5", "cx/gpt-5.4", "cx/gpt-5.4-mini", "Claude_Opus_4.6",
               "kr/claude-opus-4.7", "ag/gemini-3-flash", "kr/deepseek-3.2", "kr/glm-5"],
    "mimo": ["mimo-v2.5-pro"],
    "google": ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
    "openai": ["gpt-4o", "gpt-4o-mini"],
    "local": ["phi-3"],
}

MODES = ["So sánh", "Chỉ Chatbot", "Chỉ Agent"]


def build_provider(provider_choice: str, model_name: str):
    provider_choice = provider_choice.lower()
    if provider_choice in ("router", "9router"):
        from src.core.router_provider import RouterProvider
        return RouterProvider(model_name=model_name or "cx/gpt-5.5")
    if provider_choice == "mimo":
        from src.core.mimo_provider import MiMoProvider
        return MiMoProvider(model_name=model_name or "mimo-v2.5-pro")
    if provider_choice in ("google", "gemini"):
        return GeminiProvider(model_name=model_name or "gemini-2.0-flash")
    if provider_choice == "openai":
        return OpenAIProvider(model_name=model_name or "gpt-4o")
    if provider_choice == "local":
        from src.core.local_provider import LocalProvider
        path = os.getenv("LOCAL_MODEL_PATH", "./models/Phi-3-mini-4k-instruct-q4.gguf")
        return LocalProvider(model_path=path)
    return create_provider()


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None
if "mode" not in st.session_state:
    st.session_state.mode = MODES[0]


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Cấu hình")

    default_provider = os.getenv("DEFAULT_PROVIDER", "router")
    provider_options = ["router", "mimo", "google", "openai", "local"]
    provider_choice = st.selectbox(
        "Provider",
        provider_options,
        index=provider_options.index(default_provider) if default_provider in provider_options else 0,
    )

    model_options = PROVIDER_MODELS.get(provider_choice, [])
    if model_options:
        model_name = st.selectbox("Model", model_options, index=0)
    else:
        model_name = st.text_input("Model", value="")

    # ---- Mode as BUTTONS (active one is highlighted) ----
    st.markdown("**Chế độ**")
    mcols = st.columns(3)
    mode_labels = {"So sánh": "⚖️ So sánh", "Chỉ Chatbot": "💬 Chatbot", "Chỉ Agent": "🧠 Agent"}
    for i, m in enumerate(MODES):
        is_active = st.session_state.mode == m
        if mcols[i].button(
            mode_labels[m],
            key=f"mode_{m}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.mode = m
            st.rerun()

    max_steps = st.slider("Max steps (agent)", min_value=3, max_value=12, value=8)

    if st.button("🆕 Cuộc trò chuyện mới", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending = None
        st.rerun()

    st.markdown("---")
    st.subheader("💡 Câu hỏi mẫu")
    SAMPLE_GROUPS = {
        "🟢 Đơn giản": [
            "Giá của macbook là bao nhiêu?",
            "Mã giảm giá BLACKFRIDAY giảm bao nhiêu %?",
            "Sản phẩm pixel còn hàng không?",
        ],
        "🟡 Nhiều bước": [
            "Mua 2 iphone và áp mã WINNER (giảm 10%). Tổng tiền?",
            "Mua 1 laptop và 1 headphones, áp mã SALE20, ship tới hanoi. Tổng cuối cùng?",
            "Mua 3 smartwatch với mã VIP50 và ship tới danang. Tôi phải trả bao nhiêu?",
        ],
        "🔴 Bẫy": [
            "Tôi muốn mua 1 con mouse. Còn hàng không và giá bao nhiêu?",
            "Mua 1 iphone với mã FAKE123 ship tới sao hỏa. Tổng tiền?",
        ],
    }
    for group_name, questions in SAMPLE_GROUPS.items():
        with st.expander(group_name):
            for q in questions:
                if st.button(q, key=f"sample_{q}", use_container_width=True):
                    st.session_state.pending = q
                    st.rerun()

    st.markdown("---")
    st.caption("🛠️ " + " · ".join(f"`{t['name']}`" for t in TOOL_REGISTRY))


# --------------------------------------------------------------------------
# Core runners
# --------------------------------------------------------------------------
def run_chatbot_block(provider, q):
    start = time.time()
    answer = Chatbot(provider).run(q)
    return answer, int((time.time() - start) * 1000)


def run_agent_block(provider, q, max_steps):
    """Run agent fully, collecting steps. Returns dict."""
    agent = ReActAgent(provider, TOOL_REGISTRY, max_steps=max_steps)
    steps = []
    tokens = 0
    latency = 0
    final = None
    for ev in agent.run_iter(q):
        tokens += ev.get("usage", {}).get("total_tokens", 0)
        latency += ev.get("latency_ms", 0)
        if ev["type"] == "thought_action":
            steps.append({"kind": "step", "n": ev["step"], "thought": ev.get("thought") or "",
                          "tool": ev["tool"], "arg": ev["arg"], "obs": None})
        elif ev["type"] == "tool_call":
            # attach observation to the last step
            if steps and steps[-1]["obs"] is None:
                steps[-1]["obs"] = ev["observation"]
            else:
                steps.append({"kind": "obs", "n": ev["step"], "obs": ev["observation"]})
        elif ev["type"] == "parser_error":
            steps.append({"kind": "error", "n": ev["step"]})
        elif ev["type"] == "final":
            final = ev["answer"]
        elif ev["type"] == "timeout":
            steps.append({"kind": "timeout", "n": ev["step"]})
    return {"final": final, "tokens": tokens, "latency": latency, "steps": steps}


# --------------------------------------------------------------------------
# Renderers
# --------------------------------------------------------------------------
def render_chatbot(msg):
    with st.chat_message("assistant", avatar="💬"):
        st.markdown("**💬 Chatbot** (1 lần gọi, không tool)")
        st.markdown(msg["answer"])
        st.caption(f"⏱️ {msg['latency']} ms")


def render_agent(msg, max_steps):
    with st.chat_message("assistant", avatar="🧠"):
        st.markdown("**🧠 ReAct Agent**")
        steps = msg["steps"]
        n_tools = sum(1 for s in steps if s["kind"] == "step")
        with st.expander(f"🔧 Xem {n_tools} bước suy luận (Thought → Action → Observation)"):
            for s in steps:
                if s["kind"] == "step":
                    if s["thought"]:
                        st.markdown(f"**Bước {s['n']} · Thought:** {s['thought']}")
                    st.markdown(f"➡️ **Action:** `{s['tool']}({s['arg']})`")
                    if s["obs"] is not None:
                        st.code(s["obs"], language="text")
                    st.divider()
                elif s["kind"] == "obs":
                    st.code(s["obs"], language="text")
                elif s["kind"] == "error":
                    st.warning(f"⚠️ Bước {s['n']}: PARSER_ERROR (LLM sai format)")
                elif s["kind"] == "timeout":
                    st.error(f"⛔ Bước {s['n']}: vượt max_steps")
        if msg["final"] is not None:
            st.success(f"✅ {msg['final']}")
        else:
            st.error("Không có Final Answer.")
        st.caption(f"🔢 {msg['tokens']} tokens · ⏱️ {msg['latency']} ms · 🔁 {n_tools} bước")


def render_turn(turn, max_steps):
    with st.chat_message("user", avatar="🧑"):
        st.markdown(turn["question"])
    if "chatbot" in turn:
        render_chatbot(turn["chatbot"])
    if "agent" in turn:
        render_agent(turn["agent"], max_steps)


# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------
st.title("🤖 Chatbot vs ReAct Agent")
st.caption(
    f"Lab 3 · Provider: **{provider_choice}** · Model: **{model_name}** · Chế độ: **{st.session_state.mode}**"
)

# Render history
for turn in st.session_state.messages:
    render_turn(turn, max_steps)

# Input
typed = st.chat_input("Nhập câu hỏi của bạn...")
question = typed or st.session_state.pending
st.session_state.pending = None

if question:
    mode = st.session_state.mode
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    try:
        provider = build_provider(provider_choice, model_name)
    except Exception as exc:
        st.error(f"Không khởi tạo được provider '{provider_choice}': {exc}")
        st.stop()

    turn = {"question": question}

    if mode in ("So sánh", "Chỉ Chatbot"):
        with st.spinner("Chatbot đang trả lời..."):
            answer, latency = run_chatbot_block(provider, question)
        turn["chatbot"] = {"answer": answer, "latency": latency}
        render_chatbot(turn["chatbot"])

    if mode in ("So sánh", "Chỉ Agent"):
        with st.spinner("Agent đang suy luận..."):
            agent_res = run_agent_block(provider, question, max_steps)
        turn["agent"] = agent_res
        render_agent(turn["agent"], max_steps)

    st.session_state.messages.append(turn)
