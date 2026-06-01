# Group Report: Lab 3 - Production-Grade Agentic System

- **Team Name**: 16383
- **Team Members**:
  | # | Họ tên | MSSV | Vai trò chính |
  | :- | :--- | :--- | :--- |
  | 1 | Nguyễn Trường Giang | 2A202600624 | Agent Core (vòng lặp ReAct, parser) |
  | 2 | Phạm Văn Lợi | 2A202600784 | Chatbot baseline & Đánh giá so sánh |
  | 3 | Lê Xuân Tiến Đạt | 2A202600549 | Tools & Provider (dataset, MiMo/9router) |
  | 4 | Dương Quang Huy | 2A202600839 | Cải tiến Prompt v1 → v2 |
  | 5 | Nguyễn Ngọc Hải | 2A202600614 | Telemetry, UI (Streamlit), thu trace |
- **Deployment Date**: 2026-06-01

---

## 1. Executive Summary

Nhóm xây dựng một **Trợ lý mua sắm E-commerce** ở 2 phiên bản để so sánh: một **Chatbot baseline** (gọi LLM 1 lần, không tool) và một **ReAct Agent** (vòng lặp Thought–Action–Observation, gọi tool thật). Hệ thống chạy được trên nhiều provider (9router/`cx/gpt-5.5`, MiMo `mimo-v2.5-pro`, Gemini) và ghi telemetry JSON cho mọi lượt gọi.

- **Bộ test**: 5 câu hỏi (2 câu đơn giản 1 bước + 3 câu nhiều bước), có đáp án đúng tính tay để chấm.
- **Success Rate** (đo trên `mimo-v2.5-pro`):
  - Chatbot baseline: **0/5 (0%)** — không câu nào trả lời đúng con số.
  - ReAct Agent v1: **4/5 (80%)** — đúng câu 1, 2, 3, 4; sai câu 5.
- **Key Outcome**: Agent giải đúng cả các câu nhiều bước nhờ gọi tool tra giá/giảm giá thật, trong khi Chatbot hoặc **bịa số** (câu 4: giả định iPhone $799 → tổng sai $1438.20) hoặc **từ chối/hỏi lại** (câu 1, 2, 3, 5) vì không có dữ liệu thật.

---

## 2. System Architecture & Tooling

### 2.1 ReAct Loop Implementation

LLM **không tự chạy tool** — nó chỉ sinh text mô tả hành động. Code Python (`src/agent/agent.py`) chịu trách nhiệm thực thi:

```
Question
   │
   ▼
┌─────────────────────────────────────────────┐
│  while step < max_steps:                      │
│    1. LLM.generate(scratchpad, system_prompt) │
│    2. Parse "Final Answer:"  → nếu có thì DỪNG │
│    3. Parse "Action: tool(arg)" bằng regex    │
│    4. _execute_tool() → chạy hàm Python thật  │
│    5. Nối "Observation: <kết quả>" vào         │
│       scratchpad rồi lặp lại                   │
└─────────────────────────────────────────────┘
   │
   ▼
Final Answer
```

- **Parser**: regex `Action:\s*([a-zA-Z_]\w*)\s*\((.*?)\)` để bóc tách tên tool + tham số; regex `Final Answer:\s*(.+)` để phát hiện điểm dừng.
- **Guardrail**: giới hạn `max_steps` (mặc định 8) chống lặp vô hạn; tool không tồn tại → trả về thông báo "Tool not found" (bắt hallucination).
- **Telemetry**: mỗi lượt gọi LLM ghi `LLM_METRIC` (token, latency, cost); mỗi tool ghi `TOOL_CALL`; lỗi parse ghi `AGENT_ERROR`.

### 2.2 Tool Definitions (Inventory)

| Tool Name | Input | Use Case |
| :--- | :--- | :--- |
| `calculator` | string biểu thức | Tính toán số học an toàn (chặn ký tự lạ) |
| `lookup_product_price` | tên sản phẩm | Tra giá USD trong catalog 16 sản phẩm |
| `get_discount` | mã coupon | Tra % giảm giá (8 mã) |
| `calc_shipping` | tên thành phố | Tra phí ship (11 thành phố VN + quốc tế) |
| `check_stock` | tên sản phẩm | Kiểm tra tồn kho (có hàng hết) |
| `get_product_weight` | tên sản phẩm | Tra cân nặng (kg) |

### 2.3 LLM Providers Used

Kiến trúc **Provider Pattern** (lớp abstract `LLMProvider`), đổi provider chỉ qua `.env` không sửa code. Có cơ chế **xoay vòng API key** (`KeyManager`) tự nhảy key khi gặp rate limit.

- **Primary**: `cx/gpt-5.5` qua 9router (gateway local OpenAI-compatible, `localhost:20128/v1`).
- **Secondary**: `mimo-v2.5-pro` (Xiaomi MiMo) — provider dùng để chạy bộ đánh giá đầy đủ.
- **Backup**: Gemini (`gemini-2.0-flash`) — dùng để demo đổi provider.

---

## 3. Telemetry & Performance Dashboard

Số liệu đo thực tế trên `mimo-v2.5-pro` (trích từ `logs/2026-06-01.log`).

### So sánh tổng quan

| Chỉ số | Chatbot | ReAct Agent v1 |
| :--- | :--- | :--- |
| Tổng token (5 câu) | 3,998 | 11,178 |
| Token trung bình / câu | ~800 | ~2,236 |
| Tổng latency (5 câu) | 116,154 ms | 157,704 ms |
| Latency trung bình / câu | ~23,231 ms | ~31,541 ms |
| Cost ước tính (5 câu) | ~$0.040 | ~$0.112 |
| Tỉ lệ đúng | 0/5 | 4/5 |

### Latency theo từng lượt gọi (minh họa quan hệ token ↔ latency)

| Lượt | completion_tokens | latency_ms |
| :--- | ---: | ---: |
| Agent câu 3 / bước 3 | 801 | 21,084 |
| Chatbot câu 5 | 2,424 | 64,188 |
| Agent câu 2 / bước 1 | 119 | 3,989 |

**Insight**: latency gần như tỉ lệ thuận với số token sinh ra. Lượt sinh 2,424 token mất 64s, lượt sinh 119 token chỉ 4s. Agent tốn token và latency cao hơn Chatbot vì **gọi LLM nhiều lượt** (mỗi câu nhiều bước), đây là cái giá phải trả để đổi lấy độ chính xác.

---

## 4. Root Cause Analysis (RCA) - Failure Traces

### Trace THÀNH CÔNG (câu 3): "I want to buy 2 iphones. What is the total price?"

```
Step 1  Action: lookup_product_price(iphone)  → Observation: iphone: $999
Step 2  Action: calculator(2 * 999)           → Observation: 1998
Step 3  Final Answer: The total price for 2 iPhones is $1998.   ✅ ĐÚNG
```

### Trace LỖI (câu 5): "Buy 1 laptop and 1 headphones, apply coupon SALE20, ship to hanoi. Final total?"
**Đáp án đúng: $1085** (1200 + 150 = 1350 → −20% = 1080 → + ship hanoi $5 = **1085**)
**Agent v1 trả về: $1090 → SAI**

Trace gặp **3 loại lỗi cùng lúc**:

1. **PARSER_ERROR (bước 1)**: LLM xuất `lookup_product_price(laptop)` **thiếu tiền tố `Action:`** → parser không bắt được.
2. **PARSER_ERROR (bước 4)**: LLM xuất `The final total is $1110.` **thiếu `Final Answer:`** → không parse được, đồng thời con số $1110 là tự bịa.
3. **HALLUCINATION (bước 6) — lỗi gây sai kết quả**: LLM **tự viết** `Observation: hanoi: $10` thay vì gọi tool `calc_shipping(hanoi)` (kết quả thật = **$5**). Nó cộng nhầm ship $10 → ra $1090 thay vì $1085.

- **Root Cause**: System prompt v1 chưa **bắt buộc** mọi giá trị (giá, giảm giá, phí ship) phải lấy từ tool và chưa cấm LLM tự viết Observation. Model `cx/gpt-5.5`/`mimo` đủ "thông minh" để tự tính nhẩm → bỏ qua tool → bịa số.

---

## 5. Ablation Studies & Experiments

### Experiment 1: Prompt v1 → v2

| | v1 | v2 (đã triển khai trong `agent.py`) |
| :--- | :--- | :--- |
| Quy tắc dùng tool | Khuyến khích | **Bắt buộc**: mọi phép tính qua `calculator`, mọi giá qua `lookup_product_price`, mọi % qua `get_discount`, mọi ship qua `calc_shipping` |
| Tự viết Observation | Không cấm rõ | **Cấm tuyệt đối** (rule 7) |
| Few-shot example | Không có | **Có** 1 ví dụ multi-step đầy đủ 8 bước |
| Mục tiêu sửa | — | Khắc phục hallucination ship $10 + ép format đúng |

- **Diff chính**: thêm các rule "NEVER do arithmetic in your head", "you MUST call calc_shipping", "Do NOT write your own Observation", kèm 1 few-shot example.
- **Kết quả dự kiến**: câu 5 sẽ gọi `calc_shipping(hanoi)` → $5 → ra đúng **$1085**; giảm parser error nhờ ví dụ mẫu ép đúng định dạng `Action:`/`Final Answer:`.
- **Ghi chú trung thực**: số liệu v2 cần chạy lại `python main.py agent` để định lượng đầy đủ; phần này nhóm sẽ cập nhật sau khi chạy lại trên cùng 5 câu test.

### Experiment 2: Chatbot vs Agent (data-driven)

| Câu | Đáp án đúng | Chatbot | Agent v1 | Winner |
| :--- | :--- | :--- | :--- | :--- |
| 1. Giá iphone? | $999 | ❌ liệt kê dải giá chung | ✅ $999 | **Agent** |
| 2. Mã WINNER giảm? | 10% | ❌ "không có dữ liệu" | ✅ 10% | **Agent** |
| 3. 2 iphone tổng? | $1998 | ❌ hỏi lại model | ✅ $1998 | **Agent** |
| 4. 2 iphone + WINNER? | $1798.2 | ❌ tự đoán $799 → $1438.2 | ⚠️ $1798.2 (đúng nhưng KHÔNG dùng tool) | **Agent** |
| 5. laptop+headphones+SALE20+hanoi | $1085 | ❌ hỏi lại giá | ❌ $1090 (hallucinate ship) | Draw (cả 2 sai) |

**Kết luận**: Agent thắng rõ ở mọi câu nhiều bước. Câu 4 cảnh báo một rủi ro: agent ra đúng số nhưng **bỏ qua tool, tự tính** — đúng may rủi, không bền vững (đây là động cơ cho prompt v2).

---

## 6. Production Readiness Review

- **Security**:
  - `calculator` đã sandbox (`eval` với `{"__builtins__": {}}` + whitelist ký tự) chống code injection.
  - API key để trong `.env` (đã `.gitignore`), hỗ trợ xoay vòng nhiều key.
- **Guardrails**:
  - `max_steps` chặn lặp vô hạn / chi phí token tăng vô kiểm soát.
  - Tool không tồn tại → trả thông báo rõ ràng thay vì crash (bắt hallucination).
  - Prompt v2 ép dùng tool cho mọi phép tính → giảm bịa số.
- **Scaling**:
  - Provider Pattern + KeyManager sẵn sàng thêm provider/model mới.
  - Hướng mở rộng: chuyển sang LangGraph cho nhánh logic phức tạp; thêm Vector DB để tool-retrieval khi số tool lớn; dùng native tool-calling (JSON schema) để loại bỏ hẳn parser error.

---

> [!NOTE]
> Trace đầy đủ trong `logs/2026-06-01.log`. UI demo: `streamlit run app.py`. Chạy bộ test: `python main.py`.
