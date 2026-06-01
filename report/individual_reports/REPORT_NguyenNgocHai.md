# Individual Report: Lab 3 - Chatbot vs ReAct Agent

- **Student Name**: Nguyễn Ngọc Hải
- **Student ID**: 2A202600614
- **Date**: 2026-06-01
- **Team**: 16383

---

## I. Technical Contribution (15 Points)

Tôi phụ trách **Telemetry, UI (Streamlit) và thu thập trace**.

- **Modules Implemented**:
  - `app.py`: giao diện web phong cách ChatGPT — chọn provider/model, 3 chế độ (So sánh / Chỉ Chatbot / Chỉ Agent), hiển thị từng bước gọi tool live, đo token + latency.
  - Tinh chỉnh `src/telemetry/logger.py`: tách log JSON ghi vào file `logs/`, giữ console sạch (không spam JSON).
  - Tích hợp `tracker.track_request()` vào cả Chatbot và Agent để đo công bằng.

- **Code Highlights**:
  ```python
  # logger ghi JSON vào file, console chỉ hiện cảnh báo (gọn)
  console_handler.setLevel(logging.INFO if verbose else logging.WARNING)
  ```
  ```python
  # UI: hiển thị từng bước Thought→Action→Observation gấp trong expander
  with st.expander(f"🔧 Xem {n_tools} bước suy luận"):
      ...
  ```

- **Documentation**: UI dùng `run_iter()` của agent để render từng sự kiện; mọi lượt chạy đều sinh telemetry vào `logs/<ngày>.log` phục vụ phân tích.

---

## II. Debugging Case Study (10 Points)

- **Problem Description**: Latency cao bất thường — có lượt Chatbot mất tới **64 giây**.
- **Log Source** (`logs/2026-06-01.log`):
  ```json
  {"event": "LLM_METRIC", "data": {"completion_tokens": 2424, "latency_ms": 64188}}
  {"event": "LLM_METRIC", "data": {"completion_tokens": 119,  "latency_ms": 3989}}
  ```
- **Diagnosis**: Khi đối chiếu các bản ghi `LLM_METRIC`, tôi thấy **latency tỉ lệ thuận với `completion_tokens`**: 2,424 token → 64s, còn 119 token → 4s. Nguyên nhân latency cao không phải mạng mà do model **sinh quá nhiều token** (model `mimo-v2.5-pro` trả lời dài dòng), cộng thêm việc Agent gọi LLM nhiều lượt nên cộng dồn.
- **Solution**: Đề xuất (a) giới hạn `max_tokens` cho mỗi lượt gọi, (b) siết prompt cho ngắn gọn (đã làm ở v2), (c) chọn model flash khi cần nhanh. Việc tách JSON khỏi console cũng giúp đọc kết quả demo dễ hơn.
- **Bổ sung (Logic Failure Debugging)**: Trong quá trình test hiển thị trên Streamlit UI do tôi phụ trách, Agent gặp lỗi logic ở câu 5. LLM sinh ra output thiếu tiền tố `Action:`, khiến parser không nhận diện được và hệ thống sinh lỗi PARSER_ERROR. Bằng việc phân tích log JSON (check `"error_code": "PARSER_ERROR"` và trường `raw` của text sinh ra), tôi đã cùng nhóm tìm ra nguyên nhân và siết lại quy tắc format trong prompt v2, thêm ví dụ cụ thể (few-shot) để ép agent tuân thủ đúng định dạng, giải quyết triệt để lỗi này.

---

## III. Personal Insights: Chatbot vs ReAct (10 Points)

1. **Reasoning**: Nhìn telemetry mới thấy rõ Agent "trả giá" cho khả năng suy luận: token và latency cao gấp ~2–3 lần Chatbot (11,178 vs 3,998 token cho 5 câu).
2. **Reliability**: Đắt hơn nhưng đáng — Agent đúng 4/5 còn Chatbot 0/5. Trong sản xuất, đây là đánh đổi cost vs accuracy cần cân nhắc theo từng loại câu hỏi.
3. **Observation**: Telemetry chính là "sự thật" của hệ thống. Không có log, ta chỉ đoán; có log, ta chứng minh được vì sao agent thắng và tốn bao nhiêu.

---

## IV. Future Improvements (5 Points)

- **Scalability (System)**: Thêm dashboard tổng hợp (P50/P99 latency, token trung bình, cost) đọc tự động từ thư mục `logs/`.
- **Safety & Performance**: Hiển thị cảnh báo trên UI khi agent chạm `max_steps` hoặc gặp lỗi, đồng thời stream token thời gian thực trên UI để giảm cảm giác chờ.
- **Production-level RAG Integration**: Hiện tại danh mục sản phẩm đang bị hardcode trong tool. Để scale lên nền tảng E-commerce thực tế, tôi đề xuất tích hợp **RAG (Retrieval-Augmented Generation)** kết hợp Vector Database (như Chroma/Qdrant). Agent sẽ gọi tool `search_catalog` truy vấn vector để lấy top-k sản phẩm liên quan thay vì tra cứu tên đích danh.
- **Multi-Agent System Architecture**: Khi các tool và nghiệp vụ tăng lên, prompt của một ReAct Agent đơn lẻ sẽ bị quá tải. Hướng giải quyết là chuyển đổi sang kiến trúc **Multi-Agent** (VD: ứng dụng LangGraph): chia nhỏ thành `ProductAgent` chuyên tra cứu RAG, `CheckoutAgent` chuyên tính toán hoá đơn, và một `SupervisorAgent` để phân loại ý định người dùng và điều phối giao việc.
