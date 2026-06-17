# RAG_Core/agents/generator_agent.py

from typing import Dict, Any, List, AsyncIterator
from models.llm_model import llm_model
import logging

logger = logging.getLogger(__name__)


class GeneratorAgent:
    def __init__(self):
        self.name = "GENERATOR"

        self.standard_prompt = """Bạn là trợ lý ảo giải quyết công việc nội bộ tại Ngân hàng Chính sách Xã hội Việt Nam (VBSP,NHCSXH ).

Câu hỏi của người dùng: "{question}"

Thông tin tham khảo từ tài liệu nội bộ (đã đánh số, kèm nguồn):
{documents}

Lịch sử trò chuyện gần đây:
{history}

YÊU CẦU BẮT BUỘC VỀ DẪN CHỨNG (tuân thủ tuyệt đối, không có ngoại lệ):
- CHỈ trả lời dựa trên nội dung trong phần "Thông tin tham khảo"/"Tài liệu liên quan" ở trên, không tự suy diễn thêm.
- Mỗi đoạn trích đã có sẵn dòng "DẪN CHỨNG BẮT BUỘC KHI DÙNG ĐOẠN NÀY: (Nguồn: ...)". Khi dùng thông tin từ đoạn nào, PHẢI chèn NGUYÊN VĂN đúng dẫn chứng đó ngay sau, viết đầy đủ markdown link — không rút gọn, không viết lại theo cách khác.
- TUYỆT ĐỐI KHÔNG dùng các nhãn đánh số như "Tài liệu 1", "Tài liệu 2", "Đoạn trích 2"... để thay cho dẫn chứng. Các nhãn đó chỉ để phân biệt các đoạn trích trong prompt này, KHÔNG được xuất hiện trong câu trả lời của bạn dưới bất kỳ hình thức nào.
- Nếu phải trích dẫn cùng một nguồn nhiều lần trong câu trả lời, mỗi lần đều phải viết lại đầy đủ link markdown gốc — không viết tắt ở các lần sau.
- Nếu thông tin lấy từ nhiều đoạn trích/nguồn khác nhau, trích dẫn đầy đủ từng nguồn liên quan, không bỏ sót.

Yêu cầu trả lời:
- Giọng văn tự nhiên, ngắn gọn, súc tích, đi thẳng vào vấn đề
- Diễn đạt theo cách hiểu của bạn nhưng không làm sai lệch nội dung tài liệu
- Kết thúc bằng câu hỏi ngắn để tiếp tục hỗ trợ nếu cần

Hãy trả lời như đang trao đổi trực tiếp với đồng nghiệp:"""

        self.followup_prompt = """Bạn là trợ lý ảo giải quyết công việc nội bộ tại Ngân hàng Chính sách Xã hội Việt Nam (VBSP,NHCSXH ).

🔍 NGỮ CẢNH CUỘC TRÒ CHUYỆN:
{context_summary}

📝 LỊCH SỬ GẦN NHẤT:
{recent_history}

❓ CÂU HỎI FOLLOW-UP CỦA NGƯỜI DÙNG: "{question}"

📚 THÔNG TIN TÀI LIỆU LIÊN QUAN (đã đánh số, kèm nguồn):
{documents}

YÊU CẦU BẮT BUỘC VỀ DẪN CHỨNG (tuân thủ tuyệt đối, không có ngoại lệ):
- CHỈ trả lời dựa trên nội dung trong phần "Thông tin tham khảo"/"Tài liệu liên quan" ở trên, không tự suy diễn thêm.
- Mỗi đoạn trích đã có sẵn dòng "DẪN CHỨNG BẮT BUỘC KHI DÙNG ĐOẠN NÀY: (Nguồn: ...)". Khi dùng thông tin từ đoạn nào, PHẢI chèn NGUYÊN VĂN đúng dẫn chứng đó ngay sau, viết đầy đủ markdown link — không rút gọn, không viết lại theo cách khác.
- TUYỆT ĐỐI KHÔNG dùng các nhãn đánh số như "Tài liệu 1", "Tài liệu 2", "Đoạn trích 2"... để thay cho dẫn chứng. Các nhãn đó chỉ để phân biệt các đoạn trích trong prompt này, KHÔNG được xuất hiện trong câu trả lời của bạn dưới bất kỳ hình thức nào.
- Nếu phải trích dẫn cùng một nguồn nhiều lần trong câu trả lời, mỗi lần đều phải viết lại đầy đủ link markdown gốc — không viết tắt ở các lần sau.
- Nếu thông tin lấy từ nhiều đoạn trích/nguồn khác nhau, trích dẫn đầy đủ từng nguồn liên quan, không bỏ sót.

⚠️ YÊU CẦU ĐẶC BIỆT cho follow-up question:
1. Nhận biết rằng người dùng đang hỏi tiếp về chủ đề đã thảo luận
2. Tham chiếu đến thông tin đã cung cấp trước đó một cách tự nhiên
3. Trả lời cụ thể vào phần mà người dùng muốn biết thêm
4. KHÔNG lặp lại toàn bộ thông tin đã nói, chỉ tập trung vào phần được hỏi

📋 YÊU CẦU CHUNG:
- Giọng văn tự nhiên, ngắn gọn, súc tích, đúng trọng tâm
- Kết thúc bằng câu hỏi để tiếp tục hỗ trợ nếu cần

Hãy trả lời:"""

    def _deduplicate_references(self, references: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not references:
            return []
        seen, unique = set(), []
        for ref in references:
            doc_id = ref.get('document_id')
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                unique.append(ref)
        return unique

    def _format_documents(self, documents: List[Dict[str, Any]]) -> str:
        """
        Format chunk tài liệu kèm URL (đã được enrich bởi document_url_service
        ở rag_workflow._enrich_references_with_urls trước khi gọi Generator),
        để LLM chèn trích dẫn (Nguồn: ...) chính xác.

        Lưu ý quan trọng: KHÔNG dùng nhãn dạng "[Tài liệu N]" làm tiêu đề đoạn,
        vì LLM dễ nhầm lẫn dùng lại nhãn rút gọn này thay cho dẫn chứng thật
        (link markdown) khi phải trích dẫn lại cùng một nguồn nhiều lần.
        """
        if not documents:
            return "Không có tài liệu tham khảo"

        lines = []
        for i, doc in enumerate(documents[:5], 1):
            content = doc.get('content_with_ctx') or doc.get('content') or doc.get('description', '')
            score = doc.get('rerank_score', doc.get('similarity_score', 0))
            doc_id = doc.get('document_id', 'unknown')
            url = doc.get('url')
            filename = doc.get('filename') or doc_id
            section = doc.get('section_path', '')
            page = doc.get('page_num', 0)

            citation = f"[{filename}]({url})" if url else f"<{doc_id}>"
            meta_parts = []
            if section:
                meta_parts.append(f"Mục: {section}")
            if page:
                meta_parts.append(f"Trang: {page}")
            meta = " | ".join(meta_parts) if meta_parts else "Không có"

            lines.append(
                f"--- Đoạn trích #{i} (Độ liên quan: {score:.2f}) ---\n"
                f"DẪN CHỨNG BẮT BUỘC KHI DÙNG ĐOẠN NÀY: (Nguồn: {citation})\n"
                f"Vị trí: {meta}\n"
                f"Nội dung: {content}"
            )

        return "\n\n".join(lines)

    def _format_history(self, history: List, max_turns: int = 2) -> str:
        if not history:
            return "Không có lịch sử"

        normalized_history = []
        for msg in history:
            if isinstance(msg, dict):
                normalized_history.append({"role": msg.get("role", ""), "content": msg.get("content", "")})
            else:
                normalized_history.append({"role": getattr(msg, "role", ""), "content": getattr(msg, "content", "")})

        recent_history = normalized_history[-(max_turns * 2):] if len(normalized_history) > max_turns * 2 else normalized_history

        lines = []
        for msg in recent_history:
            role = "👤 Người dùng" if msg.get("role") == "user" else "🤖 Trợ lý"
            content = msg.get("content", "")
            if content:
                lines.append(f"{role}: {content}")

        return "\n".join(lines) if lines else "Không có lịch sử"

    def _extract_context_summary(self, history: List) -> str:
        if not history or len(history) < 2:
            return "Đây là câu hỏi đầu tiên"

        normalized_history = []
        for msg in history:
            normalized_history.append(msg if isinstance(msg, dict) else {
                "role": getattr(msg, "role", ""), "content": getattr(msg, "content", "")
            })

        for i in range(len(normalized_history) - 1, -1, -1):
            if normalized_history[i].get("role") == "user":
                prev_question = normalized_history[i].get("content", "")
                for j in range(i + 1, len(normalized_history)):
                    if normalized_history[j].get("role") == "assistant":
                        prev_answer = normalized_history[j].get("content", "")
                        return f"Chủ đề đang thảo luận: {prev_question}\nĐã trả lời: {prev_answer[:200]}..."
                return f"Chủ đề đang thảo luận: {prev_question}"

        return "Đang trong cuộc trò chuyện"

    def process(
            self,
            question: str,
            documents: List[Dict[str, Any]],
            references: List[Dict[str, Any]] = None,
            history: List[Dict[str, str]] = None,
            is_followup: bool = False,
            context_summary: str = "",
            **kwargs
    ) -> Dict[str, Any]:
        try:
            if not documents:
                return {"status": "ERROR", "answer": "Không có tài liệu để tạo câu trả lời", "references": [], "next_agent": "end"}

            doc_text = self._format_documents(documents)
            history_text = self._format_history(history or [], max_turns=2)

            if is_followup:
                if not context_summary:
                    context_summary = self._extract_context_summary(history or [])
                prompt = self.followup_prompt.format(
                    question=question, context_summary=context_summary,
                    recent_history=history_text, documents=doc_text
                )
            else:
                prompt = self.standard_prompt.format(
                    question=question, history=history_text, documents=doc_text
                )

            answer = llm_model.invoke(prompt)
            if not answer or len(answer.strip()) < 10:
                answer = "Tôi đã tìm thấy thông tin liên quan nhưng gặp khó khăn trong việc tạo câu trả lời."

            unique_references = self._deduplicate_references(references or [])

            return {"status": "SUCCESS", "answer": answer, "references": unique_references, "next_agent": "end"}

        except Exception as e:
            logger.error(f"Error in generator agent: {e}", exc_info=True)
            return {"status": "ERROR", "answer": f"Lỗi tạo câu trả lời: {str(e)}", "references": [], "next_agent": "end"}

    async def process_streaming(
            self,
            question: str,
            documents: List[Dict[str, Any]],
            references: List[Dict[str, Any]] = None,
            history: List[Dict[str, str]] = None,
            is_followup: bool = False,
            context_summary: str = "",
            **kwargs
    ) -> AsyncIterator[str]:
        try:
            logger.info(f"🚀 Generator: Starting streaming for: {question[:50]}...")

            if not documents:
                yield "Không có tài liệu để tạo câu trả lời."
                return

            doc_text = self._format_documents(documents)
            history_text = self._format_history(history or [], max_turns=2)

            if is_followup:
                if not context_summary:
                    context_summary = self._extract_context_summary(history or [])
                prompt = self.followup_prompt.format(
                    question=question, context_summary=context_summary,
                    recent_history=history_text, documents=doc_text
                )
            else:
                prompt = self.standard_prompt.format(
                    question=question, history=history_text, documents=doc_text
                )

            logger.info(f"📝 Generator: Prompt prepared, length={len(prompt)}")

            chunk_count = 0
            async for chunk in llm_model.astream(prompt):
                if chunk:
                    chunk_count += 1
                    yield chunk

            logger.info(f"✅ Generator: Completed streaming {chunk_count} chunks")

        except Exception as e:
            logger.error(f"❌ Generator streaming error: {e}", exc_info=True)
            yield f"\n\n[Lỗi: {str(e)}]"