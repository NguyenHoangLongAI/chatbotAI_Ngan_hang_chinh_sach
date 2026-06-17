# RAG_Core/agents/faq_agent.py

from typing import Dict, Any, List, AsyncIterator
from models.llm_model import llm_model
from tools.vector_search import search_faq, rerank_faq
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class FAQAgent:
    def __init__(self):
        self.name = "FAQ"
        self.vector_threshold = settings.FAQ_VECTOR_THRESHOLD
        self.rerank_threshold = settings.FAQ_RERANK_THRESHOLD

        self.llm_prompt = """Bạn là trợ lý ảo giải quyết công việc nội bộ tại Ngân hàng Chính sách Xã hội Việt Nam (VBSP).

Câu hỏi của người dùng: "{question}"

Kết quả tìm kiếm FAQ (đã được rerank theo độ phù hợp):
{faq_results}

YÊU CẦU BẮT BUỘC VỀ DẪN CHỨNG:
- CHỈ trả lời dựa trên nội dung FAQ ở trên, không tự suy diễn thêm.
- Ngay sau MỖI thông tin/số liệu/quy định lấy từ FAQ, PHẢI chèn dẫn chứng: (Nguồn: FAQ - [nội dung câu hỏi FAQ]).
- Nếu kết hợp nhiều FAQ, trích dẫn đầy đủ từng nguồn tương ứng, không bỏ sót.

Hướng dẫn trả lời:
1. Dựa vào FAQ có rerank_score cao nhất để trả lời chính.
2. Nếu không có FAQ nào phù hợp (tất cả score quá thấp), trả về "NOT_FOUND".
3. Trả lời bằng tiếng Việt, thân thiện và chính xác.
4. Có thể kết hợp thông tin từ nhiều FAQ nếu cần.
5. Đừng nói "Dựa vào FAQ..." hay "Theo tài liệu..." ở đầu câu — trả lời trực tiếp.

Trả lời:"""

    def process(
            self,
            question: str,
            is_followup: bool = False,
            context: str = "",
            **kwargs
    ) -> Dict[str, Any]:
        try:
            logger.info("=" * 50)
            logger.info("🤖 FAQ AGENT PROCESSING (NON-STREAMING)")
            logger.info("=" * 50)
            logger.info(f"📝 Question: '{question[:100]}'")

            faq_results = search_faq.invoke({"query": question})
            if not faq_results or "error" in str(faq_results):
                return self._route_to_retriever("Vector search failed")

            filtered_faqs = [
                faq for faq in faq_results
                if faq.get("similarity_score", 0) >= self.vector_threshold
            ]
            if not filtered_faqs:
                return self._route_to_retriever("No FAQ above vector threshold")

            reranked_faqs = rerank_faq.invoke({"query": question, "faq_results": filtered_faqs})
            if not reranked_faqs:
                raise RuntimeError("FAQ reranking failed")

            best_faq = reranked_faqs[0]
            rerank_score = best_faq.get("rerank_score", 0)
            similarity_score = best_faq.get("similarity_score", 0)

            if rerank_score < self.rerank_threshold:
                return self._route_to_retriever(f"Rerank score too low: {rerank_score:.3f}")

            faq_text = self._format_reranked_faq(reranked_faqs[:3])
            prompt = self.llm_prompt.format(question=question, faq_results=faq_text)
            response = llm_model.invoke(prompt)

            if "NOT_FOUND" in response.upper():
                return self._route_to_retriever("LLM rejected FAQ")
            if not response or len(response.strip()) < 10:
                return self._route_to_retriever("Answer too short")

            return {
                "status": "SUCCESS",
                "answer": response,
                "mode": "llm",
                "references": [
                    {
                        "document_id": best_faq.get("faq_id"),
                        "type": "FAQ",
                        "description": best_faq.get("question", ""),
                        "rerank_score": round(rerank_score, 4),
                        "similarity_score": round(similarity_score, 4)
                    }
                ],
                "next_agent": "end"
            }

        except RuntimeError as e:
            logger.error(f"❌ Critical FAQ error: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error in FAQ agent: {e}", exc_info=True)
            raise RuntimeError(f"FAQ agent failed: {e}") from e

    async def process_streaming(
            self,
            question: str,
            reranked_faqs: List[Dict[str, Any]] = None,
            is_followup: bool = False,
            context: str = "",
            **kwargs
    ) -> AsyncIterator[str]:
        try:
            if not reranked_faqs:
                yield "Không tìm thấy câu trả lời."
                return

            faq_text = self._format_reranked_faq(reranked_faqs[:3])
            prompt = self.llm_prompt.format(question=question, faq_results=faq_text)

            async for chunk in llm_model.astream(prompt):
                if chunk:
                    yield chunk

        except Exception as e:
            logger.error(f"❌ FAQ streaming error: {e}", exc_info=True)
            yield f"\n\n[Lỗi FAQ: {str(e)}]"

    def _format_reranked_faq(self, faq_results: List[Dict[str, Any]]) -> str:
        if not faq_results:
            return "Không tìm thấy FAQ phù hợp"

        lines = []
        for i, faq in enumerate(faq_results, 1):
            lines.append(
                f"FAQ {i} (Rerank: {faq.get('rerank_score', 0):.3f}, "
                f"Similarity: {faq.get('similarity_score', 0):.3f}):\n"
                f"Q: {faq.get('question', '')}\n"
                f"A: {faq.get('answer', '')}\n"
            )
        return "\n".join(lines)

    def _route_to_retriever(self, reason: str) -> Dict[str, Any]:
        logger.info(f"→ Routing to RETRIEVER: {reason}")
        return {"status": "NOT_FOUND", "answer": "", "references": [], "next_agent": "RETRIEVER"}

    def set_thresholds(self, vector_threshold: float = None, rerank_threshold: float = None):
        if vector_threshold is not None:
            self.vector_threshold = vector_threshold
        if rerank_threshold is not None:
            self.rerank_threshold = rerank_threshold