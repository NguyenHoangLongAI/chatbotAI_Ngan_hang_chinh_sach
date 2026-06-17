# RAG_Core/agents/base_agent.py
"""
Base class cho tất cả agents với streaming support
"""

from typing import Dict, Any, List, AsyncIterator
from models.llm_model import llm_model
import logging

logger = logging.getLogger(__name__)


class BaseStreamingAgent:
    def __init__(self, name: str, prompt_template: str):
        self.name = name
        self.prompt_template = prompt_template

    def process(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Subclass must implement process()")

    async def process_streaming(self, **kwargs) -> AsyncIterator[str]:
        try:
            prompt = self._format_prompt(**kwargs)
            logger.info(f"🚀 {self.name}: Starting streaming")

            chunk_count = 0
            async for chunk in llm_model.astream(prompt):
                if chunk:
                    chunk_count += 1
                    yield chunk

            logger.info(f"✅ {self.name}: Completed {chunk_count} chunks")

        except Exception as e:
            logger.error(f"❌ {self.name} streaming error: {e}", exc_info=True)
            yield f"\n\n[Lỗi {self.name}: {str(e)}]"

    def _format_prompt(self, **kwargs) -> str:
        raise NotImplementedError("Subclass must implement _format_prompt()")

    def _get_fallback_answer(self, **kwargs) -> str:
        return "Xin lỗi, tôi không thể xử lý yêu cầu này lúc này."


# ============================================================================
# STREAMING-ENABLED AGENTS (đồng bộ persona với các agent file riêng)
# ============================================================================

class StreamingChatterAgent(BaseStreamingAgent):
    def __init__(self):
        prompt_template = """Bạn là trợ lý ảo giải quyết công việc nội bộ tại Ngân hàng Chính sách Xã hội Việt Nam (VBSP,NHCSXH ), thân thiện và chuyên nghiệp - chuyên xử lý cảm xúc và an ủi người dùng nội bộ.

Nhiệm vụ: An ủi, làm dịu cảm xúc tiêu cực của người dùng và cung cấp thông tin liên hệ hỗ trợ.

Nội dung người dùng: "{question}"
Lịch sử hội thoại: {history}
Số điện thoại hỗ trợ: {support_phone}

Hướng dẫn:
1. Thể hiện sự thông cảm và hiểu biết cảm xúc người dùng
2. Xin lỗi một cách chân thành
3. Đảm bảo sẽ cải thiện hỗ trợ
4. Cung cấp số hotline để được hỗ trợ trực tiếp
5. Giữ thái độ ấm áp, chuyên nghiệp

Trả lời:"""
        super().__init__("CHATTER", prompt_template)
        self.support_phone = None

    def _format_prompt(self, question: str, history: List = None, support_phone: str = "", **kwargs) -> str:
        history_text = "\n".join(history) if history else "Không có lịch sử"
        return self.prompt_template.format(question=question, history=history_text, support_phone=support_phone)

    def process(self, question: str, history: List = None, **kwargs) -> Dict[str, Any]:
        try:
            from config.settings import settings
            prompt = self._format_prompt(question=question, history=history, support_phone=settings.SUPPORT_PHONE)
            answer = llm_model.invoke(prompt)
            if not answer or len(answer.strip()) < 10:
                answer = self._get_fallback_answer()
            return {
                "status": "SUCCESS", "answer": answer,
                "references": [{"document_id": "support_contact", "type": "SUPPORT"}],
                "next_agent": "end"
            }
        except Exception:
            return {"status": "ERROR", "answer": self._get_fallback_answer(), "references": [], "next_agent": "end"}


class StreamingOtherAgent(BaseStreamingAgent):
    def __init__(self):
        prompt_template = """Bạn là trợ lý ảo giải quyết công việc nội bộ tại Ngân hàng Chính sách Xã hội Việt Nam (VBSP), thân thiện và chuyên nghiệp - xử lý các yêu cầu ngoài phạm vi hỗ trợ.

Nhiệm vụ: Thông báo lịch sự khi yêu cầu nằm ngoài phạm vi và hướng dẫn người dùng.

Yêu cầu của người dùng: "{question}"
Số điện thoại hỗ trợ: {support_phone}

Hướng dẫn:
1. Giải thích rằng yêu cầu nằm ngoài phạm vi hỗ trợ hiện tại
2. Đề xuất liên hệ hotline để được tư vấn cụ thể hơn
3. Giữ thái độ lịch sự và chuyên nghiệp
4. Không từ chối một cách thô lỗ

Trả lời:"""
        super().__init__("OTHER", prompt_template)

    def _format_prompt(self, question: str, support_phone: str = "", **kwargs) -> str:
        return self.prompt_template.format(question=question, support_phone=support_phone)

    def process(self, question: str, **kwargs) -> Dict[str, Any]:
        try:
            from config.settings import settings
            prompt = self._format_prompt(question=question, support_phone=settings.SUPPORT_PHONE)
            answer = llm_model.invoke(prompt)
            if not answer or len(answer.strip()) < 10:
                answer = self._get_fallback_answer()
            return {"status": "SUCCESS", "answer": answer, "references": [], "next_agent": "end"}
        except Exception:
            return {"status": "ERROR", "answer": self._get_fallback_answer(), "references": [], "next_agent": "end"}


class StreamingNotEnoughInfoAgent(BaseStreamingAgent):
    def __init__(self):
        # Không dùng LLM để tránh bịa thông tin khi thiếu tài liệu
        super().__init__("NOT_ENOUGH_INFO", prompt_template="")

    def _build_message(self, support_phone: str = "") -> str:
        from config.settings import settings
        phone = support_phone or settings.SUPPORT_PHONE
        return (
            "Xin lỗi, hiện tại tôi không tìm thấy tài liệu hoặc thông tin nội bộ "
            "liên quan đến câu hỏi của bạn trong hệ thống.\n\n"
            f"Bạn vui lòng liên hệ hotline {phone} để được hỗ trợ chính xác hơn, "
            "hoặc thử đặt lại câu hỏi với từ khóa/nghiệp vụ cụ thể hơn."
        )

    async def process_streaming(self, question: str, support_phone: str = "", **kwargs):
        message = self._build_message(support_phone)
        for word in message.split(" "):
            yield word + " "

    def process(self, question: str, support_phone: str = "", **kwargs) -> Dict[str, Any]:
        return {
            "status": "NOT_FOUND",
            "answer": self._build_message(support_phone),
            "references": [],          # không có nguồn → không hiển thị ref-card
            "next_agent": "end"
        }