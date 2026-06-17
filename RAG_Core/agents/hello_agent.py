# RAG_Core/agents/hello_agent.py
"""
HelloAgent — xử lý chào hỏi, cảm ơn, khen ngợi, hỏi thăm, giới thiệu bản thân.
Không cần RAG, trả lời trực tiếp từ LLM với persona VBSP.
"""

from typing import Dict, Any, List, AsyncIterator
from models.llm_model import llm_model
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class HelloAgent:
    def __init__(self):
        self.name = "HELLO"
        self.prompt = """Bạn là {assistant_name} của {bank_name}.

Người dùng gửi: "{question}"

Hướng dẫn:
- Đây là tin nhắn chào hỏi / cảm ơn / khen ngợi / hỏi thăm / đề nghị giới thiệu bản thân.
- Hãy phản hồi thân thiện, ấm áp, chuyên nghiệp đúng phong cách nhân viên ngân hàng.
- Nếu người dùng hỏi bạn là ai: giới thiệu ngắn gọn tên, đơn vị, và bạn có thể hỗ trợ gì.
- Nếu người dùng cảm ơn hoặc khen: đón nhận lịch sự và mời tiếp tục hỗ trợ.
- Trả lời ngắn gọn, KHÔNG quá 3 câu.
- KHÔNG bịa thông tin nghiệp vụ.

Trả lời:"""

    def _build_prompt(self, question: str) -> str:
        return self.prompt.format(
            assistant_name=settings.ASSISTANT_NAME,
            bank_name=settings.BANK_FULL_NAME,
            question=question,
        )

    def process(self, question: str, **kwargs) -> Dict[str, Any]:
        try:
            logger.info(f"👋 HelloAgent processing: '{question[:60]}'")
            answer = llm_model.invoke(self._build_prompt(question))
            return {
                "status": "SUCCESS",
                "answer": answer or f"Xin chào! Tôi là {settings.ASSISTANT_NAME}, sẵn sàng hỗ trợ bạn.",
                "references": [],
                "next_agent": "end",
            }
        except Exception as e:
            logger.error(f"❌ HelloAgent error: {e}")
            return {
                "status": "SUCCESS",
                "answer": f"Xin chào! Tôi là {settings.ASSISTANT_NAME}. Bạn cần hỗ trợ gì?",
                "references": [],
                "next_agent": "end",
            }

    async def process_streaming(self, question: str, **kwargs) -> AsyncIterator[str]:
        try:
            prompt = self._build_prompt(question)
            async for chunk in llm_model.astream(prompt):
                if chunk:
                    yield chunk
        except Exception as e:
            logger.error(f"❌ HelloAgent streaming error: {e}")
            yield f"Xin chào! Tôi là {settings.ASSISTANT_NAME}. Bạn cần hỗ trợ gì?"