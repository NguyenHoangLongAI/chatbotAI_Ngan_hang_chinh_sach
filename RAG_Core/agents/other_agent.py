# RAG_Core/agents/other_agent.py

from typing import Dict, Any
from models.llm_model import llm_model
from config.settings import settings


class OtherAgent:
    def __init__(self):
        self.name = "OTHER"
        self.prompt_template = """Bạn là trợ lý ảo giải quyết công việc nội bộ tại Ngân hàng Chính sách Xã hội Việt Nam (VBSP), thân thiện và chuyên nghiệp - xử lý các yêu cầu ngoài phạm vi hỗ trợ.

Nhiệm vụ: Thông báo lịch sự khi yêu cầu nằm ngoài phạm vi và hướng dẫn người dùng.

Yêu cầu của người dùng: "{question}"
Số điện thoại hỗ trợ: {support_phone}

Hướng dẫn:
1. Giải thích rằng yêu cầu nằm ngoài phạm vi hỗ trợ hiện tại
2. Đề xuất liên hệ hotline để được tư vấn cụ thể hơn
3. Giữ thái độ lịch sự và chuyên nghiệp
4. Không từ chối một cách thô lỗ

Trả lời:"""

    def process(self, question: str, **kwargs) -> Dict[str, Any]:
        try:
            prompt = self.prompt_template.format(question=question, support_phone=settings.SUPPORT_PHONE)
            answer = llm_model.invoke(prompt)

            if not answer or len(answer.strip()) < 10:
                answer = f"""Cảm ơn bạn đã liên hệ!

Yêu cầu của bạn có vẻ nằm ngoài phạm vi hỗ trợ hiện tại của tôi.

Để được tư vấn và hỗ trợ tốt nhất cho yêu cầu cụ thể này, bạn vui lòng:
📞 Liên hệ hotline: {settings.SUPPORT_PHONE}
⏰ Thời gian: 24/7"""

            return {"status": "SUCCESS", "answer": answer, "references": [], "next_agent": "end"}

        except Exception:
            return {
                "status": "ERROR",
                "answer": f"Đây không phải là tác vụ của tôi. Vui lòng liên hệ {settings.SUPPORT_PHONE} để được hỗ trợ.",
                "references": [],
                "next_agent": "end"
            }