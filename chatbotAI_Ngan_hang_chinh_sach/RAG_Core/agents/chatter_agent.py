from typing import Dict, Any, List
from models.llm_model import llm_model
from config.settings import settings


class ChatterAgent:
    def __init__(self):
        self.name = "CHATTER"
        self.prompt_template = """Bạn là chuyên viên chăm sóc khách hàng tại Ngân hàng Chính sách Xã hội Việt Nam (VBSP), thân thiện và chuyên nghiệp - chuyên gia xử lý cảm xúc và an ủi khách hàng.

Nhiệm vụ: An ủi, làm dịu cảm xúc tiêu cực của khách hàng và cung cấp thông tin liên hệ hỗ trợ.

Nội dung khách hàng: "{question}"
Lịch sử hội thoại: {history}
Số điện thoại hỗ trợ: {support_phone}

Hướng dẫn:
1. Thể hiện sự thông cảm và hiểu biết cảm xúc khách hàng
2. Xin lỗi một cách chân thành
3. Đảm bảo sẽ cải thiện dịch vụ
4. Cung cấp số hotline để được hỗ trợ trực tiếp
5. Giữ thái độ ấm áp, chuyên nghiệp

Trả lời:"""

    def process(self, question: str, history: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Xử lý cảm xúc tiêu cực của khách hàng"""
        try:
            history_text = "\n".join(history) if history else "Không có lịch sử"

            prompt = self.prompt_template.format(
                question=question,
                history=history_text,
                support_phone=settings.SUPPORT_PHONE
            )

            answer = llm_model.invoke(prompt)

            # Fallback answer
            if not answer or len(answer.strip()) < 10:
                answer = f"""Tôi rất hiểu cảm xúc của bạn và chân thành xin lỗi về những bất tiện này.

Ý kiến của bạn rất quan trọng với chúng tôi và chúng tôi sẽ không ngừng cải thiện để mang đến trải nghiệm tốt hơn.

Để được hỗ trợ trực tiếp và giải quyết nhanh chóng, bạn vui lòng liên hệ:
📞 Hotline: {settings.SUPPORT_PHONE}

Đội ngũ chuyên viên sẽ hỗ trợ bạn 24/7. Cảm ơn bạn đã chia sẻ!"""

            return {
                "status": "SUCCESS",
                "answer": answer,
                "references": [{"document_id": "support_contact", "type": "SUPPORT"}],
                "next_agent": "end"
            }

        except Exception as e:
            return {
                "status": "ERROR",
                "answer": f"Tôi hiểu bạn đang không hài lòng. Vui lòng liên hệ {settings.SUPPORT_PHONE} để được hỗ trợ tốt nhất.",
                "references": [],
                "next_agent": "end"
            }
