from typing import Dict, Any, List
from models.llm_model import llm_model
from config.settings import settings


class NotEnoughInfoAgent:
    def __init__(self):
        self.name = "NOT_ENOUGH_INFO"

        # Prompt mới: yêu cầu LLM trả lời dựa trên kiến thức của nó
        self.prompt_template = """Bạn là nhân viên tư vấn khách hàng.

        TÌNH HUỐNG: Không có đủ dữ liệu trong hệ thống để trả lời chính xác.

        Câu hỏi: "{question}"

        YÊU CẦU BẮT BUỘC:
        - Trả lời NGẮN GỌN (tối đa 3–4 câu)
        - KHÔNG phân tích dài, KHÔNG kể ví dụ
        - Chỉ cung cấp thông tin mang tính tham khảo chung
        - KHÔNG suy đoán chi tiết kỹ thuật
        - Kết thúc bằng đề nghị liên hệ hotline

        CẤU TRÚC TRẢ LỜI:
        1. 1 câu mở đầu: “Dựa trên tổng hợp từ các nguồn thông tin, bạn có thể tham khảo như sau:”
        2. 1–2 câu thông tin chung
        3. 1 câu đề nghị liên hệ {support_phone}

        Chỉ trả về nội dung trả lời, không giải thích gì thêm.
        """

    def process(self, question: str, **kwargs) -> Dict[str, Any]:
        """Xử lý trường hợp không đủ thông tin - nhưng vẫn cố gắng hỗ trợ"""
        try:
            prompt = self.prompt_template.format(
                question=question,
                support_phone=settings.SUPPORT_PHONE
            )

            answer = llm_model.invoke(
                prompt,
                temperature=0.2,  # ↓ sáng tạo
                top_p=0.7,
                max_tokens=120,  # ↓ token đầu ra
                frequency_penalty=0.5,
                presence_penalty=0.0
            )

            return {
                "status": "SUCCESS",
                "answer": answer,
                "references": [
                    {
                        "document_id": "llm_knowledge",
                        "type": "GENERAL_KNOWLEDGE"
                    }
                ],
                "next_agent": "end"
            }

        except Exception as e:
            return {
                "status": "ERROR",
                "answer": f"Xin lỗi, hệ thống gặp lỗi: {str(e)}. Vui lòng liên hệ {settings.SUPPORT_PHONE} để được hỗ trợ.",
                "references": [],
                "next_agent": "end"
            }