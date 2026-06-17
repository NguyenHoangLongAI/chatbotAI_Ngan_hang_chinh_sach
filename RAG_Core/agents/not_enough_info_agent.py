from typing import Dict, Any
from config.settings import settings


class NotEnoughInfoAgent:
    def __init__(self):
        self.name = "NOT_ENOUGH_INFO"

    def process(self, question: str, **kwargs) -> Dict[str, Any]:
        message = (
            "Xin lỗi, hiện tại tôi không tìm thấy tài liệu hoặc thông tin nội bộ "
            "liên quan đến câu hỏi của bạn trong hệ thống.\n\n"
            f"Bạn vui lòng liên hệ hotline {settings.SUPPORT_PHONE} để được hỗ trợ chính xác hơn, "
            "hoặc thử đặt lại câu hỏi với từ khóa/nghiệp vụ cụ thể hơn."
        )
        return {
            "status": "NOT_FOUND",
            "answer": message,
            "references": [],
            "next_agent": "end"
        }