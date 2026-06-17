# RAG_Core/agents/reporter_agent.py

from typing import Dict, Any
from tools.vector_search import check_database_connection
from config.settings import settings


class ReporterAgent:
    def __init__(self):
        self.name = "REPORTER"

    def process(self, question: str, **kwargs) -> Dict[str, Any]:
        try:
            db_status = check_database_connection.invoke({})

            if db_status.get("connected", False):
                answer = """Hệ thống đang hoạt động bình thường. Tôi có thể hỗ trợ bạn ngay bây giờ.

Vui lòng đặt câu hỏi và tôi sẽ tìm thông tin phù hợp cho bạn."""
            else:
                answer = f"""🔧 THÔNG BÁO BẢO TRÌ HỆ THỐNG

Hiện tại hệ thống đang trong quá trình bảo trì để nâng cấp và cải thiện chất lượng dịch vụ.

Tình trạng: {db_status.get("message", "Đang kiểm tra")}

Để được hỗ trợ ngay lập tức, bạn vui lòng:
📞 Gọi hotline: {settings.SUPPORT_PHONE}
⏰ Thời gian hỗ trợ: 24/7

Chúng tôi xin lỗi về sự bất tiện này và cảm ơn sự kiên nhẫn của bạn!"""

            return {
                "status": "SUCCESS",
                "answer": answer,
                "references": [{"document_id": "system_status", "type": "SYSTEM"}],
                "next_agent": "end"
            }

        except Exception:
            return {
                "status": "ERROR",
                "answer": f"""Hệ thống đang gặp sự cố kỹ thuật.

Vui lòng liên hệ hotline {settings.SUPPORT_PHONE} để được hỗ trợ trực tiếp.

Xin lỗi về sự bất tiện này!""",
                "references": [],
                "next_agent": "end"
            }