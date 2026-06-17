# RAG_Core/agents/supervisor.py

from typing import Dict, Any, List
from models.llm_model import llm_model
import logging
import json
import re

logger = logging.getLogger(__name__)


class SupervisorAgent:
    def __init__(self):
        self.name = "SUPERVISOR"
        self.classification_prompt = """Bạn là trợ lý ảo giải quyết công việc nội bộ tại Ngân hàng Chính sách Xã hội Việt Nam (VBSP,NHCSXH) — người điều phối chính (Supervisor) của hệ thống chatbot nội bộ.

        Nhiệm vụ:
        1. Dựa vào lịch sử hội thoại và câu hỏi hiện tại, hãy xác định ngữ cảnh (context) mà người dùng đang đề cập đến.
        2. Làm rõ câu hỏi nếu cần thiết (thay thế đại từ, bổ sung thông tin từ context).
        3. Phân loại câu hỏi và chọn agent phù hợp để xử lý.

        Các agent có thể chọn:
        - HELLO:
            - Chào hỏi thông thường (xin chào, hello, hi, chào buổi sáng...).
            - Cảm ơn hoặc khen ngợi trợ lý ("cảm ơn bạn", "bạn giỏi quá", "tuyệt vời"...).
            - Hỏi thăm trợ lý ("bạn có khỏe không", "hôm nay thế nào"...).
            - Yêu cầu giới thiệu bản thân ("bạn là ai", "bạn tên gì", "bạn có thể làm gì"...).
        - FAQ:
            - Câu hỏi về nghiệp vụ, quy trình, quy định, chính sách tín dụng, sản phẩm cho vay nội bộ của VBSP.
            - Hướng dẫn sử dụng hệ thống/công cụ/phần mềm nghiệp vụ nội bộ.
            - Quy định nhân sự, văn bản pháp lý liên quan đến hoạt động của Ngân hàng Chính sách Xã hội.
        - OTHER: Câu hỏi hoặc yêu cầu nằm ngoài phạm vi nghiệp vụ nội bộ VBSP.
        - CHATTER: Người dùng có dấu hiệu không hài lòng, giận dữ, hoặc cần được an ủi, làm dịu.
        - REPORTER: Khi người dùng phản ánh lỗi, mất kết nối, hoặc vấn đề kỹ thuật của hệ thống.

        Đầu vào:
        Câu hỏi hiện tại: "{question}"
        Lịch sử hội thoại: {history}

        YÊU CẦU QUAN TRỌNG:
        - Phân tích xem câu hỏi có phải follow-up (tiếp theo cuộc trò chuyện trước) không
            - Truy vết lịch sử để xác định chính xác đối tượng được nhắc tới.
            - Đặc biệt chú ý các cụm:
             "thành phần thứ X", "phần này", "nó", "ý trên", "cái đó", "OK","có", "chi tiết","hãy hướng dẫn", "tiếp tục" ...
            - Nếu lịch sử có DANH SÁCH ĐÁNH SỐ → ánh xạ theo ĐÚNG THỨ TỰ.
            - Nếu có yêu cầu hành động không cụ thể ("OK","có", "chi tiết","hãy hướng dẫn", "tiếp tục"...) → dựa vào lịch sử hội thoại làm rõ yêu cầu
            - Viết lại câu hỏi (contextualized_question) bằng TIẾNG VIỆT ĐẦY ĐỦ – RÕ NGHĨA – CÓ NGỮ CẢNH.
            - Đảm bảo câu hỏi được làm rõ (contextualized_question) phải có:
                - ĐỐI TƯỢNG cụ thể là gì
                - HÀNH ĐỘNG cụ thể là gì
                - Trong NGỮ CẢNH cụ thể là gì
        - Nếu không phải follow-up: contextualized_question = câu hỏi gốc, context_summary = "Câu hỏi độc lập"

        Hãy trả lời đúng định dạng JSON:
        {{
          "is_followup": true hoặc false,
          "contextualized_question": "Câu hỏi đã được làm rõ rất cụ thể hoặc câu hỏi gốc",
          "context_summary": "Tóm tắt ngắn gọn ngữ cảnh BẰNG TIẾNG VIỆT",
          "agent": "FAQ" hoặc "HELLO" hoặc "CHATTER" hoặc "REPORTER" hoặc "OTHER"
        }}

        Chỉ trả về JSON, không thêm text nào khác."""

    def classify_request(
            self,
            question: str,
            history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        try:
            logger.info("-" * 50)
            logger.info("👨‍💼 SUPERVISOR CLASSIFICATION")
            logger.info("-" * 50)
            logger.info(f"📝 Question: '{question}'")
            logger.info(f"📚 History Length: {len(history) if history else 0} messages")

            history_text = self._format_history(history or [])

            prompt = self.classification_prompt.format(
                question=question,
                history=history_text,
            )

            logger.info("🤖 Calling LLM for classification + contextualization...")
            response = llm_model.invoke(prompt)

            classification = self._parse_classification_response(response)

            agent_choice = classification.get("agent", "").upper()
            is_followup = classification.get("is_followup", False)
            contextualized_question = classification.get("contextualized_question", question)
            context_summary = classification.get("context_summary", "")

            valid_agents = ["FAQ", "HELLO", "CHATTER", "REPORTER", "OTHER"]
            if agent_choice not in valid_agents:
                logger.warning(f"⚠️  Invalid agent '{agent_choice}' → default to FAQ")
                agent_choice = "FAQ"

            logger.info(f"\n🎯 CLASSIFICATION RESULT:")
            logger.info(f"   Agent: {agent_choice}")
            logger.info(f"   Is Follow-up: {is_followup}")
            logger.info(f"   Original Q: '{question[:60]}'")
            logger.info(f"   Context Q:  '{contextualized_question}'")
            logger.info(f"   Context Summary: '{context_summary}'")
            logger.info("-" * 50 + "\n")

            return {
                "agent": agent_choice,
                "contextualized_question": contextualized_question,
                "context_summary": context_summary,
                "is_followup": is_followup,
                "reasoning": classification.get("reasoning", "")
            }

        except Exception as e:
            logger.error(f"❌ Error in supervisor classification: {e}", exc_info=True)
            return {
                "agent": "FAQ",
                "contextualized_question": question,
                "context_summary": "",
                "is_followup": False,
                "reasoning": "Error - default to FAQ"
            }

    def _parse_classification_response(self, response: str) -> Dict[str, Any]:
        try:
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                parsed.setdefault("is_followup", False)
                parsed.setdefault("contextualized_question", "")
                parsed.setdefault("context_summary", "")
                parsed.setdefault("agent", "FAQ")
                return parsed

            return {
                "agent": "FAQ", "is_followup": False,
                "contextualized_question": "", "context_summary": "",
                "reasoning": "Parse failed"
            }
        except Exception as e:
            logger.error(f"Error parsing classification response: {e}")
            return {
                "agent": "FAQ", "is_followup": False,
                "contextualized_question": "", "context_summary": "",
                "reasoning": "Parse error"
            }

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        if not history:
            return "Không có lịch sử"

        recent_history = history[-6:] if len(history) > 6 else history
        history_lines = []
        for msg in recent_history:
            if isinstance(msg, dict):
                role = "Người dùng" if msg.get("role") == "user" else "Trợ lý"
                content = msg.get("content", "")[:200]
            else:
                role = "Người dùng" if getattr(msg, "role", "") == "user" else "Trợ lý"
                content = getattr(msg, "content", "")[:200]

            if content:
                history_lines.append(f"{role}: {content}")

        return "\n".join(history_lines) if history_lines else "Không có lịch sử"