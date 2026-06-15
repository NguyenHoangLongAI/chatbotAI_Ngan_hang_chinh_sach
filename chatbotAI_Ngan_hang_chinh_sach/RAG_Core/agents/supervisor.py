# RAG_Core/agents/supervisor.py - REMOVED CONTEXT PROCESSOR

from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, SystemMessage
from models.llm_model import llm_model
import logging
import json
import re

logger = logging.getLogger(__name__)


class SupervisorAgent:
    def __init__(self):
        self.name = "SUPERVISOR"
        self.classification_prompt = """Bạn là chuyên viên đào tạo kỹ năng chuyển đổi số, kiến thức sử dụng công nghệ thông tin cơ bản cho người dân - người điều phối chính của hệ thống chatbot.

        Nhiệm vụ:
        1. Dựa vào lịch sử hội thoại và câu hỏi hiện tại, hãy xác định ngữ cảnh (context) mà người dùng đang đề cập đến.
        2. Làm rõ câu hỏi nếu cần thiết (thay thế đại từ, bổ sung thông tin từ context).
        3. Phân loại câu hỏi và chọn agent phù hợp để xử lý.

        Các agent có thể chọn:
        - FAQ: 
            - Dùng cho chào hỏi thân thiện, câu hỏi thường gặp. 
            - Các yêu cầu liên quan đến đào tạo kỹ năng chuyển đổi số cho người dân và doanh nghiệp.
            - Kiến thức về AI, CNTT.
            - An toàn giao thông tin, bảo mật.
            - Luật liên quan đến chuyển đổi số.
        - OTHER: Câu hỏi hoặc yêu cầu nằm ngoài phạm vi chuyển đổi số.
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
            -Đảm bảo câu hỏi được làm rõ (contextualized_question) phải có:
                - ĐỐI TƯỢNG cụ thể là gì
                - HÀNH ĐỘNG cụ thể là gì
                - Trong NGỮ CẢNH cụ thể là gì
        - Nếu không phải follow-up: contextualized_question = câu hỏi gốc, context_summary = "Câu hỏi độc lập"

        Hãy trả lời đúng định dạng JSON:
        {{
          "is_followup": true hoặc false,
          "contextualized_question": "Câu hỏi đã được làm rõ rất cụ thể hoặc câu hỏi gốc",
          "context_summary": "Tóm tắt ngắn gọn ngữ cảnh BẰNG TIẾNG VIỆT",
          "agent": "FAQ" hoặc "CHATTER" hoặc "REPORTER" hoặc "OTHER"
        }}

        Chỉ trả về JSON, không thêm text nào khác."""

    def classify_request(
            self,
            question: str,
            history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Phân loại yêu cầu - UPDATED: Tự xử lý context trong LLM, không dùng context_processor
        """
        try:
            logger.info("-" * 50)
            logger.info("👨‍💼 SUPERVISOR CLASSIFICATION")
            logger.info("-" * 50)
            logger.info(f"📝 Question: '{question}'")
            logger.info(f"📚 History Length: {len(history) if history else 0} messages")


            # Format lịch sử
            history_text = self._format_history(history or [])

            # Tạo prompt - LLM sẽ tự xử lý context
            prompt = self.classification_prompt.format(
                question=question,
                history=history_text,
            )

            # Gọi LLM để phân loại VÀ làm rõ context
            logger.info("🤖 Calling LLM for classification + contextualization...")
            response = llm_model.invoke(prompt)

            # Parse JSON response
            classification = self._parse_classification_response(response)

            # Extract fields
            agent_choice = classification.get("agent", "").upper()
            is_followup = classification.get("is_followup", False)
            contextualized_question = classification.get("contextualized_question", question)
            context_summary = classification.get("context_summary", "")

            # Validate agent choice
            valid_agents = ["FAQ", "CHATTER", "REPORTER", "OTHER"]
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
            logger.info("↩️  Using default: agent=FAQ, no context")
            return {
                "agent": "FAQ",
                "contextualized_question": question,
                "context_summary": "",
                "is_followup": False,
                "reasoning": "Error - default to FAQ"
            }

    def _parse_classification_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response từ LLM"""
        try:
            # Tìm JSON block trong response
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)

                # Ensure required fields
                if "is_followup" not in parsed:
                    parsed["is_followup"] = False
                if "contextualized_question" not in parsed:
                    parsed["contextualized_question"] = ""
                if "context_summary" not in parsed:
                    parsed["context_summary"] = ""
                if "agent" not in parsed:
                    parsed["agent"] = "FAQ"

                return parsed

            # Fallback parsing
            return {
                "agent": "FAQ",
                "is_followup": False,
                "contextualized_question": "",
                "context_summary": "",
                "reasoning": "Parse failed"
            }

        except Exception as e:
            logger.error(f"Error parsing classification response: {e}")
            return {
                "agent": "FAQ",
                "is_followup": False,
                "contextualized_question": "",
                "context_summary": "",
                "reasoning": "Parse error"
            }

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        """Format history thành text, xử lý cả dict và ChatMessage objects"""
        if not history:
            return "Không có lịch sử"

        # Chỉ lấy 3 turn gần nhất (6 messages)
        recent_history = history[-6:] if len(history) > 6 else history

        history_lines = []
        for msg in recent_history:
            # Xử lý cả dict và ChatMessage object
            if isinstance(msg, dict):
                role = "Người dùng" if msg.get("role") == "user" else "Trợ lý"
                content = msg.get("content", "")[:200]
            else:
                # ChatMessage object
                role = "Người dùng" if getattr(msg, "role", "") == "user" else "Trợ lý"
                content = getattr(msg, "content", "")[:200]

            if content:
                history_lines.append(f"{role}: {content}")

        return "\n".join(history_lines) if history_lines else "Không có lịch sử"