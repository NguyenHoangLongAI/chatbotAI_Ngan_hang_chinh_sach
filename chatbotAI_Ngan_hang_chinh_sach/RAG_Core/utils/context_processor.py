# RAG_Core/utils/context_processor.py - VIETNAMESE CONTEXT

from typing import List, Dict, Any, Optional
from models.llm_model import llm_model
import logging
import re
from collections import deque

logger = logging.getLogger(__name__)


class ContextProcessor:
    """
    Xử lý context với Vietnamese output
    """

    def __init__(self, max_context_length: int = 500, cache_size: int = 10):
        self.max_context_length = max_context_length
        self.cache_size = cache_size
        self.context_cache = deque(maxlen=cache_size)

        # Lightweight patterns
        self.followup_indicators = {
            'pronouns': r'\b(nó|cái đó|điều đó|phần đó|ở đó|ông ấy|bà ấy|anh ấy|chị ấy|họ)\b',
            'ordinals': r'\b(thứ \d+|đầu tiên|thứ hai|thứ ba|cuối cùng|tiếp theo)\b',
            'continuations': r'\b(tiếp theo|còn|thêm|chi tiết|cụ thể|ví dụ|như thế nào|ra sao)\b',
            'short_query': lambda q: len(q.split()) < 5
        }

        # UPDATED: Vietnamese-only prompt
        self.fast_context_prompt = """Bạn là trợ lý AI người Việt Nam. Nhiệm vụ của bạn là làm rõ câu hỏi dựa trên ngữ cảnh cuộc trò chuyện.

NGỮ CẢNH GẦN NHẤT:
{context}

CÂU HỎI CẦN LÀM RÕ:
{question}

YÊU CẦU:
1. Chuyển câu hỏi thành câu độc lập, đầy đủ thông tin
2. Thay thế các đại từ (nó, cái đó, điều đó...) bằng danh từ cụ thể từ ngữ cảnh
3. BẮT BUỘC trả lời BẰNG TIẾNG VIỆT
4. Chỉ trả về câu hỏi đã làm rõ, KHÔNG giải thích

CÂU HỎI ĐÃ LÀM RÕ:"""

    def extract_context_from_history(
            self,
            history: List,
            current_question: str
    ) -> Dict[str, Any]:
        """
        Phân tích context với Vietnamese output
        """
        try:
            # Normalize history
            normalized_history = self._normalize_history(history)

            if not normalized_history:
                return self._create_standalone_result(current_question)

            # Quick check
            is_likely_followup = self._quick_followup_check(current_question)

            if not is_likely_followup:
                return self._create_standalone_result(current_question)

            # Extract context
            recent_context = self._extract_sliding_window(normalized_history)

            # Check cache
            cache_key = f"{current_question}_{recent_context[:100]}"
            cached_result = self._check_cache(cache_key)
            if cached_result:
                logger.info("✅ Using cached context result")
                return cached_result

            # LLM contextualization (Vietnamese)
            contextualized = self._fast_llm_contextualize(
                current_question,
                recent_context
            )

            result = {
                "original_question": current_question,
                "contextualized_question": contextualized,
                "is_followup": True,
                "relevant_context": recent_context[:200]
            }

            # Cache result
            self._add_to_cache(cache_key, result)

            return result

        except Exception as e:
            logger.error(f"Context processing error: {e}")
            return self._create_standalone_result(current_question)

    def _quick_followup_check(self, question: str) -> bool:
        """Lightweight pattern check"""
        q_lower = question.lower().strip()

        if re.search(self.followup_indicators['pronouns'], q_lower):
            return True
        if re.search(self.followup_indicators['ordinals'], q_lower):
            return True
        if re.search(self.followup_indicators['continuations'], q_lower):
            return True
        if self.followup_indicators['short_query'](q_lower):
            return True

        return False

    def _extract_sliding_window(
            self,
            history: List[Dict[str, str]],
            window_size: int = 2
    ) -> str:
        """Sliding window context"""
        if not history:
            return ""

        recent = history[-(window_size * 2):] if len(history) > window_size * 2 else history

        context_parts = []
        for msg in recent:
            role = "Người dùng" if msg["role"] == "user" else "Trợ lý"
            content = msg["content"][:150]
            context_parts.append(f"{role}: {content}")

        return "\n".join(context_parts)

    def _fast_llm_contextualize(
            self,
            question: str,
            context: str
    ) -> str:
        """
        LLM contextualization - FORCE VIETNAMESE OUTPUT
        """
        try:
            if len(context) > self.max_context_length:
                context = context[:self.max_context_length]

            prompt = self.fast_context_prompt.format(
                context=context,
                question=question
            )

            # Invoke LLM với temperature thấp để đảm bảo tuân thủ yêu cầu
            result = llm_model.invoke(
                prompt,
                temperature=0.1,  # Rất thấp để tuân thủ format
                max_tokens=150  # Đủ cho câu hỏi làm rõ
            )

            # Clean result
            result = result.strip()

            # Remove common prefixes nếu LLM thêm vào
            prefixes_to_remove = [
                "Câu hỏi đã làm rõ:",
                "Câu trả lời:",
                "Trả lời:",
                "Answer:",
                "Question:",
            ]

            for prefix in prefixes_to_remove:
                if result.startswith(prefix):
                    result = result[len(prefix):].strip()

            # Validate result
            if not result or len(result.strip()) < 5:
                logger.warning("LLM returned empty or too short result")
                return question

            # Check if result looks like Vietnamese
            if not self._is_vietnamese_text(result):
                logger.warning(f"Result doesn't look Vietnamese: {result[:50]}")
                return question

            logger.info(f"✅ Context summary: {result[:100]}...")
            return result.strip()

        except Exception as e:
            logger.warning(f"LLM contextualization failed: {e}")
            return question

    def _is_vietnamese_text(self, text: str) -> bool:
        """
        Quick check if text contains Vietnamese characters
        """
        vietnamese_chars = 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ'
        vietnamese_chars += vietnamese_chars.upper()

        # Check if text contains Vietnamese diacritics
        has_vietnamese = any(char in vietnamese_chars for char in text)

        # Check if text is mostly Latin script (not English-only)
        has_latin = bool(re.search(r'[a-zA-Z]', text))

        return has_vietnamese or (has_latin and not self._is_english_only(text))

    def _is_english_only(self, text: str) -> bool:
        """
        Check if text is primarily English
        """
        # Common English words that shouldn't appear in Vietnamese context
        english_words = [
            'the', 'is', 'are', 'was', 'were', 'have', 'has', 'had',
            'about', 'more', 'details', 'user', 'wants', 'digital',
            'transformation', 'platforms', 'enterprises', 'for'
        ]

        text_lower = text.lower()
        english_count = sum(1 for word in english_words if f' {word} ' in f' {text_lower} ')

        # If contains 3+ common English words, likely English
        return english_count >= 3

    def _normalize_history(self, history: List) -> List[Dict[str, str]]:
        """Fast normalization"""
        if not history:
            return []

        normalized = []
        for msg in history:
            if isinstance(msg, dict):
                normalized.append({
                    "role": msg.get("role", ""),
                    "content": msg.get("content", "")
                })
            else:
                normalized.append({
                    "role": getattr(msg, "role", ""),
                    "content": getattr(msg, "content", "")
                })

        return normalized

    def _check_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Check cache"""
        for cached_key, cached_result in self.context_cache:
            if cached_key == key:
                return cached_result
        return None

    def _add_to_cache(self, key: str, result: Dict[str, Any]):
        """Add to cache"""
        self.context_cache.append((key, result))

    def _create_standalone_result(self, question: str) -> Dict[str, Any]:
        """Create standalone result"""
        return {
            "original_question": question,
            "contextualized_question": question,
            "is_followup": False,
            "relevant_context": ""
        }


# Global instance
context_processor = ContextProcessor()