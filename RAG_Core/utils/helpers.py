import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, List
import json

logger = logging.getLogger(__name__)


def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure function execution time"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
            raise

    return wrapper


def safe_execute(func: Callable, default_value: Any = None, log_errors: bool = True) -> Any:
    """Safely execute a function with error handling"""
    try:
        return func()
    except Exception as e:
        if log_errors:
            logger.error(f"Error executing {func.__name__ if hasattr(func, '__name__') else 'function'}: {e}")
        return default_value


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""

    # Remove extra whitespace
    text = " ".join(text.split())

    # Remove special characters that might cause issues
    text = text.replace("\x00", "")  # Remove null bytes

    return text.strip()


def format_document_for_display(doc: Dict[str, Any]) -> str:
    """Format document for display"""
    doc_id = doc.get("document_id", "N/A")
    description = doc.get("description", "No description")
    score = doc.get("similarity_score", 0.0)

    return f"[Doc {doc_id}] {description[:100]}{'...' if len(description) > 100 else ''} (Score: {score:.3f})"


def format_references_for_response(references: List[Dict[str, Any]]) -> List[str]:
    """Format references for API response"""
    formatted_refs = []
    for ref in references:
        doc_id = ref.get("document_id", "unknown")
        ref_type = ref.get("type", "DOCUMENT")
        formatted_refs.append(f"{doc_id} ({ref_type})")
    return formatted_refs


def validate_question(question: str) -> tuple[bool, str]:
    """Validate user question"""
    if not question:
        return False, "Câu hỏi không được để trống"

    if len(question.strip()) < 3:
        return False, "Câu hỏi quá ngắn"

    if len(question) > 1000:
        return False, "Câu hỏi quá dài (tối đa 1000 ký tự)"

    return True, ""


def extract_keywords(text: str) -> List[str]:
    """Extract keywords from text (simple implementation)"""
    # Simple keyword extraction - in production, use more sophisticated methods
    stop_words = {
        "là", "của", "và", "có", "được", "trong", "với", "cho", "từ", "về",
        "một", "các", "những", "này", "đó", "khi", "để", "như", "sẽ", "đã",
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"
    }

    words = text.lower().split()
    keywords = [word for word in words if len(word) > 2 and word not in stop_words]

    return list(set(keywords))  # Remove duplicates


def log_interaction(question: str, agent: str, response: str, execution_time: float):
    """Log user interaction for monitoring"""
    interaction_data = {
        "timestamp": time.time(),
        "question": question[:100],  # Truncate for privacy
        "agent": agent,
        "response_length": len(response),
        "execution_time": execution_time
    }

    logger.info(f"Interaction logged: {json.dumps(interaction_data)}")


def calculate_similarity_threshold(base_threshold: float, question_length: int) -> float:
    """Dynamically calculate similarity threshold based on question complexity"""
    # Longer questions might need lower threshold for more flexibility
    if question_length > 50:
        return max(base_threshold - 0.1, 0.5)
    elif question_length < 20:
        return min(base_threshold + 0.1, 0.9)
    else:
        return base_threshold